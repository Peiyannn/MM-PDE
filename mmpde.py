import argparse
import os
import copy
import sys
import time
from datetime import datetime
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from data_creator_2d import GraphCreator_FS_2D
from gnn_2d import MP_PDE_Solver_2D
from models_cnn import BaseCNN
from train_helper_2d import *
from PDEs import *
from mesh.dmm_model import DMM
from interpolate import ItpNet
from torch.utils.tensorboard import SummaryWriter  


def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f'logs'):
        os.mkdir(f'logs')
    if not os.path.exists(f'models'):
        os.mkdir(f'models')

def criterion(x, y):
    mse = torch.nn.MSELoss()
    # return torch.sqrt(mse(x, y)) / torch.sqrt(mse(x, torch.zeros_like(x).to(x.device)))
    return mse(x, y)

def train(args: argparse,
          pde: PDE,
          epoch: int,
          model: torch.nn.Module,
          model_b: torch.nn.Module,
          itp_model: torch.nn.Module,
          mesh_model: torch.nn.Module,
          optimizer: torch.optim,
          optimizer2: torch.optim,
          loader: DataLoader,
          graph_creator: GraphCreator_FS_2D,
          criterion: torch.nn.modules.loss,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        model_b (torch.nn.Module): branch neural network PDE solver
        itp_model (torch.nn.Module): neural network for interpolation
        mesh_model (torch.nn.Module): moving mesh operator
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator_FS_2D): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    model.train()
    if model_b != None:
        model_b.train()

    # Sample number of unrolling steps during training (pushforward trick)
    # Default is to unroll zero steps in the first epoch and then increase the max amount of unrolling steps per additional epoch.
    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    # Loop over every epoch as often as the number of timesteps in one trajectory.
    # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is covered.

    itp_losses = []
    if mesh_model != None:
        itp_model.train()
        if epoch == 0:
            for i in range(graph_creator.t_res):
                losses = training_itp(itp_model, mesh_model, unrolling, 128 * args.batch_size, optimizer, optimizer2, loader, graph_creator, criterion, device)
                if(i % args.print_interval == 0):
                    print(f'Training ItpNet Loss (progress: {i / graph_creator.t_res:.2f}): {torch.mean(losses)}')
            itp_losses.append(torch.mean(losses))
    train_losses = []
    for i in range(graph_creator.t_res):
        losses = training_loop_branch(model, model_b, itp_model, mesh_model, unrolling, args.batch_size, optimizer, optimizer2, loader, graph_creator, criterion, device)
        if(i % args.print_interval == 0):
            print(f'Training Loss (progress: {i / graph_creator.t_res:.2f}): {torch.mean(losses)}')
        train_losses.append(torch.mean(losses))

    return train_losses, itp_losses

def test(args: argparse,
         pde: PDE,
         model: torch.nn.Module,
         model_b: torch.nn.Module,
         itp_model: torch.nn.Module,
         mesh_model: torch.nn.Module,
         loader: DataLoader,
         graph_creator: GraphCreator_FS_2D,
         criterion: torch.nn.modules.loss,
         device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    Test routine
    Both step wise and unrolled forward losses are computed
    and compared against low resolution solvers
    step wise = loss for one neural network forward pass at certain timepoints
    unrolled forward loss = unrolling of the whole trajectory
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        model_b (torch.nn.Module): branch neural network PDE solver
        itp_model (torch.nn.Module): neural network for interpolation
        mesh_model (torch.nn.Module): moving mesh operator
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator_FS_2D): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled forward loss
    """
    model.eval()
    if model_b != None:
        model_b.eval()
    if itp_model != None:
        itp_model.eval()

   # first we check the losses for different timesteps (one forward prediction array!)
    steps = [t for t in range(graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
    timestep_loss = test_timestep_losses(model=model,
                                  model_b=model_b,
                                  itp_model=itp_model,
                                  mesh_model=mesh_model,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  device=device)

    return timestep_loss


def main(args: argparse):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    check_directory() 

    base_resolution = args.base_resolution
    if args.experiment == 'cy':
        data = torch.load('mesh/data/cylinder_rot_tri')
        data[:, :, :, :2] *= 2
        pde = cy(ori_grid=data[0, 0, :, :2], device=device)
        u = data[:, 10:, :, 2]
        u_train = u[:80]
        u_test = u[80:]
    elif args.experiment == 'burgers':
        pde = burgers(device=device)
        u = torch.tensor(np.load('mesh/data/burgers_192.npy'), dtype=torch.float)[:, :, ::int(192/base_resolution[1]), ::int(192/base_resolution[2])]
        u_train = u[:80]
        u_test = u[80:]
    else:
        raise Exception("Wrong experiment")


    # Equation specific parameters
    pde.grid_size = base_resolution
    pde.movingmesh_grid_size = base_resolution
    pde.ori_grid_size = base_resolution

    if args.model == 'BaseCNN':
        args.moving_mesh = False
    if args.moving_mesh == False:
        itp_model = None
        mesh_model = None
    else:
        if args.experiment == 'cy':
            itp_model = ItpNet(pde.ori_grid_size[1], None, args.itpnet_node1, args.itpnet_node2, args.res_cut_node).to(device)
            checkpoint = torch.load('cy_checkpoint', \
                                    map_location=lambda storage, loc: storage)
            mesh_model = DMM(mode='graph', grid = data[0, 0, :, :2].to(device), branch_layer = checkpoint['args'].branch_layers, trunk_layer = [2] + checkpoint['args'].trunk_layers,\
                            out_layer = checkpoint['args'].out_layers).to(device)
        elif args.experiment == 'burgers':
            itp_model = ItpNet(pde.ori_grid_size[-2], pde.ori_grid_size[-1], args.itpnet_node1, args.itpnet_node2, args.res_cut_node).to(device)
            if base_resolution[1] == 48:
                checkpoint = torch.load('burgers_checkpoint', map_location=lambda storage, loc: storage)
            mesh_model = DMM(s=pde.movingmesh_grid_size[-1], mode='array', branch_layer = checkpoint['args'].branch_layers, trunk_layer = [2] + checkpoint['args'].trunk_layers, out_layer = checkpoint['args'].out_layers).to(device)
        mesh_model.load_state_dict(checkpoint['model_state_dict'])
        mesh_model.eval()

    try:
        train_dataset = TensorDataset(u_train, u_train)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4)
        test_dataset = TensorDataset(u_test, u_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)
    except:
        raise Exception("Datasets could not be loaded properly")

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}-{dateTimeObj.date().day}-{dateTimeObj.time().hour}-{dateTimeObj.time().minute}-{dateTimeObj.time().second}'

    save_path = f'{args.experiment}_{args.model}_{args.batch_size}_mesh{args.moving_mesh}_xresolution{args.base_resolution[0]}-{args.base_resolution[1]}_lr{args.lr}_n{args.neighbors}_{args.connect_edge}_tw{args.time_window}_unrolling{args.unrolling}_time{datetime.now()}'
    log_dir = os.path.join("logs", save_path)  
    writer = SummaryWriter(log_dir)  

    save_path = f'models/{args.model}_{pde}_{args.experiment}_mesh{args.moving_mesh}_xresolution{args.base_resolution[0]}-{args.base_resolution[1]}_n{args.neighbors}_{args.connect_edge}_tw{args.time_window}_unrolling{args.unrolling}_time{timestring}.pt'
    print(f'Training on dataset of {args.experiment}')
    print(device)
    print(save_path)
    
    # Equation specific input variables
    eq_variables = {}
    graph_creator = GraphCreator_FS_2D(pde=pde,
                                 neighbors=args.neighbors,
                                 connect_edge=args.connect_edge,
                                 time_window=args.time_window,
                                 t_resolution=args.base_resolution[0],
                                 ).to(device)
 
    if args.model == 'GNN':
        model = MP_PDE_Solver_2D(pde=pde,
                                time_window=graph_creator.tw,
                                eq_variables=eq_variables,
                                ).to(device)
        if args.moving_mesh == True:
            model_b = MP_PDE_Solver_2D(pde=pde,
                                time_window=graph_creator.tw,
                                eq_variables=eq_variables).to(device)
        else:
            model_b = None
    elif args.model == 'BaseCNN':
        model = BaseCNN(pde=pde,
                        hidden_channels=args.hidden_channels,
                        time_window=args.time_window).to(device)
        model_b = None
    else:
        raise Exception("Wrong model specified")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    if mesh_model != None:
        model_parameters = filter(lambda p: p.requires_grad, model_b.parameters())
        params2 = sum([np.prod(p.size()) for p in model_parameters])
        model_parameters = filter(lambda p: p.requires_grad, itp_model.parameters())
        params3 = sum([np.prod(p.size()) for p in model_parameters])
        params = params + params2 + params3
    print(f'Number of parameters: {params}')

    # Optimizer
    if mesh_model != None:
        optimizer = optim.AdamW([{'params': model.parameters()},
                                {'params': model_b.parameters()},
                                {'params': itp_model.parameters()}], lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.unrolling, 30, 50, 70], gamma=args.lr_decay)
    optimizer2 = None

    # Training loop
    test_loss = 10e30
    train_losses = []
    itp_losses = []
    test_timestep_losses = []
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        train_loss, itp_loss = train(args, pde, epoch, model, model_b, itp_model, mesh_model, optimizer, optimizer2, train_loader, graph_creator, criterion, device=device)
        train_losses.append(train_loss)
        itp_losses.append(itp_loss)
        print("Testing:")
        timestep_loss = test(args, pde, model, model_b, itp_model, mesh_model, test_loader, graph_creator, criterion, device=device)
        test_timestep_losses.append(timestep_loss)
        
        # Save model
        if args.moving_mesh == True:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_b_state_dict': model_b.state_dict(),
                        'mesh_model_state_dict': mesh_model.state_dict(),
                        'itp_model_state_dict': itp_model.state_dict(),
                        'args': args,
                        'train_losses': train_losses,
                        'itp_losses': itp_losses,
                        'test_timestep_losses': test_timestep_losses,
                        }, save_path)
        else:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'args': args,
                        'train_losses': train_losses,
                        'itp_losses': itp_losses,
                        'test_timestep_losses': test_timestep_losses,
                        }, save_path)
        print(f"Saved model at {save_path}\n")

        scheduler.step()

        for k, l in enumerate(train_loss):
            writer.add_scalar("train loss", l.item(), k+epoch*len(train_loss))  
        writer.add_scalar("test loss", timestep_loss.item(), epoch)  

    print(f"Test loss: {test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PDE solver')

    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Used device')
    # PDE
    parser.add_argument('--experiment', type=str, default='burgers',
                        help='Experiment for PDE solver should be trained: [burgers, cy]')

    # Model
    parser.add_argument('--model', type=str, default='GNN',
                        help='Model used as PDE solver: [GNN, BaseCNN]')
    parser.add_argument('--moving_mesh', type=eval, default=True,
            help='Use moving mesh method')

    # Model parameters
    parser.add_argument('--itpnet_node1', type=lambda s: [int(item) for item in s.split(',')],
            default=[128, 64], help="nodes of ItpNet1")
    parser.add_argument('--itpnet_node2', type=lambda s: [int(item) for item in s.split(',')],
            default=[128, 64], help="nodes of ItpNet2")
    parser.add_argument('--res_cut_node', type=lambda s: [int(item) for item in s.split(',')],
            default=[1, 4, 16, 4, 1], help="nodes of residual cut network")
    parser.add_argument('--hidden_channels', type=int,
            default=40, help="number of hidden channels of CNN")
    parser.add_argument('--batch_size', type=int, default=6,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=80,
            help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-3,
            help='Learning rate')
    parser.add_argument('--lr_decay', type=float,
                        default=0.4, help='multistep lr decay')

    # Base resolution and super resolution
    parser.add_argument('--base_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[31, 48, 48], help="PDE base resolution on which network is applied")
    parser.add_argument('--neighbors', type=int,
                        default=35, help="Neighbors to be considered in GNN solver")
    parser.add_argument('--connect_edge', type=str, default='knn',
                        help='The way to connect edge: [knn, radius]')
    parser.add_argument('--time_window', type=int,
                        default=1, help="Time steps to be considered in GNN solver")
    parser.add_argument('--unrolling', type=int,
                        default=0, help="Unrolling which proceeds with each epoch")

    # Misc
    parser.add_argument('--print_interval', type=int, default=2,
            help='Interval between print statements')
    parser.add_argument('--log', type=eval, default=True,
            help='pip the output to log file')

    args = parser.parse_args()
    print(args)
    main(args)
