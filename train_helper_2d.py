import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from data_creator_2d import GraphCreator_FS_2D
# from PDEs import *


def training_itp(itp_model: torch.nn.Module,
                  mesh_model: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer: torch.optim,
                  optimizer2: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator_FS_2D,
                  criterion: torch.nn.modules.loss,
                  device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory
    Args:
        mesh_model (torch.nn.Module): moving mesh operator
        unrolling (list): list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator_FS_2D): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """

    losses = []
    for (u_base, u_super) in loader:
        optimizer.zero_grad()
        if optimizer2 != None:
            optimizer2.zero_grad()
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                  graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u_super, random_steps)

        graph = graph_creator.create_graph(itp_model, data, labels, random_steps, device, mesh_model)

        itp_u = graph.x
        u_uni = graph_creator.interpolate_pred(itp_model, itp_u, graph, data, device)
        # data_uni = graph_creator.interpolate_label(data, device)
        data = data.to(device)
        loss = criterion(u_uni, data.reshape(-1, 1))

        loss.backward()
        losses.append(loss.detach() / 2)
        optimizer.step()
        if optimizer2 != None:
            optimizer2.step()

    losses = torch.stack(losses)
    return losses


def training_loop_branch(model: torch.nn.Module,
                  model_b: torch.nn.Module,
                  itp_model: torch.nn.Module,
                  mesh_model: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer: torch.optim,
                  optimizer2: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator_FS_2D,
                  criterion: torch.nn.modules.loss,
                  device: torch.cuda.device="cpu") -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory
    Args:
        model (torch.nn.Module): neural network PDE solver
        model_b (torch.nn.Module): branch neural network PDE solver
        mesh_model (torch.nn.Module): moving mesh operator
        unrolling (list): list of different unrolling steps for each batch entry
        batc-h_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator_FS_2D): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """

    losses = []
    for idx, (u_base, u_super) in enumerate(loader):
        optimizer.zero_grad()
        if optimizer2 != None:
            optimizer2.zero_grad()
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                  graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u_super, random_steps)

        if f'{model}' == 'GNN':
            graph = graph_creator.create_graph(itp_model, data, labels, random_steps, device, mesh_model)
            graph_uni = graph_creator.create_graph(itp_model, data, labels, random_steps, device, None)
        else:
            data, labels = data.to(device), labels.to(device)

        # Unrolling of the equation which serves as input at the current step
        if f'{model}' == 'GNN':
            if mesh_model != None:
                pred = graph_creator.interpolate_pred(itp_model, model_b(graph), graph, data, device) + model(graph_uni)
            else:
                pred = model(graph_uni)
            # labels_uni = graph_creator.interpolate_label(labels, device)
            labels = labels.to(device)
            loss = criterion(pred, labels.reshape(-1, 1))
        else:
            pred = model(data)
            loss = criterion(pred, labels.squeeze())

        loss.backward()
        losses.append(loss.detach())
        optimizer.step()
        if optimizer2 != None:
            if idx % 1 == 0:
                optimizer2.step()

    losses = torch.stack(losses)
    return losses


def test_timestep_losses(model: torch.nn.Module,
                         model_b: torch.nn.Module,
                         itp_model: torch.nn.Module,
                         mesh_model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator_FS_2D,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> None:
    """
    Loss for one neural network forward pass at certain timepoints on the validation/test datasets
    Args:
        model (torch.nn.Module): neural network PDE solver
        model_b (torch.nn.Module): branch neural network PDE solver
        mesh_model (torch.nn.Module): moving mesh operator
        steps (list): input list of possible starting (time) points
        batch_size (int): batch size
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator_FS_2D): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """

    losses_t = []
    losses_uni_t = []
    for step in steps:

        if (step != graph_creator.tw and step % graph_creator.tw != 0):
            continue

        losses = []
        for (u_base, u_super) in loader:
            same_steps = [step]*batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)
            if f'{model}' == 'GNN':
                if mesh_model != None:
                    graph = graph_creator.create_graph(itp_model, data, labels, same_steps, device, mesh_model)
                graph_uni = graph_creator.create_graph(itp_model, data, labels, same_steps, device, None)
            with torch.no_grad():
                if f'{model}' == 'GNN':
                    if mesh_model != None:
                        pred = graph_creator.interpolate_pred(itp_model, model_b(graph), graph, data, device) + model(graph_uni)
                    else:
                        pred = model(graph_uni)
                    labels = labels.to(device)
                    loss = criterion(pred, labels.reshape(-1, 1)) 
                else:
                    data, labels = data.to(device), labels.to(device)
                    pred = model(data)
                    loss = criterion(pred, labels.squeeze())
                losses.append(loss)

        losses = torch.stack(losses)
        losses_t.append(torch.mean(losses))
        if step % 2 == 1:
            print(f'Step {step}, time step loss {torch.mean(losses)}')

    losses_t = torch.stack(losses_t)
    print(f'Mean Timestep Test Error: {torch.mean(losses_t)}')

    return torch.mean(losses_t)

