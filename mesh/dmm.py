import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import datetime

from dmm_model import DMM
from dmm_utils import *


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    
    parser.add_argument('--experiment', default='burgers', type=str, help='experiment: burgers | cy')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='used device')
    parser.add_argument('--sub_u', default=4, type=int, help='subsample number when sampling')
    parser.add_argument('--train_sample_grid', default=5000, type=int, help='number of training grids per u') # 5000, 1500
    parser.add_argument('--test_grid_size', default=[6, 10, 20, 40], type=int, help='grid size for plotting')
    parser.add_argument('--branch_layers', type=lambda s: [int(item) for item in s.split(',')], default=7, metavar='N',\
                         help='number of hidden nodes of branch network') # 7, [4, 3]
    parser.add_argument('--trunk_layers', type=lambda s: [int(item) for item in s.split(',')], default=[32, 512], metavar='N',\
                         help='number of hidden nodes of trunk network') # [32, 512], [16, 512]
    parser.add_argument('--out_layers', type=lambda s: [int(item) for item in s.split(',')], default=[1024, 512, 1], metavar='N',\
                         help='number of hidden nodes of decoder network') 
    parser.add_argument('--bound_constraint', default='soft', type=str, help='constraint of boundary condition: soft | hard')
    parser.add_argument('--batch_size_x_adam', default=120, type=int, help='batch size of training grids per u') # 120
    parser.add_argument('--batch_size_u_adam', default=160, type=int, help='batch size of u (should be divisible by sub_u)') # 160
    parser.add_argument('--batch_size_x_lbfgs', default=100, type=int, help='batch size') # 100
    parser.add_argument('--batch_size_u_lbfgs', default=120, type=int, help='batch size') # 120
    
    parser.add_argument('--rf', default=True, type=eval, help='random feature: True | False')
    parser.add_argument('--rf_opt_alg', default='BFGS', type=str, help='optimization algorithm of random feature method: BFGS | Newton')
    parser.add_argument('--convex_rel', default=0.00, type=float, help='hyperparameter of convex relaxation') 
    parser.add_argument('--batch_size_x_rf', default=16, type=int, help='batch size') # 100
    parser.add_argument('--batch_size_u_rf', default=20, type=int, help='batch size') # 120
    parser.add_argument('--loss_bound_rf', default=True, type=eval, help='bound constraint of random feature method: True | False')
    parser.add_argument('--max_iter', default=300, type=int, help='max iteration of rf algorithm') 
    parser.add_argument('--epochs_adam', default=150, type=int, help='number of epochs of Adam optimizer') # 200
    parser.add_argument('--epochs_lbfgs', default=0, type=int, help='number of epochs of LBFGS optimizer') # 25, 0
    parser.add_argument('--epochs_rf', default=5, type=int, help='number of epochs of random feature')
    parser.add_argument('--lr_adam', default=2e-4, type=float, help='learning rate') # 2e-4
    parser.add_argument('--lr_lbfgs', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--gamma_adam', default=0.2, type=float, help='gamma of Adam optimizer')
    parser.add_argument('--gamma_lbfgs', default=0.2, type=float, help='gamma of LBFGS optimizer')
    parser.add_argument('--loss_weight0', default=1, type=float, help='weight of loss_in') 
    parser.add_argument('--loss_weight1', default=1000, type=float, help='weight of loss_bound') 
    parser.add_argument('--loss_weight2', default=1, type=float, help='weight of loss_convex')
    parser.add_argument('--loss_convex', default=True, type=eval, help='convex constraint: True | False')

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = get_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    if args.experiment == 'burgers':
        ori_u = torch.tensor(np.load('data/burgers_192.npy'), dtype=torch.float).to(device).reshape(-1, 192, 192)
        u = torch.tensor(np.load('data/burgers_192.npy'), dtype=torch.float)[:80, :, ::args.sub_u, ::args.sub_u].reshape(-1, int(192/args.sub_u), int(192/args.sub_u))
        test_u = torch.tensor(np.load('data/burgers_192.npy'), dtype=torch.float)[80:, :, ::args.sub_u, ::args.sub_u].reshape(-1, int(192/args.sub_u), int(192/args.sub_u))
    elif args.experiment == 'cy':
        ori_u = torch.load('data/cylinder_rot_tri')
        u = torch.load('data/cylinder_rot_tri')[:80, 10:].reshape(-1, ori_u.shape[-2], 5)
        # scale to a 1*1 square
        u[:, :, :2] *= 2
        test_u = torch.load('data/cylinder_rot_tri')[80:, 10:].reshape(-1, ori_u.shape[-2], 5)
        test_u[:, :, :2] *= 2

    if args.experiment == 'burgers':
        mkdir('burgers')
        model = DMM(s=u.shape[-1], mode='array', branch_layer = args.branch_layers, trunk_layer = [2] + args.trunk_layers, out_layer = args.out_layers).to(device)
    elif args.experiment == 'cy':
        mkdir('cy')
        model = DMM(mode='graph', grid = u[0, :, :2].to(device), branch_layer = args.branch_layers, trunk_layer = [2] + args.trunk_layers, out_layer = args.out_layers).to(device)
    
    print('Train moving mesh operator:')
    model, loss_in, loss_bound, loss_convex, test_equ_loss, test_equ_max, test_equ_min, test_equ_mid,\
    train_std_list, train_minmax_list, test_std_list, test_minmax_list, itp_list1, itp_list2, logs_txt\
        = train_MA_res(ori_u, u, test_u, args, model, init_mesh=False, n_epoch_adam=args.epochs_adam, n_epoch_lbfgs=args.epochs_lbfgs, device=device)
    print('Finish!')


    # plot mesh
    if args.experiment == 'burgers':
        for s in args.test_grid_size: 
            fig, axes = plt.subplots(1, 5, figsize=(20, 3), dpi=500)
            fig, axes = plot_mesh_res(s, u, model, fig, axes, args, device)
            save_path = "{}/{}_{}_bound{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png"\
                    .format(args.experiment, datetime.now(), args.rf, args.loss_bound_rf, args.epochs_rf, args.max_iter, args.sub_u,\
                    args.epochs_adam, args.batch_size_u_adam, args.batch_size_x_adam, args.loss_weight1, args.train_sample_grid, args.branch_layers, args.lr_adam, args.trunk_layers, s)
            plt.savefig(save_path)
            print(save_path)
    elif args.experiment == 'cy':
        for s in args.test_grid_size: 
            fig, axes = plt.subplots(1, 5, figsize=(20, 3), dpi=500)
            fig, axes = plot_mesh_res_tri_s(s, u, model, fig, axes, args, device)
            save_path = "{}/{}_{}_bound{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png"\
                    .format(args.experiment, datetime.now(), args.rf, args.loss_bound_rf, args.epochs_rf, args.max_iter, args.sub_u,\
                    args.epochs_adam, args.batch_size_u_adam, args.batch_size_x_adam, args.loss_weight1, args.train_sample_grid, args.branch_layers, args.lr_adam, args.trunk_layers, s)
            plt.savefig(save_path)
            print(save_path)
        fig, axes = plt.subplots(1, 5, figsize=(20, 3), dpi=500)
        fig, axes = plot_mesh_res_tri(u, model, fig, axes, args, device)
        savepath = "{}/{}_{}_bound{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png"\
                .format(args.experiment, datetime.now(), args.rf, args.loss_bound_rf, args.epochs_rf, args.max_iter, args.sub_u, args.epochs_adam, \
                args.batch_size_u_adam, args.batch_size_x_adam, args.loss_weight1, args.train_sample_grid, args.branch_layers, args.lr_adam, args.trunk_layers)
        plt.savefig(savepath)
        print(savepath)

    with open("{}/{}_{}_bound_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(args.experiment, datetime.now(), args.rf, args.loss_bound_rf, args.epochs_rf, args.max_iter, args.sub_u, args.epochs_adam, args.batch_size_u_adam, args.batch_size_x_adam, args.loss_weight1, args.train_sample_grid, args.branch_layers, args.lr_adam, args.trunk_layers),"w") as f:
        f.write('\n'.join(logs_txt))
