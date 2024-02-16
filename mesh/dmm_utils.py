import torch
import numpy as np
from torch import nn
import scipy
from scipy.integrate import dblquad
from scipy.spatial import Delaunay  
import functools
import sympy as sp
from sklearn.neighbors import NearestNeighbors  
from torch.utils.data import DataLoader,TensorDataset
import os
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
from datetime import datetime
from functools import reduce
import operator
from torchmin import minimize # pip install pytorch-minimize

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c


def sample_train_data(u, nx, nu, device):
    grid = torch.tensor(np.random.uniform(0, 1, (nu, 40 * nx, 2)), dtype=torch.float).to(device)
    u_idx = np.random.choice(a=u.shape[0], size=nu, replace=True)
    u = u[u_idx].to(device)
    ux = diff_x(u) * (u.shape[-1] - 1)
    uy = diff_y(u) * (u.shape[-1] - 1)
    alpha = torch.sum((torch.abs(ux)**2 + torch.abs(uy)**2)**(1/2), dim=(-2, -1)) / (u.shape[-1]-1)**2
    m = monitor(alpha.unsqueeze(-1).unsqueeze(-1).repeat(1, ux.shape[-1], ux.shape[-1]), ux, uy)
    RHS = torch.sum(m, dim=(-2, -1)) / (u.shape[-1]-1)**2
    m_normalized = m 
    
    all_p = []
    sub_nu = 4
    N = int(nu / sub_nu)
    for i in range(sub_nu):
        all_p.append(interpolate(m_normalized[i * N : (i+1) * N].unsqueeze(1).repeat(1, grid.shape[1], 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]),\
                grid[i * N : (i+1) * N, :, [0]].reshape(-1, 1), grid[i * N : (i+1) * N, :, [1]].reshape(-1, 1))[:, 0].cpu().numpy())
    all_p = np.array(all_p).flatten().reshape(nu, grid.shape[1])

    grid_choosed = torch.zeros(nu, nx, 2).to(device)
    for i in range(nu):
        p = all_p[i] / np.sum(all_p[i])
        idx = np.random.choice(a=grid.shape[1], size=nx, replace=False, p=p)
        grid_choosed[i] = grid[i, idx]
    
    return u, ux, uy, alpha, m, RHS, grid_choosed.reshape(-1, 2)

def sample_train_data_bound(u, nx, nu, device):
    u_idx = np.random.choice(a=u.shape[0], size=4*nu, replace=True)
    u = u[u_idx].to(device)
    ux = diff_x(u) * (u.shape[-1] - 1)
    uy = diff_y(u) * (u.shape[-1] - 1)
    alpha = torch.sum((torch.abs(ux)**2 + torch.abs(uy)**2)**(1/2), dim=(-2, -1)) / (u.shape[-1]-1)**2
    m = monitor(alpha.unsqueeze(-1).unsqueeze(-1).repeat(1, ux.shape[-1], ux.shape[-1]), ux, uy)
    RHS = torch.sum(m, dim=(-2, -1)) / (u.shape[-1]-1)**2

    n = int(nx/4)
    # n = nx
    bound1 = []
    bound2 = []
    bound3 = []
    bound4 = []
    X1 = np.linspace(0, 1, n)
    for i in X1:
        data1 = [0, i]
        bound1.append(data1)
    X2 = np.linspace(0, 1, n)
    for i in X2:
        data2 = [1, i]
        bound2.append(data2)
    X3 = np.linspace(0, 1, n)
    for i in X3:
        data3 = [i, 0]
        bound3.append(data3)
    X4 = np.linspace(0, 1, n)
    for i in X4:
        data4 = [i, 1]
        bound4.append(data4)
    bound1 = torch.tensor(bound1, dtype=torch.float).to(device)
    bound2 = torch.tensor(bound2, dtype=torch.float).to(device)
    bound3 = torch.tensor(bound3, dtype=torch.float).to(device)
    bound4 = torch.tensor(bound4, dtype=torch.float).to(device)

    bound1_u = u[:nu]
    bound2_u = u[nu:2*nu]
    bound3_u = u[2*nu:3*nu]
    bound4_u = u[3*nu:4*nu]

    bound1_m = m[:nu]
    bound2_m = m[nu:2*nu]
    bound3_m = m[2*nu:3*nu]
    bound4_m = m[3*nu:4*nu]

    return bound1.repeat(nu, 1, 1).reshape(-1, 2), bound2.repeat(nu, 1, 1).reshape(-1, 2), bound3.repeat(nu, 1, 1).reshape(-1, 2),\
        bound4.repeat(nu, 1, 1).reshape(-1, 2), bound1_u, bound2_u, bound3_u, bound4_u, bound1_m, bound2_m, bound3_m, bound4_m


def sample_train_data_tri(all_u, nx, nu, device):
    u = all_u[:, :, 2].to(device)
    grid = torch.tensor(np.random.uniform(0, 1, (nu, 40 * nx, 2)), dtype=torch.float).to(device)
    u_idx = np.random.choice(a=u.shape[0], size=nu, replace=True)
    u = u[u_idx].to(device)
    ori_mesh_x = all_u[u_idx, :, 0].unsqueeze(-1).to(device)
    ori_mesh_y = all_u[u_idx, :, 1].unsqueeze(-1).to(device)

    n = int(np.sqrt(u.shape[-1]))
    uni_grid = torch.tensor(np.array(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))), dtype=torch.float)\
                    .reshape(1, 2, -1).repeat(nu, 1, 1).permute(0, 2, 1).to(device)
    
    all_p = []
    uni_ux = torch.Tensor().to(device)
    uni_uy = torch.Tensor().to(device)
    alpha = torch.Tensor().to(device)
    uni_m = torch.Tensor().to(device)
    RHS = torch.Tensor().to(device)
    sub_nu = 10
    N = int(nu / sub_nu)
    for i in range(sub_nu):

        # alpha
        x1_ = uni_grid[i * N : (i+1) * N, :, [0]].reshape(-1, 1)
        x2_ = uni_grid[i * N : (i+1) * N, :, [1]].reshape(-1, 1)
        x1_.requires_grad = True
        x2_.requires_grad = True
        u_ = interpolate_tri(u[i*N : (i+1)*N].unsqueeze(1).repeat(1, n**2, 1).reshape(-1, u.shape[-1]), \
                        ori_mesh_x[i*N : (i+1)*N].unsqueeze(1).repeat(1, n**2, 1, 1).reshape(-1, u.shape[-1], 1), \
                        ori_mesh_y[i*N : (i+1)*N].unsqueeze(1).repeat(1, n**2, 1, 1).reshape(-1, u.shape[-1], 1), \
                        x1_.unsqueeze(1).repeat(1, u.shape[-1], 1), x2_.unsqueeze(1).repeat(1, u.shape[-1], 1))
        w = torch.ones(u_.shape).to(device)
        uni_ux_ = torch.autograd.grad(u_, x1_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0].reshape(N, n, n)
        uni_uy_ = torch.autograd.grad(u_, x2_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0].reshape(N, n, n)
        alpha_ = torch.sum((torch.abs(uni_ux_)**2 + torch.abs(uni_uy_)**2)**(1/2), dim=(-2, -1)) / (n-1)**2
        uni_m_ = monitor(alpha_.unsqueeze(-1).unsqueeze(-1).repeat(1, n, n), uni_ux_, uni_uy_)
        RHS_ = torch.sum(uni_m_, dim=(-2, -1)) / (n-1)**2
        
        uni_ux = torch.cat((uni_ux, uni_ux_), )
        uni_uy = torch.cat((uni_uy, uni_uy_), )
        alpha = torch.cat((alpha, alpha_), )
        uni_m = torch.cat((uni_m, uni_m_), )
        RHS = torch.cat((RHS, RHS_), )

        # ux, uy
        x1_ = grid[i * N : (i+1) * N, :, [0]].reshape(-1, 1)
        x2_ = grid[i * N : (i+1) * N, :, [1]].reshape(-1, 1)
        ux_ = interpolate(uni_ux_.unsqueeze(1).repeat(1, grid.shape[1], 1, 1).reshape(-1, n, n), x1_, x2_).reshape(N, grid.shape[1])
        uy_ = interpolate(uni_uy_.unsqueeze(1).repeat(1, grid.shape[1], 1, 1).reshape(-1, n, n), x1_, x2_).reshape(N, grid.shape[1])
        m = monitor(alpha_.unsqueeze(-1).repeat(1, grid.shape[1]), ux_, uy_)

        all_p.append(m.detach().cpu().numpy())
    all_p = np.array(all_p).flatten().reshape(nu, grid.shape[1])

    grid_choosed = torch.zeros(nu, nx, 2).to(device)
    for i in range(nu):
        p = all_p[i] / np.sum(all_p[i])
        idx = np.random.choice(a=grid.shape[1], size=nx, replace=False, p=p)
        grid_choosed[i] = grid[i, idx]
    # plot_grid(grid_choosed[0], uni_m[0])
    
    return u, uni_ux, uni_uy, alpha, uni_m, RHS, grid_choosed.reshape(-1, 2)

def sample_train_data_bound_tri(u, nx, nu, device):
    u_idx = np.random.choice(a=u.shape[0], size=4*nu, replace=True)
    u = u[u_idx, :, 2].to(device)

    n = int(nx/4)
    # n = nx
    bound1 = []
    bound2 = []
    bound3 = []
    bound4 = []
    X1 = np.linspace(0, 1, n)
    for i in X1:
        data1 = [0, i]
        bound1.append(data1)
    X2 = np.linspace(0, 1, n)
    for i in X2:
        data2 = [1, i]
        bound2.append(data2)
    X3 = np.linspace(0, 1, n)
    for i in X3:
        data3 = [i, 0]
        bound3.append(data3)
    X4 = np.linspace(0, 1, n)
    for i in X4:
        data4 = [i, 1]
        bound4.append(data4)
    bound1 = torch.tensor(bound1, dtype=torch.float).to(device)
    bound2 = torch.tensor(bound2, dtype=torch.float).to(device)
    bound3 = torch.tensor(bound3, dtype=torch.float).to(device)
    bound4 = torch.tensor(bound4, dtype=torch.float).to(device)

    bound1_u = u[:nu]
    bound2_u = u[nu:2*nu]
    bound3_u = u[2*nu:3*nu]
    bound4_u = u[3*nu:4*nu]

    return bound1.repeat(nu, 1, 1).reshape(-1, 2), bound2.repeat(nu, 1, 1).reshape(-1, 2), bound3.repeat(nu, 1, 1).reshape(-1, 2),\
        bound4.repeat(nu, 1, 1).reshape(-1, 2), bound1_u, bound2_u, bound3_u, bound4_u


def monitor(alpha, ux, uy):
    return (1 + (torch.abs(ux)**2 + torch.abs(uy)**2) ** (1/2) / (0.01*alpha))

def monitor_np(alpha, ux, uy):
    return (1 + (np.abs(ux)**2 + np.abs(uy)**2) ** (1/2) / (0.01*alpha))

def diff_x(u):
    ux = torch.zeros_like(u)
    ux[:,:-1,:] = torch.diff(u, dim=-2)
    ux[:,-1,:] = ux[:,-2,:]
    return ux

def diff_y(u):
    uy = torch.zeros_like(u)
    uy[:,:,:-1] = torch.diff(u, dim=-1)
    uy[:,:,-1] = uy[:,:,-2]
    return uy

def init_weights(t):
    with torch.no_grad():
        if type(t) == torch.nn.Linear:
            t.weight.normal_(0, 0.02)
            t.bias.normal_(0, 0.02)

def interpolate(u, x, y, n_neighbors = 50):
    """
    u: b*n*n
    x: b*1
    y: b*1
    """

    n = u.shape[-1]
    grid_x = np.linspace(0, 1, n)
    grid_y = np.linspace(0, 1, n)
    grid = torch.tensor(np.array(np.meshgrid(grid_x, grid_y)), dtype=torch.float).reshape(1, 2, -1).permute(0, 2, 1).to(u.device)
    d = -torch.norm(grid.repeat(x.shape[0], 1, 1) - torch.cat((x, y), dim=-1).unsqueeze(1).repeat(1, n*n, 1), dim=-1) * n
    normalize = nn.Softmax(dim=-1)
    weight = normalize(d)  
    interpolated = torch.sum(u.reshape(-1, n**2) * weight, dim=-1).unsqueeze(-1)

    return interpolated # b*1

def interpolate_tri(u, ori_x, ori_y, x, y):
    """
    u: b*n
    ori_x: b*n*1
    ori_y: b*n*1
    x: b*n*1
    y: b*n*1
    """
    n = u.shape[-1]
    grid = torch.cat((ori_x, ori_y), dim=-1)
    d = -torch.norm(grid - torch.cat((x, y), dim=-1), dim=-1) * np.sqrt(n)
    normalize = nn.Softmax(dim=-1)
    weight = normalize(d)  
    # weight = softmax(d, dim=-1)  
    interpolated = torch.sum(u * weight, dim=-1).unsqueeze(-1)

    return interpolated # b*1

def softmax(x, dim):  
    exp_x = torch.exp(x)  
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)  
    softmax_output = exp_x / sum_exp_x  
  
    return softmax_output 


def train_data_loader_p(n, m):
    grid = torch.tensor(np.random.uniform(0, 1, (50 * n, 2)), dtype=torch.float).cuda()
    p = []
    N = int(grid.shape[0] / 5)
    for i in range(5):
        p.append(interpolate(m.reshape(-1, 1).repeat(N, 1).reshape(N, 50, 50), grid[i * N : (i+1) * N, [0]], grid[i * N : (i+1) * N, [1]])[:, 0].cpu().numpy())
    p = np.array(p).flatten()
    p = p / np.sum(p)
    idx = np.random.choice(a=grid.shape[0], size=n, replace=False, p=p)
    # dataset = TensorDataset(grid[idx].cpu())
    return grid[idx].cpu()


def random_feature_torch(weight, convex_rel, second_out, second_out_bound1, second_out_bound2, second_out_bound3, second_out_bound4, branch, branch_bound1, branch_bound2, branch_bound3, branch_bound4, args, m_xi,\
                          alpha, x1, x2, ux, uy, so_x_bound1, so_x_bound2, so_y_bound3, so_y_bound4, so_x, so_y, so_xx, so_yy, so_xy, so_yx, RHS):
    criterion = nn.MSELoss()
    device = second_out.device
    weight = weight.reshape(branch.shape[1], second_out.shape[1])
    # weight_bias = params['weight_bias'].value

    # loss of boundary condition 
    branch_bound1 = branch_bound1.reshape(-1, branch.shape[-1])
    # so_x_one_bound1 = torch.cat((so_x_bound1, torch.ones((second_out_bound1.shape[0], 1)).to(device)), dim=1)
    trunkx_bound1 = torch.matmul(so_x_bound1, weight.T)
    phix_bound1 = torch.sum(branch_bound1 * trunkx_bound1, dim=1).reshape(-1, 1)
    loss_bound1 = criterion(phix_bound1, torch.zeros_like(phix_bound1).to(device))

    branch_bound2 = branch_bound2.reshape(-1, branch.shape[-1])
    # so_x_one_bound2 = torch.cat((so_x_bound2, torch.ones((second_out_bound2.shape[0], 1)).to(device)), dim=1)
    trunkx_bound2 = torch.matmul(so_x_bound2, weight.T)
    phix_bound2 = torch.sum(branch_bound2 * trunkx_bound2, dim=1).reshape(-1, 1)
    loss_bound2 = criterion(phix_bound2, torch.zeros_like(phix_bound2).to(device))

    branch_bound3 = branch_bound3.reshape(-1, branch.shape[-1])
    # so_y_one_bound3 = torch.cat((so_y_bound3, torch.ones((second_out_bound3.shape[0], 1)).to(device)), dim=1)
    trunky_bound3 = torch.matmul(so_y_bound3, weight.T)
    phiy_bound3 = torch.sum(branch_bound3 * trunky_bound3, dim=1).reshape(-1, 1)
    loss_bound3 = criterion(phiy_bound3, torch.zeros_like(phiy_bound3).to(device))

    branch_bound4 = branch_bound4.reshape(-1, branch.shape[-1])
    # so_y_one_bound4 = torch.cat((so_y_bound4, torch.ones((second_out_bound4.shape[0], 1)).to(device)), dim=1)
    trunky_bound4 = torch.matmul(so_y_bound4, weight.T)
    phiy_bound4 = torch.sum(branch_bound4 * trunky_bound4, dim=1).reshape(-1, 1)
    loss_bound4 = criterion(phiy_bound4, torch.zeros_like(phiy_bound4).to(device))

    loss_bound = (loss_bound1 + loss_bound2 + loss_bound3 + loss_bound4) / 4

    # loss in
    branch = branch.reshape(-1, branch.shape[-1])
    trunkx = torch.matmul(so_x, weight.T)
    trunky = torch.matmul(so_y, weight.T)
    trunkxx = torch.matmul(so_xx, weight.T)
    trunkxy = torch.matmul(so_xy, weight.T)
    trunkyx = torch.matmul(so_yx, weight.T)
    trunkyy = torch.matmul(so_yy, weight.T)
    phix = torch.sum(branch * trunkx, dim=1).reshape(-1, 1)
    phiy = torch.sum(branch * trunky, dim=1).reshape(-1, 1)
    phixx = torch.sum(branch * trunkxx, dim=1).reshape(-1, 1)
    phixy = torch.sum(branch * trunkxy, dim=1).reshape(-1, 1)
    phiyx = torch.sum(branch * trunkyx, dim=1).reshape(-1, 1)
    phiyy = torch.sum(branch * trunkyy, dim=1).reshape(-1, 1)
    ux_ = interpolate(ux.unsqueeze(1).repeat(1, args.batch_size_x_rf, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
    uy_ = interpolate(uy.unsqueeze(1).repeat(1, args.batch_size_x_rf, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
    u_xi_x = ux_ * (1 + phixx) + uy_ * phiyx
    u_xi_y = ux_ * phixy + uy_ * (1 + phiyy)
    m_xi = monitor(alpha.unsqueeze(1).repeat(1, args.batch_size_x_rf).reshape(-1, 1), u_xi_x, u_xi_y)
    LHS = m_xi * ((1 + phixx) * (1 + phiyy) - phixy * phiyx)

    loss_in = criterion(LHS / RHS.unsqueeze(1).repeat(1, args.batch_size_x_rf).reshape(-1, 1), torch.ones_like(LHS))
    loss_convex = torch.mean(torch.min(torch.tensor(0).type_as(phixx).to(device), 1 + phixx)**2 + torch.min(torch.tensor(0).type_as(phiyy).to(device), 1 + phiyy)**2)
    # print(loss_in.item(), loss_bound.item())
    return convex_rel * (torch.sum(weight ** 2)) ** 2 + args.loss_weight1 * loss_bound + args.loss_weight0 * loss_in + args.loss_weight2 * loss_convex 


def random_feature_torch2(weight, convex_rel, second_out, second_out_bound1, second_out_bound2, second_out_bound3, second_out_bound4, args,\
                          alpha, x1, x2, ux, uy, so_x_bound1, so_x_bound2, so_y_bound3, so_y_bound4, so_x, so_y, so_xx, so_yy, so_xy, so_yx, RHS):
    criterion = nn.MSELoss()
    device = second_out.device
    weight = weight.reshape(1, second_out.shape[1])

    # loss of boundary condition 
    phix_bound1 = torch.matmul(so_x_bound1, weight.T).reshape(-1, 1)
    loss_bound1 = criterion(phix_bound1, torch.zeros_like(phix_bound1).to(device))

    phix_bound2 = torch.matmul(so_x_bound2, weight.T).reshape(-1, 1)
    loss_bound2 = criterion(phix_bound2, torch.zeros_like(phix_bound2).to(device))

    phiy_bound3 = torch.matmul(so_y_bound3, weight.T).reshape(-1, 1)
    loss_bound3 = criterion(phiy_bound3, torch.zeros_like(phiy_bound3).to(device))

    phiy_bound4 = torch.matmul(so_y_bound4, weight.T).reshape(-1, 1)
    loss_bound4 = criterion(phiy_bound4, torch.zeros_like(phiy_bound4).to(device))

    loss_bound = (loss_bound1 + loss_bound2 + loss_bound3 + loss_bound4) / 4

    # loss in
    phix = torch.matmul(so_x, weight.T).reshape(-1, 1)
    phiy = torch.matmul(so_y, weight.T).reshape(-1, 1)
    phixx = torch.matmul(so_xx, weight.T).reshape(-1, 1)
    phixy = torch.matmul(so_xy, weight.T).reshape(-1, 1)
    phiyx = torch.matmul(so_yx, weight.T).reshape(-1, 1)
    phiyy = torch.matmul(so_yy, weight.T).reshape(-1, 1)
    ux_ = interpolate(ux.unsqueeze(1).repeat(1, args.batch_size_x_rf, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
    uy_ = interpolate(uy.unsqueeze(1).repeat(1, args.batch_size_x_rf, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
    u_xi_x = ux_ * (1 + phixx) + uy_ * phiyx
    u_xi_y = ux_ * phixy + uy_ * (1 + phiyy)
    m_xi = monitor(alpha.unsqueeze(1).repeat(1, args.batch_size_x_rf).reshape(-1, 1), u_xi_x, u_xi_y)
    LHS = m_xi * ((1 + phixx) * (1 + phiyy) - phixy * phiyx)

    loss_in = criterion(LHS / RHS.unsqueeze(1).repeat(1, args.batch_size_x_rf).reshape(-1, 1), torch.ones_like(LHS))
    loss_convex = torch.mean(torch.min(torch.tensor(0).type_as(phixx).to(device), 1 + phixx)**2 + torch.min(torch.tensor(0).type_as(phiyy).to(device), 1 + phiyy)**2)
    return convex_rel * (torch.sum(weight ** 2)) ** 2 + args.loss_weight1 * loss_bound + args.loss_weight0 * loss_in + args.loss_weight2 * loss_convex 


def train_MA_res(ori_u, all_u, test_u, args, model, init_mesh, n_epoch_adam, n_epoch_lbfgs, device):
    # writer = SummaryWriter(logdir='runs')
    logs_txt = []
    logs_txt.append(str(args))

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=args.lr_adam, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler_adam = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam, milestones=[100, 150], gamma=args.gamma_adam)
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=args.lr_lbfgs, tolerance_grad=-1, tolerance_change=-1)
    scheduler_lbfgs = torch.optim.lr_scheduler.MultiStepLR(optimizer_lbfgs, milestones=[75, 125], gamma=args.gamma_lbfgs)
    criterion = nn.MSELoss()
    
    loss_in_list = []
    test_loss_in_list = []
    loss_bound_list = []
    loss_convex_list = []
    test_equ_loss_list = []
    test_equ_max_list = []
    test_equ_min_list = []
    test_equ_mid_list = []
    LHS_list = []
    RHS_list = []
    itp_list1 = []
    itp_list2 = []
    train_std_list = []
    train_minmax_list = []
    test_std_list = []
    test_minmax_list = []

    log_count1 = []
    log_count2 = []
    log_count1.append(0)
    log_count2.append(0)

    epoch = 0
    for epoch in range(1, n_epoch_adam + n_epoch_lbfgs + 1):
        start = datetime.now()
        # Adam
        if epoch < n_epoch_adam + 1:

            for i in range(np.max((1, int(args.train_sample_grid * all_u.shape[0] / (args.batch_size_x_adam * args.batch_size_u_adam))))):
                # sample points
                if args.experiment == 'burgers':
                    u, ux, uy, alpha, m, RHS, x = sample_train_data(all_u, args.batch_size_x_adam, args.batch_size_u_adam, device) # b
                    bound1, bound2, bound3, bound4, bound1_u, bound2_u, bound3_u, bound4_u, bound1_m, bound2_m, bound3_m, bound4_m = sample_train_data_bound(all_u, args.batch_size_x_adam, args.batch_size_u_adam, device) # 4 * (b//4)
                elif args.experiment == 'cy':
                    u, ux, uy, alpha, m, RHS, x = sample_train_data_tri(all_u, args.batch_size_x_adam, args.batch_size_u_adam, device) # b
                    bound1, bound2, bound3, bound4, bound1_u, bound2_u, bound3_u, bound4_u = sample_train_data_bound_tri(all_u, args.batch_size_x_adam, args.batch_size_u_adam, device) # 4 * (b//4)

                optimizer_adam.zero_grad()
                
                if args.bound_constraint == 'soft':
                    # loss of boundary condition
                    if len(bound1) == 0:
                        loss_bound1 = torch.zeros(1).to(device)
                    else:
                        bound11 = bound1[:, 0].view(-1, 1)
                        bound12 = bound1[:, 1].view(-1, 1)
                        # bound1t = bound1[:, 2].view(-1, 1)
                        bound11.requires_grad = True
                        bound12.requires_grad = True
                        # X1 = torch.cat((bound11, bound12, bound1t), dim=1)
                        X1 = torch.cat((bound11, bound12), dim=1)
                        output_bound1 = model(bound1_u, X1) 
                        v1 = torch.ones(output_bound1.shape).to(device)
                        bound1_x = torch.autograd.grad(output_bound1, bound11, grad_outputs=v1, retain_graph = True, create_graph=True, allow_unused=True)[0]
                        loss_bound1 = criterion(bound1_x, torch.zeros_like(bound1_x))

                    if len(bound2) == 0:
                        loss_bound2 = torch.zeros(1).to(device)
                    else:
                        bound21 = bound2[:, 0].view(-1, 1)
                        bound22 = bound2[:, 1].view(-1, 1)
                        # bound2t = bound2[:, 2].view(-1, 1)
                        bound21.requires_grad = True
                        bound22.requires_grad = True
                        # X2 = torch.cat((bound21, bound22, bound2t), dim=1)
                        X2 = torch.cat((bound21, bound22), dim=1)
                        output_bound2 = model(bound2_u, X2)
                        v2 = torch.ones(output_bound2.shape).to(device)
                        bound2_x = torch.autograd.grad(output_bound2, bound21, grad_outputs=v2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                        loss_bound2 = criterion(bound2_x, torch.zeros_like(bound2_x))

                    if len(bound3) == 0:
                        loss_bound3 = torch.zeros(1).to(device)
                    else:
                        bound31 = bound3[:, 0].view(-1, 1)
                        bound32 = bound3[:, 1].view(-1, 1)
                        # bound3t = bound3[:, 2].view(-1, 1)
                        bound31.requires_grad = True
                        bound32.requires_grad = True
                        # X3 = torch.cat((bound31, bound32, bound3t), dim=1)
                        X3 = torch.cat((bound31, bound32), dim=1)
                        output_bound3 = model(bound3_u, X3)
                        v3 = torch.ones(output_bound3.shape).to(device)
                        bound3_y = torch.autograd.grad(output_bound3, bound32, grad_outputs=v3, retain_graph = True, create_graph=True, allow_unused=True)[0]
                        loss_bound3 = criterion(bound3_y, torch.zeros_like(bound3_y))

                    if len(bound4) == 0:
                        loss_bound4 = torch.zeros(1).to(device)
                    else:
                        bound41 = bound4[:, 0].view(-1, 1)
                        bound42 = bound4[:, 1].view(-1, 1)
                        # bound4t = bound4[:, 2].view(-1, 1)
                        bound41.requires_grad = True
                        bound42.requires_grad = True
                        # X4 = torch.cat((bound41, bound42, bound4t), dim=1)
                        X4 = torch.cat((bound41, bound42), dim=1)
                        output_bound4 = model(bound4_u, X4)
                        v4 = torch.ones(output_bound4.shape).to(device)
                        bound4_y = torch.autograd.grad(output_bound4, bound42, grad_outputs=v4, retain_graph = True, create_graph=True, allow_unused=True)[0]
                        loss_bound4 = criterion(bound4_y, torch.zeros_like(bound4_y))

                    loss_bound = (loss_bound1 + loss_bound2 + loss_bound3 + loss_bound4) / 4
                else: 
                    loss_bound = torch.tensor(0).to(device)

                # loss inside
                x1 = x[:, 0].view(x.shape[0], 1)
                x2 = x[:, 1].view(x.shape[0], 1)
                # xt = x[:, 2].view(x.shape[0], 1)
                x1.requires_grad = True
                x2.requires_grad = True
                # x_ = torch.cat((x1, x2, xt), dim=1)
                x_ = torch.cat((x1, x2), dim=1)
                if args.bound_constraint == 'soft':
                    output = model(u, x_) # nu*nx
                else: 
                    output = ((x1**2) * (x2**2) * ((x1-1)**2) * ((x2-1)**2)) * model(u, x_) + (1/2) * (x1**2) + (1/2) * (x2**2)
                w = torch.ones(output.shape).to(device)
                phix = torch.autograd.grad(output, x1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                phiy = torch.autograd.grad(output, x2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                if init_mesh == True:
                    loss_in = (criterion(x1 + phix, x1) + criterion(x2 + phiy, x2)) / 2
                    loss = args.loss_weight1 * loss_bound + args.loss_weight0 * loss_in 
                    loss.backward()

                else:
                    w2 = torch.ones(phix.shape).to(device)
                    phixy = torch.autograd.grad(phix, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                    phixx = torch.autograd.grad(phix, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                    phiyx = torch.autograd.grad(phiy, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                    phiyy = torch.autograd.grad(phiy, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]

                    if args.experiment == 'burgers':
                        ux_ = interpolate(ux.unsqueeze(1).repeat(1, args.batch_size_x_adam, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
                        uy_ = interpolate(uy.unsqueeze(1).repeat(1, args.batch_size_x_adam, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
                    elif args.experiment == 'cy':
                        ux_ = interpolate(ux.unsqueeze(1).repeat(1, args.batch_size_x_adam, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
                        uy_ = interpolate(uy.unsqueeze(1).repeat(1, args.batch_size_x_adam, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
                    u_xi_x = ux_ * (1 + phixx) + uy_ * phiyx
                    u_xi_y = ux_ * phixy + uy_ * (1 + phiyy)
                    m_xi = monitor(alpha.unsqueeze(1).repeat(1, args.batch_size_x_adam).reshape(-1, 1), u_xi_x, u_xi_y)
                    LHS = m_xi * ((1 + phixx) * (1 + phiyy) - phixy * phiyx)

                    loss_in = criterion(LHS / RHS.unsqueeze(1).repeat(1, args.batch_size_x_adam).reshape(-1, 1), torch.ones_like(LHS))
                    loss_convex = torch.mean(torch.min(torch.tensor(0).type_as(phixx), 1 + phixx)**2 + torch.min(torch.tensor(0).type_as(phiyy), 1 + phiyy)**2)

                    if args.loss_convex == True:
                        loss = args.loss_weight1 * loss_bound + args.loss_weight0 * loss_in + args.loss_weight2 * loss_convex
                    else:
                        loss = args.loss_weight1 * loss_bound + args.loss_weight0 * loss_in
                    loss.backward()

                if log_count1[0] % 200 == 0:
                    loss_in_list.append(loss_in.item())
                    loss_convex_list.append(loss_convex.item())
                    loss_bound_list.append(loss_bound.item())
                    LHS_list.append(LHS.detach())
                    RHS_list.append(RHS)
                log_count1[0] += 1
                
                optimizer_adam.step()

        # LBFGS
        else:
            for i in range(np.max((1, int(args.train_sample_grid * all_u.shape[0] / (args.batch_size_x_lbfgs * args.batch_size_u_lbfgs))))):
                def closure():
                    if args.experiment == 'burgers':
                        u, ux, uy, alpha, m, RHS, x = sample_train_data(all_u, args.batch_size_x_lbfgs, args.batch_size_u_lbfgs, device) # b
                        bound1, bound2, bound3, bound4, bound1_u, bound2_u, bound3_u, bound4_u, bound1_m, bound2_m, bound3_m, bound4_m = sample_train_data_bound(all_u, args.batch_size_x_lbfgs, args.batch_size_u_lbfgs, device) # 4 * (b//4)
                    elif args.experiment == 'cy':
                        u, ux, uy, alpha, m, RHS, x = sample_train_data_tri(all_u, args.batch_size_x_lbfgs, args.batch_size_u_lbfgs, device) # b
                        bound1, bound2, bound3, bound4, bound1_u, bound2_u, bound3_u, bound4_u = sample_train_data_bound_tri(all_u, args.batch_size_x_lbfgs, args.batch_size_u_lbfgs, device) # 4 * (b//4)
                    weight = torch.ones_like(RHS).unsqueeze(-1)

                    optimizer_lbfgs.zero_grad()
                    
                    if args.bound_constraint == 'soft':
                    # loss of boundary condition
                        if len(bound1) == 0:
                            loss_bound1 = torch.zeros(1).to(device)
                        else:
                            bound11 = bound1[:, 0].view(-1, 1)
                            bound12 = bound1[:, 1].view(-1, 1)
                            # bound1t = bound1[:, 2].view(-1, 1)
                            bound11.requires_grad = True
                            bound12.requires_grad = True
                            # X1 = torch.cat((bound11, bound12, bound1t), dim=1)
                            X1 = torch.cat((bound11, bound12), dim=1)
                            output_bound1 = model(bound1_u, X1) 
                            v1 = torch.ones(output_bound1.shape).to(device)
                            bound1_x = torch.autograd.grad(output_bound1, bound11, grad_outputs=v1, retain_graph = True, create_graph=True, allow_unused=True)[0]
                            loss_bound1 = criterion(bound1_x, torch.zeros_like(bound1_x))

                        if len(bound2) == 0:
                            loss_bound2 = torch.zeros(1).to(device)
                        else:
                            bound21 = bound2[:, 0].view(-1, 1)
                            bound22 = bound2[:, 1].view(-1, 1)
                            # bound2t = bound2[:, 2].view(-1, 1)
                            bound21.requires_grad = True
                            bound22.requires_grad = True
                            # X2 = torch.cat((bound21, bound22, bound2t), dim=1)
                            X2 = torch.cat((bound21, bound22), dim=1)
                            output_bound2 = model(bound2_u, X2)
                            v2 = torch.ones(output_bound2.shape).to(device)
                            bound2_x = torch.autograd.grad(output_bound2, bound21, grad_outputs=v2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                            loss_bound2 = criterion(bound2_x, torch.zeros_like(bound2_x))

                        if len(bound3) == 0:
                            loss_bound3 = torch.zeros(1).to(device)
                        else:
                            bound31 = bound3[:, 0].view(-1, 1)
                            bound32 = bound3[:, 1].view(-1, 1)
                            # bound3t = bound3[:, 2].view(-1, 1)
                            bound31.requires_grad = True
                            bound32.requires_grad = True
                            # X3 = torch.cat((bound31, bound32, bound3t), dim=1)
                            X3 = torch.cat((bound31, bound32), dim=1)
                            output_bound3 = model(bound3_u, X3)
                            v3 = torch.ones(output_bound3.shape).to(device)
                            bound3_y = torch.autograd.grad(output_bound3, bound32, grad_outputs=v3, retain_graph = True, create_graph=True, allow_unused=True)[0]
                            loss_bound3 = criterion(bound3_y, torch.zeros_like(bound3_y))

                        if len(bound4) == 0:
                            loss_bound4 = torch.zeros(1).to(device)
                        else:
                            bound41 = bound4[:, 0].view(-1, 1)
                            bound42 = bound4[:, 1].view(-1, 1)
                            # bound4t = bound4[:, 2].view(-1, 1)
                            bound41.requires_grad = True
                            bound42.requires_grad = True
                            # X4 = torch.cat((bound41, bound42, bound4t), dim=1)
                            X4 = torch.cat((bound41, bound42), dim=1)
                            output_bound4 = model(bound4_u, X4)
                            v4 = torch.ones(output_bound4.shape).to(device)
                            bound4_y = torch.autograd.grad(output_bound4, bound42, grad_outputs=v4, retain_graph = True, create_graph=True, allow_unused=True)[0]
                            loss_bound4 = criterion(bound4_y, torch.zeros_like(bound4_y))

                        loss_bound = (loss_bound1 + loss_bound2 + loss_bound3 + loss_bound4) / 4
                    else: 
                        loss_bound = torch.tensor(0).to(device)

                    # loss inside
                    x1 = x[:, 0].view(x.shape[0], 1)
                    x2 = x[:, 1].view(x.shape[0], 1)
                    # xt = x[:, 2].view(x.shape[0], 1)
                    x1.requires_grad = True
                    x2.requires_grad = True
                    # x_ = torch.cat((x1, x2, xt), dim=1)
                    x_ = torch.cat((x1, x2), dim=1)
                    if args.bound_constraint == 'soft':
                        output = model(u, x_) # nu*nx
                    else: 
                        output = ((x1**2) * (x2**2) * ((x1-1)**2) * ((x2-1)**2)) * model(u, x_) + (1/2) * (x1**2) + (1/2) * (x2**2)
                    w = torch.ones(output.shape).to(device)
                    phix = torch.autograd.grad(output, x1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                    phiy = torch.autograd.grad(output, x2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                    if init_mesh == True:
                        loss_in = (criterion(x1 + phix, x1) + criterion(x2 + phiy, x2)) / 2
                        loss = args.loss_weight1 * loss_bound + args.loss_weight0 * loss_in 

                    else:
                        w2 = torch.ones(phix.shape).to(device)
                        phixy = torch.autograd.grad(phix, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                        phixx = torch.autograd.grad(phix, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                        phiyx = torch.autograd.grad(phiy, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                        phiyy = torch.autograd.grad(phiy, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]

                        if args.experiment == 'burgers':
                            ux_ = interpolate(ux.unsqueeze(1).repeat(1, args.batch_size_x_lbfgs, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
                            uy_ = interpolate(uy.unsqueeze(1).repeat(1, args.batch_size_x_lbfgs, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
                        elif args.experiment == 'cy':
                            ux_ = interpolate(ux.unsqueeze(1).repeat(1, args.batch_size_x_lbfgs, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
                            uy_ = interpolate(uy.unsqueeze(1).repeat(1, args.batch_size_x_lbfgs, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
                        u_xi_x = ux_ * (1 + phixx) + uy_ * phiyx
                        u_xi_y = ux_ * phixy + uy_ * (1 + phiyy)
                        m_xi = monitor(alpha.unsqueeze(1).repeat(1, args.batch_size_x_lbfgs).reshape(-1, 1), u_xi_x, u_xi_y)
                        LHS = m_xi * ((1 + phixx) * (1 + phiyy) - phixy * phiyx)

                        loss_in = criterion(LHS / RHS.unsqueeze(1).repeat(1, args.batch_size_x_lbfgs).reshape(-1, 1), torch.ones_like(LHS))
                        loss_convex = torch.mean(torch.min(torch.tensor(0).type_as(phixx), 1 + phixx)**2 + torch.min(torch.tensor(0).type_as(phiyy), 1 + phiyy)**2)

                        if args.loss_convex == True:
                            loss = args.loss_weight1 * loss_bound + args.loss_weight0 * loss_in + args.loss_weight2 * loss_convex
                        else:
                            loss = args.loss_weight1 * loss_bound + args.loss_weight0 * loss_in

                        loss.backward()

                    if log_count2[0] % 200 == 0:
                        loss_in_list.append(loss_in.item())
                        loss_convex_list.append(loss_convex.item())
                        loss_bound_list.append(loss_bound.item())
                        LHS_list.append(LHS.detach())
                        RHS_list.append(RHS)
                    log_count2[0] += 1

                    return loss
                    
                optimizer_lbfgs.step(closure)
            
        test_equ = LHS_list[-1] / RHS_list[-1] - torch.tensor(1).to(device)
        test_equ_max = torch.max(test_equ)
        test_equ_min = torch.min(test_equ)
        test_equ_min = torch.min(test_equ)
        test_equ_mid = torch.median(test_equ)
        test_equ_loss = torch.mean(torch.abs(test_equ))
        test_equ_loss_list.append(test_equ_loss.item())
        test_equ_max_list.append(test_equ_max.item())
        test_equ_min_list.append(test_equ_min.item())
        test_equ_mid_list.append(test_equ_mid.item())

        torch.cuda.empty_cache()
        end = datetime.now()

        # print & evaluate
        if epoch < n_epoch_adam + 1:
            scheduler_adam.step()

            if epoch % 1 == 0:
                print(end - start)
                print('Epoch: {} | Loss in: {} | Loss bound: {} | Loss convex: {} | Test equ loss: {:1.4f}'\
                    .format(epoch, loss_in_list[-1], loss_bound_list[-1], loss_convex_list[-1], test_equ_loss_list[-1]))
                logs_txt.append('Epoch: {} | Loss in: {} | Loss bound: {} | Loss convex: {} | Test equ loss: {:1.4f}'\
                    .format(epoch, loss_in_list[-1], loss_bound_list[-1], loss_convex_list[-1], test_equ_loss_list[-1]))
            if epoch % 1 == 0 or epoch == n_epoch_adam:
                if args.experiment == 'burgers':
                    train_mean, train_std, train_minmax = evaluate(model, all_u, device, epoch)
                    test_mean, test_std, test_minmax = evaluate(model, test_u, device, epoch)
                elif args.experiment =='cy':
                    train_mean, train_std, train_minmax = evaluate_tri(model, all_u[:, :, 2], all_u[0, :, :2], device, epoch)
                    test_mean, test_std, test_minmax = evaluate_tri(model, test_u[:, :, 2], all_u[0, :, :2], device, epoch)
                train_std_list.append(train_std)
                train_minmax_list.append(train_minmax)
                test_std_list.append(test_std)
                test_minmax_list.append(test_minmax)
                print('Train mean: {:1.6f} | Train std: {:1.6f} | Train minmax: {:1.6f} | Test mean: {:1.6f} | Test std: {:1.6f} | Test minmax: {:1.6f}'\
                            .format(train_mean, train_std, train_minmax, test_mean, test_std, test_minmax))
                logs_txt.append('Train mean: {:1.6f} | Train std: {:1.6f} | Train minmax: {:1.6f} | Test mean: {:1.6f} | Test std: {:1.6f} | Test minmax: {:1.6f}'\
                            .format(train_mean, train_std, train_minmax, test_mean, test_std, test_minmax))
            
        else:
            scheduler_lbfgs.step()

            if epoch % 1 == 0:
                print(end - start)
                print('Epoch: {} | Loss in: {} | Loss bound: {} | Loss convex: {} | Test equ loss: {:1.4f}'\
                    .format(epoch, loss_in_list[-1], loss_bound_list[-1], loss_convex_list[-1], test_equ_loss_list[-1]))
                logs_txt.append('Epoch: {} | Loss in: {} | Loss bound: {} | Loss convex: {} | Test equ loss: {:1.4f}'\
                    .format(epoch, loss_in_list[-1], loss_bound_list[-1], loss_convex_list[-1], test_equ_loss_list[-1]))
            if epoch % 1 == 0:
                if args.experiment == 'burgers':
                    train_mean, train_std, train_minmax = evaluate(model, all_u, device, epoch)
                    test_mean, test_std, test_minmax = evaluate(model, test_u, device, epoch)
                elif args.experiment =='cy':
                    train_mean, train_std, train_minmax = evaluate_tri(model, all_u[:, :, 2], all_u[0, :, :2], device, epoch)
                    test_mean, test_std, test_minmax = evaluate_tri(model, test_u[:, :, 2], all_u[0, :, :2], device, epoch)
                train_std_list.append(train_std)
                train_minmax_list.append(train_minmax)
                test_std_list.append(test_std)
                test_minmax_list.append(test_minmax)
                print('Train mean: {:1.6f} | Train std: {:1.6f} | Train minmax: {:1.6f} | Test mean: {:1.6f} | Test std: {:1.6f} | Test minmax: {:1.6f}'\
                            .format(train_mean, train_std, train_minmax, test_mean, test_std, test_minmax))
                logs_txt.append('Train mean: {:1.6f} | Train std: {:1.6f} | Train minmax: {:1.6f} | Test mean: {:1.6f} | Test std: {:1.6f} | Test minmax: {:1.6f}'\
                            .format(train_mean, train_std, train_minmax, test_mean, test_std, test_minmax))

        save_path = '{}/{}_{}_bound{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'\
                    .format(args.experiment, datetime.now(), args.rf, args.loss_bound_rf, args.epochs_rf, args.max_iter, args.sub_u, args.epochs_lbfgs,\
                    args.batch_size_u_adam, args.batch_size_x_adam, args.loss_weight1, args.train_sample_grid, args.branch_layers, args.lr_adam, args.trunk_layers, args.gamma_adam)
    
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss_in': loss_in_list,
            'loss_bound': loss_bound_list,
            'loss_convex': loss_convex_list,
            'args': args,
            'train_std': train_std_list,
            'train_minmax': train_minmax_list,
            'test_std': test_std_list,
            'test_minmax': test_minmax_list,
            }, save_path)

    # random feature method
    if args.rf == True:
        c = 1
        for i in range(args.epochs_rf):
            start = datetime.now()
            # print("time start: ", start)
            print('random feature method epoch No.', i)
            if args.experiment == 'burgers':
                u, ux, uy, alpha, m, RHS, x = sample_train_data(all_u, args.batch_size_x_rf, args.batch_size_u_rf, device) # b
                bound1, bound2, bound3, bound4, bound1_u, bound2_u, bound3_u, bound4_u, bound1_m, bound2_m, bound3_m, bound4_m = sample_train_data_bound(all_u, args.batch_size_x_rf, args.batch_size_u_rf, device) # 4 * (b//4)
            elif args.experiment == 'cy':
                u, ux, uy, alpha, m, RHS, x = sample_train_data_tri(all_u, args.batch_size_x_rf, args.batch_size_u_rf, device) # b
                bound1, bound2, bound3, bound4, bound1_u, bound2_u, bound3_u, bound4_u = sample_train_data_bound_tri(all_u, args.batch_size_x_rf, args.batch_size_u_rf, device) # 4 * (b//4)
            # loss of boundary condition
            bound11 = bound1[:, 0].view(-1, 1)
            bound12 = bound1[:, 1].view(-1, 1)
            # bound1t = bound1[:, 2].view(-1, 1)
            bound11.requires_grad = True
            bound12.requires_grad = True
            # X1 = torch.cat((bound11, bound12, bound1t), dim=1)
            X1 = torch.cat((bound11, bound12), dim=1)
            output_bound1, second_out_bound1, branch_bound1 = model(bound1_u, X1, rf = True)
            so_x_bound1, so_y_bound1 = [], []
            for k in range(int(second_out_bound1.shape[-1])):
                second_out_bound1_ = second_out_bound1[:, k]
                w = torch.ones_like(second_out_bound1_).to(device)
                so_x_bound1_ = torch.autograd.grad(second_out_bound1_, bound11, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_y_bound1_ = torch.autograd.grad(second_out_bound1_, bound12, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_x_bound1.append(so_x_bound1_)
                so_y_bound1.append(so_y_bound1_)
            so_x_bound1 = torch.stack(so_x_bound1)[:, :, 0].permute(1, 0)
            so_y_bound1 = torch.stack(so_y_bound1)[:, :, 0].permute(1, 0)

            bound21 = bound2[:, 0].view(-1, 1)
            bound22 = bound2[:, 1].view(-1, 1)
            # bound2t = bound2[:, 2].view(-1, 1)
            bound21.requires_grad = True
            bound22.requires_grad = True
            # X2 = torch.cat((bound21, bound22, bound2t), dim=1)
            X2 = torch.cat((bound21, bound22), dim=1)
            output_bound2, second_out_bound2, branch_bound2 = model(bound2_u, X2, rf = True)
            so_x_bound2, so_y_bound2 = [], []
            for k in range(int(second_out_bound2.shape[-1])):
                second_out_bound2_ = second_out_bound2[:, k]
                w = torch.ones_like(second_out_bound2_).to(device)
                so_x_bound2_ = torch.autograd.grad(second_out_bound2_, bound21, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_y_bound2_ = torch.autograd.grad(second_out_bound2_, bound22, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_x_bound2.append(so_x_bound2_)
                so_y_bound2.append(so_y_bound2_)
            so_x_bound2 = torch.stack(so_x_bound2)[:, :, 0].permute(1, 0)
            so_y_bound2 = torch.stack(so_y_bound2)[:, :, 0].permute(1, 0)

            bound31 = bound3[:, 0].view(-1, 1)
            bound32 = bound3[:, 1].view(-1, 1)
            # bound3t = bound3[:, 2].view(-1, 1)
            bound31.requires_grad = True
            bound32.requires_grad = True
            # X3 = torch.cat((bound31, bound32, bound3t), dim=1)
            X3 = torch.cat((bound31, bound32), dim=1)
            output_bound3, second_out_bound3, branch_bound3 = model(bound3_u, X3, rf = True)
            so_x_bound3, so_y_bound3 = [], []
            for k in range(int(second_out_bound3.shape[-1])):
                second_out_bound3_ = second_out_bound3[:, k]
                w = torch.ones_like(second_out_bound3_).to(device)
                so_x_bound3_ = torch.autograd.grad(second_out_bound3_, bound31, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_y_bound3_ = torch.autograd.grad(second_out_bound3_, bound32, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_x_bound3.append(so_x_bound3_)
                so_y_bound3.append(so_y_bound3_)
            so_x_bound3 = torch.stack(so_x_bound3)[:, :, 0].permute(1, 0)
            so_y_bound3 = torch.stack(so_y_bound3)[:, :, 0].permute(1, 0)

            bound41 = bound4[:, 0].view(-1, 1)
            bound42 = bound4[:, 1].view(-1, 1)
            # bound4t = bound4[:, 2].view(-1, 1)
            bound41.requires_grad = True
            bound42.requires_grad = True
            # X4 = torch.cat((bound41, bound42, bound4t), dim=1)
            X4 = torch.cat((bound41, bound42), dim=1)
            output_bound4, second_out_bound4, branch_bound4 = model(bound4_u, X4, rf = True)
            so_x_bound4, so_y_bound4 = [], []
            for k in range(int(second_out_bound4.shape[-1])):
                second_out_bound4_ = second_out_bound4[:, k]
                w = torch.ones_like(second_out_bound4_).to(device)
                so_x_bound4_ = torch.autograd.grad(second_out_bound4_, bound41, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_y_bound4_ = torch.autograd.grad(second_out_bound4_, bound42, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_x_bound4.append(so_x_bound4_)
                so_y_bound4.append(so_y_bound4_)
            so_x_bound4 = torch.stack(so_x_bound4)[:, :, 0].permute(1, 0)
            so_y_bound4 = torch.stack(so_y_bound4)[:, :, 0].permute(1, 0)

            # loss in
            x1 = x[:, 0].view(x.shape[0], 1)
            x2 = x[:, 1].view(x.shape[0], 1)
            # xt = x[:, 2].view(x.shape[0], 1)
            x1.requires_grad = True
            x2.requires_grad = True
            # x_ = torch.cat((x1, x2, xt), dim=1)
            x_ = torch.cat((x1, x2), dim=1)
            output, second_out, branch = model(u, x_, rf = True)
            so_x, so_y, so_xx, so_xy, so_yx, so_yy = [], [], [], [], [], []
            for k in range(int(second_out.shape[-1])):
                second_out_ = second_out[:, k]
                w = torch.ones_like(second_out_).to(device)
                so_x_ = torch.autograd.grad(second_out_, x1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_y_ = torch.autograd.grad(second_out_, x2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
                w2 = torch.ones_like(so_x_).to(device)
                so_xy_ = torch.autograd.grad(so_x_, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_xx_ = torch.autograd.grad(so_x_, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_yx_ = torch.autograd.grad(so_y_, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_yy_ = torch.autograd.grad(so_y_, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                so_x.append(so_x_)
                so_y.append(so_y_)
                so_xx.append(so_xx_)
                so_xy.append(so_xy_)
                so_yx.append(so_yx_)
                so_yy.append(so_yy_)
            so_x = torch.stack(so_x)[:, :, 0].permute(1, 0)
            so_y = torch.stack(so_y)[:, :, 0].permute(1, 0)
            so_xx = torch.stack(so_xx)[:, :, 0].permute(1, 0)
            so_xy = torch.stack(so_xy)[:, :, 0].permute(1, 0)
            so_yx = torch.stack(so_yx)[:, :, 0].permute(1, 0)
            so_yy = torch.stack(so_yy)[:, :, 0].permute(1, 0)
            
            w = torch.ones(output.shape).to(device)
            phix = torch.autograd.grad(output, x1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
            phiy = torch.autograd.grad(output, x2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
            w2 = torch.ones(phix.shape).to(device)
            phixy = torch.autograd.grad(phix, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
            phixx = torch.autograd.grad(phix, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
            phiyx = torch.autograd.grad(phiy, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
            phiyy = torch.autograd.grad(phiy, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]

            ux_ = interpolate(ux.unsqueeze(1).repeat(1, args.batch_size_x_rf, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
            uy_ = interpolate(uy.unsqueeze(1).repeat(1, args.batch_size_x_rf, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
            u_xi_x = ux_ * (1 + phixx) + uy_ * phiyx
            u_xi_y = ux_ * phixy + uy_ * (1 + phiyy)
            m_xi = monitor(alpha.unsqueeze(1).repeat(1, args.batch_size_x_rf).reshape(-1, 1), u_xi_x, u_xi_y)
                
            init = model.out_nn.layers[-1].weight.data.reshape(-1, 1)
            if args.rf_opt_alg == 'BFGS':
                desired_weights = minimize(lambda x: random_feature_torch2(x, args.convex_rel, second_out,\
                                        second_out_bound1, second_out_bound2, second_out_bound3, second_out_bound4,\
                                        args, alpha, x1, x2, ux, uy, so_x_bound1, so_x_bound2, so_y_bound3, so_y_bound4,\
                                        so_x, so_y, so_xx,so_yy, so_xy, so_yx, RHS), 
                                        init, 
                                        method='bfgs', 
                                        options=dict(line_search='strong-wolfe'), 
                                        max_iter=args.max_iter,
                                        disp=0,
                                        tol=0)
            elif args.rf_opt_alg == 'Newton':
                desired_weights = minimize(lambda x: random_feature_torch2(x, args.convex_rel, second_out,\
                                        second_out_bound1, second_out_bound2, second_out_bound3, second_out_bound4,\
                                        args, alpha, x1, x2, ux, uy, so_x_bound1, so_x_bound2, so_y_bound3, so_y_bound4,\
                                        so_x, so_y, so_xx, so_yy, so_xy, so_yx, RHS), 
                                        init, 
                                        method='newton-cg', 
                                        options=dict(line_search='strong-wolfe'), 
                                        max_iter=args.max_iter,
                                        disp=0, 
                                        tol=0)
            model.out_nn.layers[-1].weight.data = desired_weights.x.reshape(1, second_out.shape[1])
            end = datetime.now()
            print("time per epoch of random feature method: ", end - start)
            c = c + 1

            # test for random feature method
            if args.experiment == 'burgers':
                u, ux, uy, alpha, m, RHS, x = sample_train_data(all_u, args.batch_size_x_rf, args.batch_size_u_rf, device) # b
                bound1, bound2, bound3, bound4, bound1_u, bound2_u, bound3_u, bound4_u, bound1_m, bound2_m, bound3_m, bound4_m = sample_train_data_bound(all_u, args.batch_size_x_rf, args.batch_size_u_rf, device) # 4 * (b//4)
            elif args.experiment == 'cy':
                u, ux, uy, alpha, m, RHS, x = sample_train_data_tri(all_u, args.batch_size_x_rf, args.batch_size_u_rf, device) # b
                bound1, bound2, bound3, bound4, bound1_u, bound2_u, bound3_u, bound4_u = sample_train_data_bound_tri(all_u, args.batch_size_x_rf, args.batch_size_u_rf, device) # 4 * (b//4)
            
            # loss of boundary condition
            if len(bound1) == 0:
                loss_bound1 = torch.zeros(1).to(device)
            else:
                bound11 = bound1[:, 0].view(-1, 1)
                bound12 = bound1[:, 1].view(-1, 1)
                # bound1t = bound1[:, 2].view(-1, 1)
                bound11.requires_grad = True
                bound12.requires_grad = True
                # X1 = torch.cat((bound11, bound12, bound1t), dim=1)
                X1 = torch.cat((bound11, bound12), dim=1)
                output_bound1 = model(bound1_u, X1) 
                v1 = torch.ones(output_bound1.shape).to(device)
                bound1_x = torch.autograd.grad(output_bound1, bound11, grad_outputs=v1, retain_graph = True, create_graph=True, allow_unused=True)[0]
                loss_bound1 = criterion(bound1_x, torch.zeros_like(bound1_x))

            if len(bound2) == 0:
                loss_bound2 = torch.zeros(1).to(device)
            else:
                bound21 = bound2[:, 0].view(-1, 1)
                bound22 = bound2[:, 1].view(-1, 1)
                # bound2t = bound2[:, 2].view(-1, 1)
                bound21.requires_grad = True
                bound22.requires_grad = True
                # X2 = torch.cat((bound21, bound22, bound2t), dim=1)
                X2 = torch.cat((bound21, bound22), dim=1)
                output_bound2 = model(bound2_u, X2)
                v2 = torch.ones(output_bound2.shape).to(device)
                bound2_x = torch.autograd.grad(output_bound2, bound21, grad_outputs=v2, retain_graph = True, create_graph=True, allow_unused=True)[0]
                loss_bound2 = criterion(bound2_x, torch.zeros_like(bound2_x))

            if len(bound3) == 0:
                loss_bound3 = torch.zeros(1).to(device)
            else:
                bound31 = bound3[:, 0].view(-1, 1)
                bound32 = bound3[:, 1].view(-1, 1)
                # bound3t = bound3[:, 2].view(-1, 1)
                bound31.requires_grad = True
                bound32.requires_grad = True
                # X3 = torch.cat((bound31, bound32, bound3t), dim=1)
                X3 = torch.cat((bound31, bound32), dim=1)
                output_bound3 = model(bound3_u, X3)
                v3 = torch.ones(output_bound3.shape).to(device)
                bound3_y = torch.autograd.grad(output_bound3, bound32, grad_outputs=v3, retain_graph = True, create_graph=True, allow_unused=True)[0]
                loss_bound3 = criterion(bound3_y, torch.zeros_like(bound3_y))

            if len(bound4) == 0:
                loss_bound4 = torch.zeros(1).to(device)
            else:
                bound41 = bound4[:, 0].view(-1, 1)
                bound42 = bound4[:, 1].view(-1, 1)
                # bound4t = bound4[:, 2].view(-1, 1)
                bound41.requires_grad = True
                bound42.requires_grad = True
                # X4 = torch.cat((bound41, bound42, bound4t), dim=1)
                X4 = torch.cat((bound41, bound42), dim=1)
                output_bound4, second_out_bound4, branch = model(bound4_u, X4, rf=True)
                v4 = torch.ones(output_bound4.shape).to(device)
                bound4_y = torch.autograd.grad(output_bound4, bound42, grad_outputs=v4, retain_graph = True, create_graph=True, allow_unused=True)[0]
                loss_bound4 = criterion(bound4_y, torch.zeros_like(bound4_y))
            loss_bound = (loss_bound1 + loss_bound2 + loss_bound3 + loss_bound4) / 4

            # loss inside
            x1 = x[:, 0].view(x.shape[0], 1)
            x2 = x[:, 1].view(x.shape[0], 1)
            # xt = x[:, 2].view(x.shape[0], 1)
            x1.requires_grad = True
            x2.requires_grad = True
            # x_ = torch.cat((x1, x2, xt), dim=1)
            x_ = torch.cat((x1, x2), dim=1)
            if args.bound_constraint == 'soft':
                output = model(u, x_) # nu*nx
            else: 
                output = ((x1**2) * (x2**2) * ((x1-1)**2) * ((x2-1)**2)) * model(u, x_) + (1/2) * (x1**2) + (1/2) * (x2**2)
            w = torch.ones(output.shape).to(device)
            phix = torch.autograd.grad(output, x1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]
            phiy = torch.autograd.grad(output, x2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0]

            w2 = torch.ones(phix.shape).to(device)
            phixy = torch.autograd.grad(phix, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
            phixx = torch.autograd.grad(phix, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
            phiyx = torch.autograd.grad(phiy, x1, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]
            phiyy = torch.autograd.grad(phiy, x2, grad_outputs=w2, retain_graph = True, create_graph=True, allow_unused=True)[0]

            ux_ = interpolate(ux.unsqueeze(1).repeat(1, args.batch_size_x_rf, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
            uy_ = interpolate(uy.unsqueeze(1).repeat(1, args.batch_size_x_rf, 1, 1).reshape(-1, ux.shape[-1], ux.shape[-1]), x1 + phix, x2 + phiy)
            u_xi_x = ux_ * (1 + phixx) + uy_ * phiyx
            u_xi_y = ux_ * phixy + uy_ * (1 + phiyy)
            m_xi = monitor(alpha.unsqueeze(1).repeat(1, args.batch_size_x_rf).reshape(-1, 1), u_xi_x, u_xi_y)
            LHS = m_xi * ((1 + phixx) * (1 + phiyy) - phixy * phiyx)

            loss_in = criterion(LHS / RHS.unsqueeze(1).repeat(1, args.batch_size_x_rf).reshape(-1, 1), torch.ones_like(LHS))
            loss_convex = torch.mean(torch.min(torch.tensor(0).type_as(phixx), 1 + phixx)**2 + torch.min(torch.tensor(0).type_as(phiyy), 1 + phiyy)**2)

            test_equ = LHS / RHS - torch.tensor(1).to(device)
            test_equ_loss = torch.mean(torch.abs(test_equ))
            test_equ_loss_list.append(test_equ_loss.item())
               
            loss_in_list.append(loss_in.item())
            loss_bound_list.append(loss_bound.item())
            print('Epoch: {} | Loss in: {} | Loss bound: {} | Loss convex: {} | Test equ loss: {:1.4f}'\
                    .format(epoch + c - 1, loss_in_list[-1], loss_bound_list[-1], 0, test_equ_loss_list[-1]))
            logs_txt.append('Epoch: {} | Loss in: {} | Loss bound: {} | Loss convex: {} | Test equ loss: {:1.4f}'\
                    .format(epoch + c - 1, loss_in_list[-1], loss_bound_list[-1], 0, test_equ_loss_list[-1]))

            if args.experiment == 'burgers':
                train_mean, train_std, train_minmax = evaluate(model, all_u, device, epoch)
                test_mean, test_std, test_minmax = evaluate(model, test_u, device, epoch)
            elif args.experiment =='cy':
                train_mean, train_std, train_minmax = evaluate_tri(model, all_u[:, :, 2], all_u[0, :, :2], device, epoch)
                test_mean, test_std, test_minmax = evaluate_tri(model, test_u[:, :, 2], all_u[0, :, :2], device, epoch)
            train_std_list.append(train_std)
            train_minmax_list.append(train_minmax)
            test_std_list.append(test_std)
            test_minmax_list.append(test_minmax)
            print('Train mean: {:1.6f} | Train std: {:1.6f} | Train minmax: {:1.6f} | Test mean: {:1.6f} | Test std: {:1.6f} | Test minmax: {:1.6f}'\
                        .format(train_mean, train_std, train_minmax, test_mean, test_std, test_minmax))
            logs_txt.append('Train mean: {:1.6f} | Train std: {:1.6f} | Train minmax: {:1.6f} | Test mean: {:1.6f} | Test std: {:1.6f} | Test minmax: {:1.6f}'\
                        .format(train_mean, train_std, train_minmax, test_mean, test_std, test_minmax))
            
    save_path = '{}/{}_{}_bound{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'\
            .format(args.experiment, datetime.now(), args.rf, args.loss_bound_rf, args.epochs_rf, args.max_iter, args.sub_u, args.epochs_lbfgs,\
            args.batch_size_u_adam, args.batch_size_x_adam, args.loss_weight1, args.train_sample_grid, args.branch_layers, args.lr_adam, args.trunk_layers, args.gamma_adam)
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_in': loss_in_list,
        'loss_bound': loss_bound_list,
        'loss_convex': loss_convex_list,
        'args': args,
        'train_std': train_std_list,
        'train_minmax': train_minmax_list,
        'test_std': test_std_list,
        'test_minmax': test_minmax_list,
        }, save_path)
    print(save_path)

    return model, loss_in_list, loss_bound_list, loss_convex_list, test_equ_loss_list, test_equ_max_list, test_equ_min_list, test_equ_mid_list,\
          train_std_list, train_minmax_list, test_std_list, test_minmax_list, itp_list1, itp_list2, logs_txt


#####evaluate & plot#####

def interpolate3(u, init_x, init_y, x, y, n):
    d = -torch.norm(torch.cat((init_x, init_y), dim=-1).unsqueeze(0).repeat(x.shape[0], 1, 1) - torch.cat((x, y), dim=-1).unsqueeze(1).repeat(1, init_x.shape[0], 1), dim=-1) * n
    normalize = nn.Softmax(dim=-1)
    weight = normalize(d)  
    interpolated = torch.sum(u.reshape(1, init_x.shape[0]).repeat(x.shape[0], 1) * weight, dim=-1)

    return interpolated


def itp_error(u, model, device, plot):
    ori_nx = u.shape[-2]
    ori_ny = u.shape[-1]
    ori_x = torch.linspace(0, 1, ori_nx).to(device)
    ori_y = torch.linspace(0, 1, ori_ny).to(device)
    ori_grid_x, ori_grid_y = torch.meshgrid(ori_x, ori_y)
    
    nx, ny = int(ori_nx / 4), int(ori_nx / 4)
    grid1 = np.linspace(0, 1, nx)
    grid2 = np.linspace(0, 1, ny)
    grid = torch.tensor(np.array(np.meshgrid(grid1, grid2)), dtype=torch.float).reshape(2, -1).permute(1, 0).to(device)
    xi1, xi2 = grid[:, [0]], grid[:, [1]]
    xi1.requires_grad = True
    xi2.requires_grad = True
    xi = torch.cat((xi1, xi2), dim=-1)
    phi = model(u[[0]], xi)
    w = torch.ones(phi.shape).to(device)
    mesh_x1 = (torch.autograd.grad(phi, xi1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi1)
    mesh_y1 = (torch.autograd.grad(phi, xi2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi2)

    mesh_x2, mesh_y2 = xi1, xi2

    mesh_u1 = interpolate3(u.reshape(-1, ori_nx, ori_ny), ori_grid_x.reshape(-1, 1), ori_grid_y.reshape(-1, 1), mesh_x1, mesh_y1, ori_nx).reshape(-1, nx, ny)
    mesh_u2 = interpolate3(u.reshape(-1, ori_nx, ori_ny), ori_grid_x.reshape(-1, 1), ori_grid_y.reshape(-1, 1), mesh_x2, mesh_y2, ori_nx).reshape(-1, nx, ny)
    interpolated_uni_u1 = interpolate3(mesh_u1, mesh_x1, mesh_y1, ori_grid_x.reshape(-1, 1), ori_grid_y.reshape(-1, 1), ori_nx).reshape(-1, ori_nx, ori_ny)
    interpolated_uni_u2 = interpolate3(mesh_u2, mesh_x2, mesh_y2, ori_grid_x.reshape(-1, 1), ori_grid_y.reshape(-1, 1), ori_nx).reshape(-1, ori_nx, ori_ny)

    itp_error1 = torch.norm((interpolated_uni_u1-u)).item()/torch.norm(u).item()
    itp_error2 = torch.norm((interpolated_uni_u2-u)).item()/torch.norm(u).item()

    if plot == True:
        norm = matplotlib.colors.Normalize(vmin=(torch.abs(interpolated_uni_u1-u)).cpu().min(), vmax=(torch.abs(interpolated_uni_u1-u)).cpu().max())
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.binary), format='%.2f')
        plt.contourf((torch.abs(interpolated_uni_u1-u))[0].detach().cpu().numpy(), 50, cmap=plt.cm.binary, norm = norm)
        plt.savefig("itp/{}.png".format(datetime.now()))
        plt.clf()

    return itp_error1, itp_error2


def triangle_area_and_centroid(v1, v2, v3):  
    x1, y1 = v1  
    x2, y2 = v2  
    x3, y3 = v3  
  
    area = 0.5 * abs((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))  
  
    centroid_x = (x1 + x2 + x3) / 3  
    centroid_y = (y1 + y2 + y3) / 3  
  
    return area, (centroid_x, centroid_y)  


def evaluate_tri(model, u, grid, device, epoch):
    u = u.to(device)
    grid = grid.to(device)
    xi1, xi2 = grid[:, [0]], grid[:, [1]]
    xi1.requires_grad = True
    xi2.requires_grad = True
    xi = torch.cat((xi1, xi2), dim=-1)

    n = int(np.sqrt(u.shape[-1]))
    uni_grid = torch.tensor(np.array(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))), dtype=torch.float)\
                    .reshape(2, -1).permute(1, 0).to(device)

    x1 = xi1[:, 0].cpu().detach().numpy()
    x2 = xi2[:, 0].cpu().detach().numpy()
    points = np.column_stack((x1, x2))  
    tri = Delaunay(points)  
    triangles_indices = tri.simplices  

    mean = []
    std = []
    minmax = []
    # idx = np.random.choice(u.shape[0], 30, replace=False)
    idx = np.random.choice(u.shape[0], min(150, u.shape[0]), replace=False)
    for t in idx:
        phi = model(u[[t]], xi)
        w = torch.ones(phi.shape).to(device)
        x1 = (torch.autograd.grad(phi, xi1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi1)[:, 0].detach().cpu().numpy()
        x2 = (torch.autograd.grad(phi, xi2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi2)[:, 0].detach().cpu().numpy()
        points = np.column_stack((x1, x2))  
         
        areas = []  
        centroids = []  
        for triangle_indices in triangles_indices:  
            v1, v2, v3 = points[triangle_indices]  
            area, centroid = triangle_area_and_centroid(v1, v2, v3)  
            areas.append(area)  
            centroids.append(centroid)  
        areas = torch.tensor(np.array(areas), device=device)  
        centroids = torch.tensor(np.array(centroids), device=device)  
        center1 = centroids[:, [0]]
        center2 = centroids[:, [1]]

        x1_ = uni_grid[:, [0]]
        x2_ = uni_grid[:, [1]]
        x1_.requires_grad = True
        x2_.requires_grad = True
        u_ = interpolate_tri(u[[t]].unsqueeze(1).repeat(1, n**2, 1).reshape(-1, u.shape[-1]), \
                        xi1.unsqueeze(0).unsqueeze(0).repeat(1, n**2, 1, 1).reshape(-1, u.shape[-1], 1), \
                        xi2.unsqueeze(0).unsqueeze(0).repeat(1, n**2, 1, 1).reshape(-1, u.shape[-1], 1), \
                        x1_.unsqueeze(1).repeat(1, u.shape[-1], 1), x2_.unsqueeze(1).repeat(1, u.shape[-1], 1))
        w = torch.ones(u_.shape).to(device)
        uni_ux = torch.autograd.grad(u_, x1_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0].reshape(1, n, n)
        uni_uy = torch.autograd.grad(u_, x2_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0].reshape(1, n, n)
        alpha = torch.sum((torch.abs(uni_ux)**2 + torch.abs(uni_uy)**2)**(1/2), dim=(-2, -1)) / (n-1)**2
        m = monitor(alpha.unsqueeze(-1).unsqueeze(-1).repeat(1, n, n), uni_ux, uni_uy).reshape(1, -1)
        RHS = torch.sum(m, dim=(-2, -1)) / (n-1)**2
        
        m_center = []
        N = int(center1.shape[0] / 2)
        m_center.append(interpolate_tri(m.repeat(N, 1), x1_[None].repeat(N, 1, 1), x2_[None].repeat(N, 1, 1),\
                        center1[:N].unsqueeze(1).repeat(1, n**2, 1), center2[:N].unsqueeze(1).repeat(1, n**2, 1)))
        m_center.append(interpolate_tri(m.repeat(center1.shape[0] - N, 1), x1_[None].repeat(center1.shape[0] - N, 1, 1),\
                        x2_[None].repeat(center1.shape[0] - N, 1, 1), center1[N:].unsqueeze(1).repeat(1, n**2, 1),\
                        center2[N:].unsqueeze(1).repeat(1, n**2, 1)))
        m_center = torch.cat(m_center).reshape(-1)
        m_per_grid = m_center * areas
        mean.append(torch.mean(m_per_grid).cpu().detach().numpy())
        std.append(torch.std(m_per_grid).cpu().detach().numpy())
        minmax.append((torch.max(m_per_grid) - torch.min(m_per_grid)).cpu().detach().numpy())
        
    return np.mean(mean), np.mean(std), np.mean(minmax)


def evaluate(model, u, device, epoch):
    # computational mesh
    s = u.shape[-1]
    grid1 = np.linspace(0, 1, s)
    grid2 = np.linspace(0, 1, s)
    grid = torch.tensor(np.array(np.meshgrid(grid1, grid2)), dtype=torch.float).reshape(2, -1).permute(1, 0).to(device)
    xi1, xi2 = grid[:, [0]], grid[:, [1]]
    xi1.requires_grad = True
    xi2.requires_grad = True
    xi = torch.cat((xi1, xi2), dim=-1)

    # monitor function
    u = u.to(device)
    ux = diff_x(u) * (u.shape[-1] - 1)
    uy = diff_y(u) * (u.shape[-1] - 1)
    alpha = torch.sum((torch.abs(ux)**2 + torch.abs(uy)**2)**(1/2), dim=(-2, -1)) / (u.shape[-1]-1)**2
    m = monitor(alpha.unsqueeze(-1).unsqueeze(-1).repeat(1, ux.shape[-1], ux.shape[-1]), ux, uy)
    ideal_m_per_grid = ((torch.sum(m, dim=(-2, -1)) / (u.shape[-1]-1)**2) / (s - 1)**2).cpu().numpy()

    mean = []
    std = []
    minmax = []
    # idx = np.random.choice(u.shape[0], 30, replace=False)
    idx = np.random.choice(u.shape[0], u.shape[0], replace=False)
    for t in idx:
        phi = model(u[[t]], xi)
        w = torch.ones(phi.shape).to(device)
        x1 = (torch.autograd.grad(phi, xi1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi1).reshape(s, s)
        x2 = (torch.autograd.grad(phi, xi2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi2).reshape(s, s)
        bottom_left1, bottom_left2 = x1[:(s-1), :(s-1)], x2[:(s-1), :(s-1)]
        bottom_right1, bottom_right2 = x1[1:s, :(s-1)], x2[1:s, :(s-1)]
        top_left1, top_left2 = x1[:(s-1), 1:s], x2[:(s-1), 1:s]
        top_right1, top_right2 = x1[1:s, 1:s], x2[1:s, 1:s]
        # diagonal
        d1 = ((bottom_left1 - top_right1) ** 2 + (bottom_left2 - top_right2) ** 2) ** 0.5
        d2 = ((bottom_right1 - top_left1) ** 2 + (bottom_right2 - top_left2) ** 2) ** 0.5
        # area of the quadrilateral
        area = d1 * d2 / 2
        center1, center2 = (bottom_left1 + bottom_right1 + top_left1 + top_right1) / 4, (bottom_left2 + bottom_right2 + top_left2 + top_right2) / 4
        m_center = []
        N = int((s-1)**2 / 2)
        m_center.append(interpolate(m[[t]].repeat(N, 1, 1), center1.reshape(-1, 1)[:N], center2.reshape(-1, 1)[:N]))
        m_center.append(interpolate(m[[t]].repeat((s-1)**2 - N, 1, 1), center1.reshape(-1, 1)[N:], center2.reshape(-1, 1)[N:]))
        m_center = torch.cat(m_center).reshape(s-1, s-1)
        m_per_grid = m_center * area
        mean.append(torch.mean(m_per_grid).cpu().detach().numpy())
        std.append(torch.std(m_per_grid).cpu().detach().numpy())
        minmax.append((torch.max(m_per_grid) - torch.min(m_per_grid)).cpu().detach().numpy())
        
    return np.mean(mean), np.mean(std), np.mean(minmax)

  

def plot_mesh_res_tri_s(s, u, model, fig, axes, args, device):
    u = u[:, :, 2].to(device)

    # mesh
    grid = model.ori_grid
    xi1, xi2 = grid[:, [0]], grid[:, [1]]
    xi1.requires_grad = True
    xi2.requires_grad = True
    xi = torch.cat((xi1, xi2), dim=-1)

    grid1 = np.linspace(0, 1, s)
    grid2 = np.linspace(0, 1, s)
    grid = torch.tensor(np.array(np.meshgrid(grid1, grid2)), dtype=torch.float).reshape(2, -1).permute(1, 0).to(device)
    xi1_, xi2_ = grid[:, [0]], grid[:, [1]]
    xi1_.requires_grad = True
    xi2_.requires_grad = True
    xi_ = torch.cat((xi1_, xi2_), dim=-1)

    n = int(np.sqrt(u.shape[-1]))
    uni_grid = torch.tensor(np.array(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))), dtype=torch.float)\
                    .reshape(2, -1).permute(1, 0).to(device)

    plt.xticks([0, int(u.shape[-1]/2-1), n-1],['0.0','0.5','1.0'])
    plt.yticks([0, int(u.shape[-1]/2-1), n-1],['0.0','0.5','1.0'])
    
    for i in range(5):
        # t = 0
        t = 6*i + 5
        plt.subplot(1, 5, i+1)
        plt.title('t={}'.format(t), fontsize=18)
        ax = axes[i]

        x1_ = uni_grid[:, [0]]
        x2_ = uni_grid[:, [1]]
        x1_.requires_grad = True
        x2_.requires_grad = True
        u_ = interpolate_tri(u[[t]].unsqueeze(1).repeat(1, n**2, 1).reshape(-1, u.shape[-1]), \
                        xi1.unsqueeze(0).unsqueeze(0).repeat(1, n**2, 1, 1).reshape(-1, u.shape[-1], 1), \
                        xi2.unsqueeze(0).unsqueeze(0).repeat(1, n**2, 1, 1).reshape(-1, u.shape[-1], 1), \
                        x1_.unsqueeze(1).repeat(1, u.shape[-1], 1), x2_.unsqueeze(1).repeat(1, u.shape[-1], 1))
        w = torch.ones(u_.shape).to(device)
        uni_ux = torch.autograd.grad(u_, x1_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0].reshape(1, n, n)
        uni_uy = torch.autograd.grad(u_, x2_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0].reshape(1, n, n)
        alpha = torch.sum((torch.abs(uni_ux)**2 + torch.abs(uni_uy)**2)**(1/2), dim=(-2, -1)) / (n-1)**2
        m = monitor(alpha.unsqueeze(-1).unsqueeze(-1).repeat(1, n, n), uni_ux, uni_uy)[0]
        
        norm = matplotlib.colors.Normalize(vmin=m.cpu().min(), vmax=m.cpu().max())
        im = ax.contourf(m.detach().cpu().numpy(), 50, cmap=plt.cm.binary, norm = norm)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.binary), ax=ax, format='%.2f')

        # xi_ = torch.cat((xi, t * torch.ones_like(xi1)), dim=1)
        phi = model(u[[t]], xi_)
        w = torch.ones(phi.shape).to(device)
        x1 = (torch.autograd.grad(phi, xi1_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi1_).cpu().detach().numpy() * (n-1)
        x2 = (torch.autograd.grad(phi, xi2_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi2_).cpu().detach().numpy() * (n-1)
        for j in range(s):
            for i in range(s - 1):
                plt.plot(np.concatenate((x1[i + j*s], x1[i + j*s + 1]), axis=0),\
                        np.concatenate((x2[i + j*s], x2[i + j*s + 1]), axis=0), lw=0.2, color='green')
                plt.plot(np.concatenate((x1[j + i*s], x1[j + (i+1)*s]), axis=0),\
                        np.concatenate((x2[j + i*s], x2[j + (i+1)*s]), axis=0), lw=0.2, color='green')

    return fig, axes


def plot_mesh_res_tri(u, model, fig, axes, args, device):
    u = u[:, :, 2].to(device)

    # mesh
    grid = model.ori_grid
    xi1, xi2 = grid[:, [0]], grid[:, [1]]
    xi1.requires_grad = True
    xi2.requires_grad = True
    xi = torch.cat((xi1, xi2), dim=-1)

    n = int(np.sqrt(u.shape[-1]))
    uni_grid = torch.tensor(np.array(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))), dtype=torch.float)\
                    .reshape(2, -1).permute(1, 0).to(device)

    plt.xticks([0, int(u.shape[-1]/2-1), n-1],['0.0','0.5','1.0'])
    plt.yticks([0, int(u.shape[-1]/2-1), n-1],['0.0','0.5','1.0'])
    
    for i in range(5):
        # t = 0
        t = 6*i + 5
        plt.subplot(1, 5, i+1)
        plt.title('t={}'.format(t), fontsize=18)
        ax = axes[i]

        x1_ = uni_grid[:, [0]]
        x2_ = uni_grid[:, [1]]
        x1_.requires_grad = True
        x2_.requires_grad = True
        u_ = interpolate_tri(u[[t]].unsqueeze(1).repeat(1, n**2, 1).reshape(-1, u.shape[-1]), \
                        xi1.unsqueeze(0).unsqueeze(0).repeat(1, n**2, 1, 1).reshape(-1, u.shape[-1], 1), \
                        xi2.unsqueeze(0).unsqueeze(0).repeat(1, n**2, 1, 1).reshape(-1, u.shape[-1], 1), \
                        x1_.unsqueeze(1).repeat(1, u.shape[-1], 1), x2_.unsqueeze(1).repeat(1, u.shape[-1], 1))
        w = torch.ones(u_.shape).to(device)
        uni_ux = torch.autograd.grad(u_, x1_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0].reshape(1, n, n)
        uni_uy = torch.autograd.grad(u_, x2_, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0].reshape(1, n, n)
        alpha = torch.sum((torch.abs(uni_ux)**2 + torch.abs(uni_uy)**2)**(1/2), dim=(-2, -1)) / (n-1)**2
        m = monitor(alpha.unsqueeze(-1).unsqueeze(-1).repeat(1, n, n), uni_ux, uni_uy)[0]
        
        norm = matplotlib.colors.Normalize(vmin=m.cpu().min(), vmax=m.cpu().max())
        im = ax.contourf(m.detach().cpu().numpy(), 50, cmap=plt.cm.binary, norm = norm)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.binary), ax=ax, format='%.2f')

        # xi_ = torch.cat((xi, t * torch.ones_like(xi1)), dim=1)
        if args.bound_constraint == 'soft':
            phi = model(u[[t]], xi)
        else:
            phi = ((xi1**2) * (xi2**2) * ((xi1-1)**2) * ((xi2-1)**2)) * model(u, xi) + (1/2) * (xi1**2) + (1/2) * (xi2**2)
        w = torch.ones(phi.shape).to(device)
        x1 = xi1[:, 0].cpu().detach().numpy() * (n-1)
        x2 = xi2[:, 0].cpu().detach().numpy() * (n-1)
        # triangulation = tri.Triangulation(x1, x2)
        # plt.triplot(triangulation, '-', linewidth=0.1, c='b')  
        points = np.column_stack((x1, x2))  
        tri = Delaunay(points)  
        triangles_indices = tri.simplices  
        # plt.triplot(x1, x2, tri.simplices, '-', linewidth=0.1, c='b')
        x1 = (torch.autograd.grad(phi, xi1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi1)[:, 0].cpu().detach().numpy() * (n-1)
        x2 = (torch.autograd.grad(phi, xi2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi2)[:, 0].cpu().detach().numpy() * (n-1)
        # triangulation = tri.Triangulation(x1, x2)
        # plt.triplot(triangulation, '-', linewidth=0.1, c='g')  
        plt.triplot(x1, x2, tri.simplices, '-', linewidth=0.1, c='g')

    return fig, axes


def plot_mesh_res(s, u, model, fig, axes, args, device):

    # mesh
    grid1 = np.linspace(0, 1, s)
    grid2 = np.linspace(0, 1, s)
    grid = torch.tensor(np.array(np.meshgrid(grid1, grid2)), dtype=torch.float).reshape(2, -1).permute(1, 0).to(device)
    xi1, xi2 = grid[:, [0]], grid[:, [1]]
    xi1.requires_grad = True
    xi2.requires_grad = True
    xi = torch.cat((xi1, xi2), dim=-1)

    # monitor function
    u = u.to(device)
    ux = diff_x(u) * (u.shape[-1] - 1)
    uy = diff_y(u) * (u.shape[-1] - 1)
    alpha = torch.sum((torch.abs(ux)**2 + torch.abs(uy)**2)**(1/2), dim=(-2, -1)) / (u.shape[-1]-1)**2

    plt.xticks([0, int(u.shape[-1]/2-1), u.shape[-1]-1],['0.0','0.5','1.0'])
    plt.yticks([0, int(u.shape[-1]/2-1), u.shape[-1]-1],['0.0','0.5','1.0'])
    m = monitor(alpha.unsqueeze(-1).unsqueeze(-1).repeat(1, ux.shape[-1], ux.shape[-1]), ux, uy)
    norm = matplotlib.colors.Normalize(vmin=m.cpu().min(), vmax=m.cpu().max())
    
    for i in range(5):
        t = 22*i + 22
        plt.subplot(1, 5, i+1)
        plt.title('t={}'.format(t), fontsize=18)
        ax = axes[i]
        im = ax.contourf(m[t].cpu().numpy(), 50, cmap=plt.cm.binary, norm = norm)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.cm.binary), ax=ax, format='%.2f')

        # xi_ = torch.cat((xi, t * torch.ones_like(xi1)), dim=1)
        if args.bound_constraint == 'soft':
            phi = model(u[[t]], xi)
        else:
            phi = ((xi1**2) * (xi2**2) * ((xi1-1)**2) * ((xi2-1)**2)) * model(u, xi) + (1/2) * (xi1**2) + (1/2) * (xi2**2)
        w = torch.ones(phi.shape).to(device)
        x1 = (torch.autograd.grad(phi, xi1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi1).cpu().detach().numpy() * (u.shape[-1]-1)
        x2 = (torch.autograd.grad(phi, xi2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi2).cpu().detach().numpy() * (u.shape[-1]-1)
        for j in range(s):
            for i in range(s - 1):
                plt.plot(np.concatenate((x1[i + j*s], x1[i + j*s + 1]), axis=0),\
                        np.concatenate((x2[i + j*s], x2[i + j*s + 1]), axis=0), lw=0.2, color='black')
                plt.plot(np.concatenate((x1[j + i*s], x1[j + (i+1)*s]), axis=0),\
                        np.concatenate((x2[j + i*s], x2[j + (i+1)*s]), axis=0), lw=0.2, color='black')

    return fig, axes

