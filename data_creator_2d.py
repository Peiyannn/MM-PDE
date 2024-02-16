import os

import h5py
import numpy as np
import torch
import sys

from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph
from sklearn.neighbors import NearestNeighbors  
from interpolate import ItpNet
# from einops import rearrange


class GraphCreator_FS_2D(nn.Module):
    """
    Helper class to construct graph datasets
    params:
        neighbors: now many neighbors the graph has in each direction
        time_window: how many time steps are used for PDE prediction
        time_ratio: time ratio between base and super resolution
        space_ratio: space ratio between base and super resolution
    """

    def __init__(self,
                 pde,
                 neighbors: int=2,
                 connect_edge: str='knn',
                 time_window: int=10,
                 t_resolution: int=100,
                 ):
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.e = connect_edge
        self.tw = time_window
        self.t_res = t_resolution

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)

    
    def interpolate(self, itp_model, u, init_x, init_y, x, y, mode):
        """
        u: (nu,nx,ny)
        init_x: (nu*nx*ny,1)
        init_y: (nu*nx*ny,1)
        x: (nu*nx'*ny',1)
        y: (nu*nx'*ny',1)
        return: interpolated: (nu*nx'*ny')
        """
        nu = u.shape[0]
        nx = u.shape[-2]
        ny = u.shape[-1]
        output_res = int(x.shape[0] / nu)

        all_points = torch.cat((init_x, init_y), -1).reshape(nu, -1, 2)
        all_query_points = torch.cat((x, y), dim=-1).reshape(nu, -1, 2) # (8, 2304, 2)
        if mode == '1':
            n_neighbors = 30
        elif mode == '2':
            n_neighbors = 30
        knn = NearestNeighbors(n_neighbors=n_neighbors)  
        weights = []
        neighbors = []
        neighbor_labels = []
        for k in range(nu):
            labels = u[k].reshape(-1)
            points = all_points[k]
            query_points = all_query_points[k]

            knn.fit(points.detach().cpu().numpy())
            distances, indices = knn.kneighbors(query_points.detach().cpu().numpy())  
            neighbors.append(points[indices].to(u.device))
            neighbor_labels.append(labels[indices])

        neighbors = torch.stack(neighbors) # [8, 2304, n, 2]
        neighbor_labels = torch.stack(neighbor_labels) # [8, 2304, n]
        weights = itp_model(neighbors, all_query_points.unsqueeze(-2), mode)
        interpolated = torch.sum(weights * neighbor_labels, dim=-1).reshape(-1)
        
        return interpolated

    
    def moving_mesh(self, u, mesh_model, n_grid_x, n_grid_y):
        """
        getting the moved mesh
        u: (nu,nx,ny)
        return: x1, x2: (nu*nx*ny, 1)
        """
        grid_x = np.linspace(0, self.pde.Lx, n_grid_x)
        grid_y = np.linspace(0, self.pde.Ly, n_grid_y)
        grid = torch.tensor(np.array(np.meshgrid(grid_x, grid_y)), dtype=torch.float).reshape(2, -1).permute(1, 0).to(u.device)
        xi1, xi2 = grid[:, [0]].unsqueeze(0).repeat(u.shape[0], 1, 1).reshape(-1, 1), grid[:, [1]].unsqueeze(0).repeat(u.shape[0], 1, 1).reshape(-1, 1)
        xi1.requires_grad = True
        xi2.requires_grad = True
        xi = torch.cat((xi1, xi2), dim=-1)

        if self.pde.movingmesh_grid_size[-2] != n_grid_x or self.pde.movingmesh_grid_size[-1] != n_grid_y:
            u = F.interpolate(u.reshape(-1, 1, u.shape[-2], u.shape[-1]), size=(self.pde.movingmesh_grid_size[-2], self.pde.movingmesh_grid_size[-1]), mode='bilinear', align_corners=True).squeeze(1)
        phi = mesh_model(u, xi)
        w = torch.ones(phi.shape).to(u.device)
        x1 = (torch.autograd.grad(phi, xi1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi1)
        x2 = (torch.autograd.grad(phi, xi2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi2)

        alpha = 1
        x1 = alpha * x1 + (1 - alpha) * xi1
        x2 = alpha * x2 + (1 - alpha) * xi2
        
        return x1, x2

    def moving_mesh_tri(self, u, mesh_model, grid_x, grid_y):
        """
        getting the moved mesh
        u: (nu,n)
        grid_x: (n)
        grid_y: (n)
        return: x1, x2: (nu*n, 1)
        """
        xi1, xi2 = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
        xi1.requires_grad = True
        xi2.requires_grad = True
        xi = torch.cat((xi1, xi2), dim=-1)

        phi = mesh_model(u, xi)
        w = torch.ones(phi.shape).to(u.device)
        x1 = (torch.autograd.grad(phi, xi1, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi1)
        x2 = (torch.autograd.grad(phi, xi2, grad_outputs=w, retain_graph = True, create_graph=True, allow_unused=True)[0] + xi2)

        alpha = 1
        x1 = alpha * x1 + (1 - alpha) * xi1
        x2 = alpha * x2 + (1 - alpha) * xi2
        
        return x1, x2

    def create_data(self, datapoints, steps):
        """
        getting data out of PDEs
        """
        data = torch.Tensor()
        labels = torch.Tensor()

        for (dp, step) in zip(datapoints, steps):
            # d = dp[step - self.tw*2:step]
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]

            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels


    def create_graph(self, itp_model, data, labels, steps, device, mesh_model=None):
        """
        getting moved mesh and interpolate data
        getting graph structure out of data sample
        previous timesteps are combined in one node
        """
        data = data.to(device)
        labels = labels.to(device)

        if len(self.pde.grid_size) == 3:
            # h = 2
            ori_nx = data.shape[-2]
            ori_ny = data.shape[-1]
            ori_x = torch.linspace(0, self.pde.Lx, ori_nx).to(device)
            ori_y = torch.linspace(0, self.pde.Ly, ori_ny).to(device)
            ori_grid_x, ori_grid_y = torch.meshgrid(ori_x, ori_y)

            # h = 2
            mm_nx = self.pde.movingmesh_grid_size[-2]
            mm_ny = self.pde.movingmesh_grid_size[-1]
            mm_x = torch.linspace(0, self.pde.Lx, mm_nx).to(device)
            mm_y = torch.linspace(0, self.pde.Ly, mm_ny).to(device)
            mm_grid_x, mm_grid_y = torch.meshgrid(mm_x, mm_y)

            nt = self.pde.grid_size[0]
            nx = self.pde.grid_size[1]
            ny = self.pde.grid_size[2]
            n = nx * ny
            t = torch.linspace(self.pde.tmin, self.pde.tmax, nt).to(device)
            dt = t[1] - t[0]
            x = torch.linspace(0, self.pde.Lx, nx).to(device)
            dx = x[1]-x[0]
            y = torch.linspace(0, self.pde.Ly, ny).to(device)
            dy = y[1]-y[0]
            
            grid_x, grid_y = torch.meshgrid(x, y)
            grid = torch.stack((grid_x, grid_y), 2).float()
            grid = grid.view(-1, 2)[None].repeat(data.shape[0], 1, 1)
            radius = self.n * torch.sqrt(dx**2 + dy**2) + 0.0001

            if mesh_model != None:
                mesh_x, mesh_y = self.moving_mesh(data.reshape(-1, ori_nx, ori_ny)[:, ::int(ori_nx / mm_nx), ::int(ori_ny / mm_ny)], mesh_model, nx, ny)
                mesh = torch.cat((mesh_x, mesh_y), dim=-1).reshape(-1, nx*ny, 2)
                
            else:
                mesh_x, mesh_y = grid[:, :, 0].reshape(-1, 1), grid[:, :, 1].reshape(-1, 1)
                mesh = grid
            
            if mesh_model != None:
                data = self.interpolate(itp_model, data.reshape(-1, ori_nx, ori_ny), ori_grid_x[None].repeat(data.shape[0], 1, 1).reshape(-1, 1), ori_grid_y[None].repeat(data.shape[0], 1, 1).reshape(-1, 1),\
                                        mesh_x, mesh_y, mode='1').reshape(-1, self.tw, nx, ny)
                labels = self.interpolate(itp_model, labels.reshape(-1, ori_nx, ori_ny), ori_grid_x[None].repeat(data.shape[0], 1, 1).reshape(-1, 1), ori_grid_y[None].repeat(data.shape[0], 1, 1).reshape(-1, 1),\
                                            mesh_x, mesh_y, mode='1').reshape(-1, self.tw, nx, ny)

        elif len(self.pde.grid_size) == 2:
            # h = 2
            n = self.pde.ori_grid_size[1]
            grid = self.pde.ori_grid[None].repeat(data.shape[0], 1, 1).to(device)
            grid_x, grid_y = grid[:, :, 0], grid[:, :, 1]

            nt = self.pde.grid_size[0]
            nx = int(np.sqrt(self.pde.grid_size[1]))
            ny = int(np.sqrt(self.pde.grid_size[1]))
            t = torch.linspace(self.pde.tmin, self.pde.tmax, nt).to(device)
            dt = t[1] - t[0]
            x = torch.linspace(0, self.pde.Lx, nx).to(device)
            dx = x[1]-x[0]
            y = torch.linspace(0, self.pde.Ly, ny).to(device)
            dy = y[1]-y[0]
            radius = self.n * torch.sqrt(dx**2 + dy**2) + 0.0001

            if mesh_model != None:
                mesh_x, mesh_y = self.moving_mesh_tri(data.reshape(-1, n), mesh_model, grid_x, grid_y)
                mesh = torch.cat((mesh_x, mesh_y), dim=-1).reshape(-1, n, 2)

            else:
                mesh_x, mesh_y = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)
                mesh = grid

        u_new = torch.Tensor().to(device)
        x_new = torch.Tensor().to(device)
        grid_new = torch.Tensor().to(device)
        t_new = torch.Tensor().to(device)
        y_new = torch.Tensor().to(device)
        batch = torch.Tensor().to(device)
        for b, (data_batch, mesh_batch, grid_batch, labels_batch, step) in enumerate(zip(data, mesh, grid, labels, steps)):
            u_tmp = torch.transpose(torch.cat([d.reshape(-1, n) for d in data_batch]), 0, 1)
            y_tmp = torch.transpose(torch.cat([l.reshape(-1, n) for l in labels_batch]), 0, 1)

            u_new = torch.cat((u_new, u_tmp), )
            x_new = torch.cat((x_new, mesh_batch), )
            grid_new = torch.cat((grid_new, grid_batch), )
            y_new = torch.cat((y_new, y_tmp), )
            b_new = torch.ones(n).to(device)*b
            t_tmp = torch.ones(n).to(device)*t[step]

            t_new = torch.cat((t_new, t_tmp), )
            batch = torch.cat((batch, b_new), )

        # calculating the edge_index
        if self.e == 'radius':
            edge_index = radius_graph(x_new, r=radius, batch=batch.long(), loop=False)
        elif self.e == 'knn':
            edge_index = knn_graph(x_new, k=self.n, batch=batch.long(), loop=False)

        graph = Data(x=u_new, edge_index=edge_index)
        graph.y = y_new
        graph.pos = torch.cat((t_new[:, None], x_new), 1)
        graph.batch = batch.long()

        return graph.to(device)


    def interpolate_pred(self, itp_model, pred, graph, data, device):
        """
        interpolating the prediction to the uniform mesh
        """
        data = data.to(device)

        if len(self.pde.grid_size) == 3:
            ori_nx = self.pde.ori_grid_size[1]
            ori_ny = self.pde.ori_grid_size[2]
            ori_x = torch.linspace(0, self.pde.Lx, ori_nx).to(device)
            ori_y = torch.linspace(0, self.pde.Ly, ori_ny).to(device)
            ori_grid_x, ori_grid_y = torch.meshgrid(ori_x, ori_y)

            nx = self.pde.grid_size[1]
            ny = self.pde.grid_size[2]
            nu = int(pred.shape[0] / (nx * ny))
            x = torch.linspace(0, self.pde.Lx, nx).to(device)
            y = torch.linspace(0, self.pde.Ly, ny).to(device)
            grid_x, grid_y = torch.meshgrid(x, y)

            pred_grid = self.interpolate(itp_model, pred.reshape(-1, nx, ny), graph.pos[:, [1]], graph.pos[:, [2]], ori_grid_x[None].repeat(nu, 1, 1).reshape(-1, 1),\
                                    ori_grid_y[None].repeat(nu, 1, 1).reshape(-1, 1), mode='2').reshape(-1, 1, ori_nx, ori_ny)

            out = itp_model(None, None, mode = 'res_cut', data = data).reshape(-1, 1, ori_nx, ori_ny) + pred_grid

        elif len(self.pde.grid_size) == 2:
            n = self.pde.ori_grid_size[1]
            nu = int(pred.shape[0] / n)
            grid_x, grid_y = self.pde.ori_grid[:, 0].to(device), self.pde.ori_grid[:, 1].to(device)

            pred_grid = self.interpolate(itp_model, pred.reshape(-1, n), graph.pos[:, [1]], graph.pos[:, [2]], grid_x[None].repeat(nu, 1).reshape(-1, 1),\
                                    grid_y[None].repeat(nu, 1).reshape(-1, 1), mode='2').reshape(-1, n)

            out = itp_model(None, None, mode = 'res_cut', data = data.reshape(-1, n)).reshape(-1, n) + pred_grid

        return out.reshape(-1, 1)
