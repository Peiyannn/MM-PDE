import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm
from torch_cluster import radius_graph, knn_graph
from torch_geometric.data import Data


class DenseNet(nn.Module):
    def __init__(self, layers, width=32, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = nn.ModuleList()
        self.act = torch.tanh
        self.normalize = normalize

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

        self.width = width
        self.center = torch.tensor([0.5,0.5], device="cuda").reshape(1,2)
        self.B = np.pi*torch.pow(2, torch.arange(0, self.width//4, dtype=torch.float, device="cuda")).reshape(1,1,1,self.width//4)
        self.fc0 = nn.Linear(4, self.width)

    def forward(self, x):
        if self.normalize:
            for _, l in enumerate(self.layers):
                if _ != 2 * self.n_layers - 2 and _%2 != 1:
                    x = self.act(l(x))
                else:
                    out = l(x)
        else:
            for _, l in enumerate(self.layers):
                if _ != self.n_layers - 1:
                    x = self.act(l(x))
                else:
                    out = l(x)

        return out, x


class ConvNet(nn.Module):
    def __init__(self, s, layers):
        super().__init__()
        self.layers = nn.ModuleList()

        if layers == 7:
            self.layers.append(nn.Conv2d(1, 8, 5, stride=2, padding=2))
            self.layers.append(nn.Conv2d(8, 16, 5, padding=2))
            self.layers.append(nn.Conv2d(16, 8, 5, padding=2))
            self.layers.append(nn.Conv2d(8, 1, 5, stride=2, padding=2))
            self.fc1 = None
            self.fc2 = nn.Linear(int(((s + 1) / 2 + 1) / 2)**2, 1024)
            self.fc3 = nn.Linear(1024, 512) # burgers: 1024,512

        self.act = torch.tanh


    def forward(self, x):
        for i, l in enumerate(self.layers):
            # x = self.pool(self.act(l(x)))
            if i != len(self.layers) - 2:
                x = self.act(l(x))
            if i == 0:
                ori_x = x
            if i == len(self.layers) - 2:
                x = self.act(ori_x + l(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = self.bn(x)
        if self.fc1 != None:
            x = self.act(self.fc1(x))
        if self.fc2 != None:
            x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x



class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer_FS_2D(MessagePassing):
    """
    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    hidden_features : int
        Dimensionality of hidden features.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,):
        super(GNN_Layer_FS_2D, self).__init__(node_dim=-2, aggr='mean')

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + 3, hidden_features),
                                           nn.Tanh()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                           nn.Tanh()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features),
                                          nn.Tanh()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          nn.Tanh()
                                          )

        self.norm = BatchNorm(hidden_features)

    def forward(self, x, u, pos_x, pos_y, edge_index, batch):
        """ Propagate messages along edges """
        x = self.propagate(edge_index, x=x, u=u, pos_x=pos_x, pos_y=pos_y)
        x = self.norm(x)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_x_i, pos_x_j, pos_y_i, pos_y_j):
        """ Message update """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_x_i - pos_x_j, pos_y_i - pos_y_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        """ Node update """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return x + update


class DMM(nn.Module):
    def __init__(self, branch_layer, trunk_layer, grid=None, out_layer=None, s=None, mode='array'):
        super(DMM, self).__init__()
        self.mode = mode
        self.ori_grid = grid
        if mode == 'array':
            self.branch = ConvNet(s, branch_layer)
            self.trunk = DenseNet(trunk_layer)
            self.out_nn = DenseNet(out_layer)
        elif mode == 'graph':
            self.hidden_features = branch_layer[0]
            self.hidden_layer = branch_layer[1]

            # in_features have to be of the same size as out_features for the time being
            self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer_FS_2D(
                in_features=self.hidden_features,
                hidden_features=self.hidden_features,
                out_features=self.hidden_features,
                ) for _ in range(self.hidden_layer)))

            self.embedding_mlp = nn.Sequential(
                nn.Linear(3, self.hidden_features),
                nn.BatchNorm1d(self.hidden_features),
                nn.Tanh(),
                nn.Linear(self.hidden_features, self.hidden_features),
                nn.BatchNorm1d(self.hidden_features)
                #Swish()
                )
            self.decoding_mlp = DenseNet([self.hidden_features, 128, 1])

            self.output_mlp = nn.Sequential(
                                        nn.Linear(grid.shape[0], 512),
                                        nn.Tanh(),
                                        nn.Linear(512, 256),
                                        nn.Tanh(),
                                        nn.Linear(256, trunk_layer[-1])
                                                )
            self.trunk = DenseNet(trunk_layer)
            self.out_nn = DenseNet(out_layer)

    def forward(self, u, grid, rf = False):
        if self.mode == 'array':
            branch = self.branch(u.unsqueeze(1)).unsqueeze(1).repeat(1, int(grid.shape[0]/u.shape[0]), 1)
            # (batchsize, S, S) -> (batchsize, latent) -> (batchsize, grid_per_u, latent)
            trunk, second_out = self.trunk(grid)
            out, second_out = self.out_nn(torch.cat((branch.reshape(-1, branch.shape[-1]), trunk.reshape(-1, branch.shape[-1])), dim=-1))
            if rf == False:
                return out # (batchsize)
            else:
                return out, second_out, torch.ones_like(second_out).type_as(trunk).reshape(-1, 1)

        elif self.mode == 'graph':
            data = create_graph(u, self.ori_grid, device=u.device)
            x = data.x
            pos = data.pos
            pos_x = pos[:, 0][:, None]
            pos_y = pos[:, 1][:, None]
            edge_index = data.edge_index
            batch = data.batch

            node_input = torch.cat((x, pos_x, pos_y), -1)
            h = self.embedding_mlp(node_input)
            for i in range(self.hidden_layer):
                h = self.gnn_layers[i](h, x, pos_x, pos_y, edge_index, batch)
            h, _ = self.decoding_mlp(h)
            branch = self.output_mlp(h.reshape(u.shape[0], 1, -1)).repeat(1, int(grid.shape[0]/u.shape[0]), 1)
            trunk, _ = self.trunk(grid)

            out, second_out = self.out_nn(torch.cat((branch.reshape(-1, branch.shape[-1]), trunk.reshape(-1, branch.shape[-1])), dim=-1))
            if rf == False:
                return out # (batchsize)
                # return trunk # (batchsize)
            else:
                return out, second_out, torch.ones_like(second_out).type_as(trunk).reshape(-1, 1)
                # return trunk, second_out, torch.ones((trunk.shape[0], 1)).type_as(trunk).reshape(-1, 1)


def create_graph(data, grid, device, n=35):
    """
    getting graph structure out of data sample
    """

    batch = torch.arange(0, data.shape[0], 1).to(device)[:, None].repeat(1, grid.shape[0]).reshape(-1)
    edge_index = knn_graph(grid[None].repeat(data.shape[0], 1, 1).reshape(-1, 2), n, batch=batch.long(), loop=False)

    graph = Data(x=data.reshape(-1, 1), edge_index=edge_index)
    graph.pos = grid[None].repeat(data.shape[0], 1, 1).reshape(-1, 2)
    graph.batch = batch.long()

    return graph.to(device)
