import torch
import sys
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm
# from einops import rearrange

from IPython import embed

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
                 hidden_features,
                 time_window,
                 n_variables):
        super(GNN_Layer_FS_2D, self).__init__(node_dim=-2, aggr='mean')

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 2 + n_variables, hidden_features),
                                           nn.ReLU()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                           nn.ReLU()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          nn.ReLU()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          nn.ReLU()
                                          )

        self.norm = BatchNorm(hidden_features)

    def forward(self, x, u, pos_x, pos_y, variables, edge_index, batch):
        """ Propagate messages along edges """
        x = self.propagate(edge_index, x=x, u=u, pos_x=pos_x, pos_y=pos_y, variables=variables)
        x = self.norm(x)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_x_i, pos_x_j, pos_y_i, pos_y_j, variables_i):
        """ Message update """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_x_i - pos_x_j, pos_y_i - pos_y_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """ Node update """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        return x + update


class MP_PDE_Solver_2D(torch.nn.Module):
    def __init__(
            self,
            pde,
            time_window=1,
            hidden_features=128,
            hidden_layer=6,
            eq_variables={}
    ):

        super(MP_PDE_Solver_2D, self).__init__()
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window
        self.eq_variables = eq_variables

        # in_features have to be of the same size as out_features for the time being
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer_FS_2D(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer)))

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window + 3 + len(self.eq_variables), self.hidden_features),
            nn.BatchNorm1d(self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.BatchNorm1d(self.hidden_features)
            #Swish()
            )

        self.output_mlp = nn.Sequential(nn.Conv1d(1, 4, 16, stride=3),
                                        # nn.BatchNorm1d(8),
                                        nn.ReLU(),
                                        nn.Conv1d(4, 8, 12, stride=3),
                                        nn.ReLU(),
                                        nn.Conv1d(8, 1, 8, stride=2)
                                                )

    def __repr__(self):
        return f'GNN'
        
    def forward(self, data):
        u = data.x
        pos = data.pos
        pos_x = pos[:, 1][:, None]/self.pde.Lx
        pos_y = pos[:, 2][:, None]/self.pde.Ly
        pos_t = pos[:, 0][:, None]/self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        variables = pos_t    # we put the time as equation variable

        node_input = torch.cat((u, pos_x, pos_y, variables), -1)
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos_x, pos_y, variables, edge_index, batch)


        diff = self.output_mlp(h[:, None]).squeeze(1)
        dt = (torch.ones(1, self.time_window) * self.pde.dt * 0.1).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        out = dt * diff

        return out