import torch
import torch.nn as nn


class ItpNet(nn.Module):
    def __init__(self, ori_nx, ori_ny, layers1, layers2, layers3, normalize=False):
        super(ItpNet, self).__init__()
        self.n = 30

        self.layers1_node = [self.n * 2 + 2] + layers1 
        self.layers1_node.append(self.n)
        self.n_layers1 = len(self.layers1_node) - 1
        assert self.n_layers1 >= 1
        self.layers = nn.ModuleList()
        self.act = torch.tanh

        for j in range(self.n_layers1):
            self.layers.append(nn.Linear(self.layers1_node[j], self.layers1_node[j+1]))

            if j != self.n_layers1 - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(self.layers1_node[j+1]))
        
        self.layers2_node = [self.n * 2 + 2] + layers2
        self.layers2_node.append(self.n)
        self.n_layers2 = len(self.layers2_node) - 1
        self.layers2 = nn.ModuleList()
        self.act2 = torch.tanh

        for j in range(self.n_layers2):
            self.layers2.append(nn.Linear(self.layers2_node[j], self.layers2_node[j+1]))

            if j != self.n_layers2 - 1:
                if normalize:
                    self.layers2.append(nn.BatchNorm1d(self.layers2_node[j+1]))

        if ori_ny != None:
            self.layers3_node = [ori_nx * ori_ny] + layers3
            self.layers3_node.append(ori_nx * ori_ny)
        else:
            self.layers3_node = [ori_nx] + layers3
            self.layers3_node.append(ori_nx)
        self.n_layers3 = len(self.layers3_node) - 1
        self.layers3 = nn.ModuleList()
        self.act3 = torch.tanh

        for j in range(self.n_layers3):
            self.layers3.append(nn.Linear(self.layers3_node[j], self.layers3_node[j+1]))

            if j != self.n_layers3 - 1:
                if normalize:
                    self.layers3.append(nn.BatchNorm1d(self.layers3_node[j+1]))

        if ori_ny != None:
            self.down = nn.Sequential(
                nn.Conv2d(layers3[0], layers3[1], 5, padding=2),
                nn.Tanh(),
                nn.Conv2d(layers3[1], layers3[2], 5, padding=2), 
                nn.Tanh(),
                nn.Conv2d(layers3[2], layers3[3], 5, padding=2), 
                nn.Tanh(),
                nn.Conv2d(layers3[3], layers3[4], 5, padding=2), 
                nn.Tanh(),
            )
        else:
            self.down = nn.Sequential(
                nn.Linear(ori_nx, 2048),
                nn.Tanh(),
                nn.Linear(2048, 512),
                nn.Tanh(),
                nn.Linear(512, 2048),
                nn.Tanh(),
                nn.Linear(2048, ori_nx),
            )
            

    def forward(self, neighbors, query_points, mode, data = None):
        
        if mode == '1':
            data = torch.cat((neighbors, query_points), dim=-2).reshape(neighbors.shape[0], neighbors.shape[1], -1)
            for _, l in enumerate(self.layers):
                if _ != self.n_layers1 - 1:
                    data = self.act(l(data))
                else:
                    data = l(data)

        elif mode == '2':
            data = torch.cat((neighbors, query_points), dim=-2).reshape(neighbors.shape[0], neighbors.shape[1], -1)
            for _, l in enumerate(self.layers2):
                if _ != self.n_layers2 - 1:
                    data = self.act2(l(data))
                else:
                    data = l(data)

        elif mode == 'res_cut':
            for _, l in enumerate(self.down):
                data = l(data)

        return data