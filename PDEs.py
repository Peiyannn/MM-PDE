import os
import sys
import math
import numpy as np
import torch
from torch import nn


class PDE(nn.Module):
    """Generic PDE template"""
    def __init__(self):
        # Data params for grid and initial conditions
        super().__init__()
        pass

    def __repr__(self):
        return "PDE"


class burgers(PDE):
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 L: float=None,
                 flux_splitting: str=None,
                 device: torch.cuda.device = "cpu") -> None:

        # Data params for grid 
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 30 if tmax is None else tmax
        # Length of the spatial domain
        self.Lx = 1 if L is None else L
        self.Ly = 1 if L is None else L
        self.grid_size = (31, 96, 96) if grid_size is None else grid_size
        self.movingmesh_grid_size = (31, 96, 96)
        self.ori_grid_size = (31, 96, 96)
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.device = device


class cy(PDE):
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 ori_grid: torch.Tensor=None,
                 L: float=None,
                 flux_splitting: str=None,
                 device: torch.cuda.device = "cpu") -> None:

        # Data params for grid
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 2.9 if tmax is None else tmax
        # Length of the spatial domain
        self.Lx = 1 if L is None else L
        self.Ly = 1 if L is None else L
        self.grid_size = (30, 2521) if grid_size is None else grid_size
        self.ori_grid_size = (30, 2521) if grid_size is None else grid_size
        self.movingmesh_grid_size = (30, 2521) if grid_size is None else grid_size
        self.ori_grid = ori_grid
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.device = device

