#%%
import torch
import torch.nn as nn
import numpy as np
from torch.func import jacfwd, vmap
from neuralode.networkBase import *    


#%%
class HyperNetworkLN(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,  blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = torch.tanh(self.fc3(params))
        params = self.fc4(params)

        # restructure
        params = params.reshape(-1) #?
        W = params[:self.blocksize].reshape( self.in_out_dim, self.width)
        B = params[self.blocksize:].reshape(1, self.width)
        return (W,B)
    

class CNF(nn.Module):

    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetworkLN(in_out_dim, hidden_dim, width)
        self.hyper_net2 = HyperNetworkLN(width, hidden_dim, width)
        self.fl1 = nn.Linear(width, width)
        self.fl2 = nn.Linear(width, width)
        self.fl3 = nn.Linear(width, in_out_dim)
        self.forward_count = 0
        self.trace_compute = True

    def set_trace_compute(self, val):
        self.trace_compute = val

    def forward(self, t, states):
        Z = states[0]
        batchsize = Z.shape[0]

        with torch.set_grad_enabled(True):
            Z.requires_grad_(True)
            dz_dt = self.dz_dtfun(Z, t)

            if not self.trace_compute:
                return dz_dt, torch.zeros(batchsize, 1).to(dz_dt)

            if self.training:
                dlogp_z_dt = trace_df_dz_random(dz_dt, Z).view(batchsize, 1)
            else:
                ttensor = t.view(1)
                Jacob = vmap(jacfwd(self.dz_dtfun, argnums=0), in_dims=(0, None) )(Z, ttensor)
                dlogp_z_dt = torch.diagonal(Jacob.squeeze(1), dim1 = 1, dim2 = 2).sum(1).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)

    def dz_dtfun(self, Z, t):
        W, B = self.hyper_net(t)
        H = torch.tanh(torch.matmul(Z, W) + B)

        W2, B2 = self.hyper_net2(t)
        H2 = torch.tanh(torch.matmul(H, W2) + B2)
        #dz_dt = torch.matmul(h, U).mean(0)
        H3 = torch.tanh(self.fl1(H2))
        H4 = torch.tanh(self.fl2(H3))
        dz_dt = self.fl3(H4)
        return dz_dt

