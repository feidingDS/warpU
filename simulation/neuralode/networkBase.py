#%%
import torch
import torch.nn as nn
import numpy as np
from torch.func import jacfwd, vmap
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions.normal import Normal

#%%

device = torch.device('cuda:' + str(0)
                       if torch.cuda.is_available() else 'cpu')


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val




def trace_df_dz_random(f, x, num_samples=2):
    p = x.shape[1]
    n = x.shape[0]
    trace_estimate = torch.zeros(n).to(x)
    for _ in range(num_samples):
        #v = torch.randn(x.shape[1]).to(x)
        v = torch.randint(0,2,(n ,p)).to(x)
        v = 2*v - 1
        Jv = torch.autograd.grad(outputs=f, inputs=x, grad_outputs=v,  #.sum(0)
                                retain_graph=True, create_graph=True)[0]
        trace_estimate += torch.sum(Jv*v, dim=1) #torch.matmul(Jv, v)
    
    return trace_estimate / num_samples


    
def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, retain_graph=True,
                                        create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()
    




def get_batch(num_samples, theta_d): 
    m1 = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

    x = torch.randn(num_samples, theta_d).type(torch.float32).to(device) 
    xlogp1 = m1.log_prob(x).sum(1)
    return (x, xlogp1)



def neural_sample(sample_base, neuralFun, phimix, psi, t0, t1, ode_stepsize):
    backwardRange = torch.tensor([t1, t0], dtype=torch.float32).to(device)

    mu_psi = phimix.get_mu(psi)
    mu_psi_tensor = torch.tensor(mu_psi, dtype=torch.float32).to(device)
    scale_psi = phimix.get_scale(psi).T 
    scale_psi_tensor = torch.tensor(scale_psi, dtype=torch.float32).to(device)
    logdet_scale = phimix.get_logdet(psi) 
    logdet_scale = torch.tensor(logdet_scale, dtype=torch.float32).to(device)

    num_samples = sample_base.shape[0]
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)
    z_t, logp_diff_t = odeint(
        neuralFun,  (sample_base, logp_diff_t1), backwardRange,
        method='rk4', options=dict(step_size=ode_stepsize) )

    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
    z_t01 = torch.matmul(z_t0, scale_psi_tensor) + mu_psi_tensor
    logp_diff_t0 = logp_diff_t0 + logdet_scale
    return z_t01, logp_diff_t0

class wrapup(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, phimix, psi):
        nsample = input.shape[0]
        input_numpy = input.detach().cpu().numpy()
        #output = torch.zeros(nsample, dtype=torch.float32).to(input.device)
        output = phimix.logq_slice(input_numpy, psi)
        output = torch.tensor(output).to(input)
        ctx.phimix = phimix
        ctx.psi = psi
        ctx.input_numpy = input_numpy
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        phimix = ctx.phimix
        psi = ctx.psi
        input_numpy = ctx.input_numpy
        nsample = input_numpy.shape[0]
        theta_d = input_numpy.shape[1]
        grad_input_np = np.zeros((nsample, theta_d))
        grad_input_np = phimix.logq_slice_grad(input_numpy, psi)
        grad_input = torch.tensor(grad_input_np).to(grad_output)
        grad_input *= grad_output.unsqueeze(1)
        return grad_input, None, None
    