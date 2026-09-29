####
# Fitting neural transformed mixture to the target distribution
###
#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from neuralode.networkBase import *
from neuralode.deepphimixv2 import deepphimixClass_v2
import numpy.random as rd
from scipy.stats import norm as gauss
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--setID', type=int, default=1)
parser.add_argument('--theta_d', type=int, default=100)
parser.add_argument('--phimix_type', type=int, default=1)
args, unknown = parser.parse_known_args()

if len(sys.argv) > 2:
    theta_d = args.theta_d
    setID = args.setID
    phimix_type = args.phimix_type
else:
    setID = 3
    phimix_type = 1
    theta_d = 30


#%%
if setID==3:
    import params.eg03mcmc as mcmcset
hidden_dim = mcmcset.hidden_dim_neural
width = mcmcset.width_neural
lr = mcmcset.lr_neural
niters = mcmcset.niters_neural
num_samples = mcmcset.nSample_neural
t0 = mcmcset.t0_neural
t1 = mcmcset.t1_neural
ode_stepsize = mcmcset.ode_stepsize_neural

#%%
if setID==3:
    import params.eg03setup as expset
    simu_setup = expset.simuSshape( theta_d)
    if phimix_type == 1:
        stName = "sshape_V2K2"
    elif phimix_type == 2:
        stName = "sshape_V2K6"


targetD = simu_setup.get_targetD()
logq_fun = targetD.logphimix_batch 
def logq_grad(theta):
    _, loggrad = targetD.logphimix_val_grad_batch(theta)
    return loggrad

target_val_grad_batch = targetD.logphimix_val_grad_batch
base_path = expset.basepath
phimix_base = mcmcset.get_phimix(phimix_type, base_path, 30)
KComp = phimix_base.get_K()
phimix_base.setlogq(logq_fun)
phimix_base.setlogq_grad(logq_grad)
#%%


device = torch.device('cuda:0'
                        if torch.cuda.is_available() else 'cpu')


#%%
def train_mode(neuralPhimix, phimix_base,  optimList, 
               niters=500, figsavepath=None): 
    loss_meter_kl = RunningAverageMeter()
    samples_all = np.zeros((num_samples*KComp, theta_d))
    #logp_all = np.zeros((KComp, num_samples))
    for psi in range(KComp):
        neuralPhimix.func_list[psi].trace_compute = True
        neuralPhimix.func_list[psi].training = True

    for itr in range( niters + 1 ):
        kldiv = 0.0
        sample_base = rd.randn(num_samples, theta_d).astype(np.float32)
        logphi_base = np.sum(gauss.logpdf(sample_base), axis=1)
        sample_base_tensor = torch.tensor(sample_base, dtype=torch.float32).to(device)

        logp_all = torch.zeros(KComp, num_samples, dtype=torch.float32)

        for psi in range(KComp): 
            optimList[psi].zero_grad()
            z_t01, logp_diff_t0 = neural_sample(sample_base_tensor, neuralPhimix.func_list[psi],
                                                phimix_base, psi, t0, t1, ode_stepsize)

            logp_x =  wrapup.apply(z_t01, phimix_base, psi)
            logp_x +=  logp_diff_t0.view(-1)
            loss_psi = -torch.mean(logp_x)
            loss_psi.backward()
            optimList[psi].step()
            logp_all[psi] = logp_x.detach().cpu()
            if itr % 50 == 1:
                samples_all[psi*num_samples:(psi+1)*num_samples, :] = z_t01.cpu().detach().numpy()

        #loss = -torch.logsumexp(logp_all, dim=0).mean()
        #loss.backward()

        #for psi in range(KComp):
        #    optimList[psi].step()

        logq_trans = torch.logsumexp(logp_all, dim=0).detach().numpy()
        kldiv = np.mean(logphi_base - logq_trans)
        loss_meter_kl.update(kldiv)

        if itr % 50 == 1:
            print('Iter: {}, KL loss: {:.4f}'.format(itr, loss_meter_kl.avg))
            # if figsavepath is not None:
            #     fig, ax = plt.subplots(1,2, figsize=(10,5))
            #     ax[0].scatter(logphi_base.flatten(), logq_trans.flatten(), s=1)
            #     xseq = np.linspace(logphi_base.min(), logphi_base.max(), 100)
            #     ax[0].plot(xseq, xseq, 'r--')
            #     ax[1].hist2d(samples_all[:,0], samples_all[:,1], bins=50, density=True)            
            #     fig.savefig(f"{figsavepath}/psi_{psi}_iter_{itr}.png")
            #     plt.show()
            torch.cuda.empty_cache()



#%%
neuralPhimix = deepphimixClass_v2(theta_d, phimix_base)
neuralPhimix.set_ode_param(t0, t1, ode_stepsize)
neuralPhimix.set_network(width, hidden_dim)
neuralPhimix.setlogq(logq_fun)
optimList = list()
for k in range(KComp):
    optimList.append(optim.Adam(neuralPhimix.func_list[k].parameters(), lr= 1e-4))


#%%
import time
start = time.time()
np.random.seed(123)
func2 = train_mode(neuralPhimix, phimix_base, optimList, 
                   niters=niters, figsavepath = "./figs/tmp")
end = time.time()
print("Time taken: ", end-start)


    # %%
neuralPhimix.save_network(base_path, stName)
