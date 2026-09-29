# update the mixture of skew Gaussian distribution
#%%
import numpy as np
import warpu.GaussSkewMix as gsm
import matplotlib.pyplot as plt
from vifitting.updatemixSkew import skew_cvi_general
#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--setID', type=int, default=1)
parser.add_argument('--theta_d', type=int, default=100)

args = parser.parse_args()
theta_d = args.theta_d
setID = args.setID

print("The current setting is %d, %d" % (setID, theta_d))

#%%
# setID = 1
# theta_d = 10


if setID == 1:
    import params.eg01setup as expset
    import params.eg01skewVIparams as viparams
elif setID == 2:
    import params.eg02setup as expset
    import params.eg02params as viparams

params = viparams.get_params(theta_d)
KComp = expset.KComp


#%%
maxIter = params["maxIter"]
lr = params["lr"]
nSamples = params["nSamples"]
lr_prec = params["lr_prec"]
if "lr_w" in params:
    lr_w = params["lr_w"]
else:
    lr_w = None
#%%

simu_setup = expset.simuSKT( theta_d)
targetD = simu_setup.get_targetD()
logq_fun = targetD.logphimix_batch 
def logq_grad(theta):
    _, loggrad = targetD.logphimix_val_grad_batch(theta)
    return loggrad

target_val_grad_batch = targetD.logphimix_val_grad_batch
#%%
initfile = expset.basepath + "gaussVIV2_" + str(theta_d) + ".npz"
init = np.load(initfile)
Prec_init = init["Prec"]

#%%
mu_init = simu_setup.mu.copy()
alpha_init = np.zeros((KComp, theta_d))
ww = np.ones(KComp)/KComp

# Prec_init = np.zeros((KComp, theta_d, theta_d))
# for psi in range(KComp):
#     Prec_init[psi] =  np.linalg.inv(covs_init[psi])
    
phimix =  gsm.GaussSkewMix(KComp, theta_d) 
for iterI in range(KComp):
    Prec_L = np.linalg.cholesky(Prec_init[iterI])
    phimix.set_parameter(iterI, mu_init[iterI],\
           Prec_L,  ww[iterI],  alpha_init[iterI])
phimix.setlogq(logq_fun)
phimix.setlogq_grad(logq_grad)


#%%
# lr_prec = 1e-3
# lr = 1e-3
# maxIter = 2000
# nSamples = 100
kl_trace = skew_cvi_general(phimix, target_val_grad_batch, 
                                nSamples, maxIter,
                                mu_init, Prec_init, alpha_init, ww, 
                                lr, lr_prec = lr_prec, lr_w = lr_w, printGap=500)



#%%
savefile = expset.basepath + "skewVI_" + str(theta_d) + ".npz"
np.savez(savefile, kl_trace = kl_trace, mu = mu_init, 
         Prec = Prec_init, alpha = alpha_init, ww = ww)

print("Finished!")
print()
# %%
for psi in range(KComp):
    print("psi = %d" % psi) 
    Cov = np.linalg.inv(Prec_init[psi])
    covMat = np.linalg.inv(simu_setup.Smat[psi])
    diff = np.abs(Cov - covMat) 
    print("Cov Diff max = %0.4f and median = %0.4f" % (np.max(diff), np.median(diff)))
    diagRel = np.diag(diff) / np.diag(covMat)
    print("ref diag loss =  %0.4f" % np.mean(diagRel))
    alpha_normlized = alpha_init[psi] / (np.sqrt(np.sum(alpha_init[psi]**2)) + 1e-10)
    alpha0_normlized = simu_setup.alpha[psi] / (np.sqrt(np.sum(simu_setup.alpha[psi]**2)) + 1e-10)
    print("alpha loss =  %0.4f" % np.sum(alpha_normlized * alpha0_normlized))
    print("-----------------")

# %%
