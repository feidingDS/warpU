# update the mixture of gamma Gaussian distribution
#%%
import numpy as np
import warpu.GaussGammaMix as ggm
import matplotlib.pyplot as plt
from vifitting.updatemixt import gamma_cvi_general
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
    import params.eg01gammaVIparams as viparams
elif setID == 2:
    import params.eg02setup as expset
    import params.eg02params as viparams

KComp = expset.KComp
params = viparams.get_params(theta_d)
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

simu_setup = expset.simuSKT(theta_d)
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
ww = np.ones(KComp)/KComp
nu = 20 #expset.nu

#Prec_init = np.zeros((KComp, theta_d, theta_d))   
phimix =  ggm.GaussGammaMix(KComp, theta_d)
for iterI in range(KComp):
    #Prec_init[iterI] =  np.linalg.inv(covs_init[iterI])
    Prec_L = np.linalg.cholesky(Prec_init[iterI])
    phimix.set_parameter(iterI, mu_init[iterI],\
           Prec_L,  ww[iterI], nu/2, nu/2)
phimix.setlogq(logq_fun)
phimix.setlogq_grad(logq_grad)


#%%
# lr_prec = 1e-3
# lr = 1e-3
# maxIter = 2000
# nSamples = 100
kl_trace = gamma_cvi_general(phimix, target_val_grad_batch, 
                                nSamples, maxIter,
                                mu_init, Prec_init, nu, ww, 
                                lr, lr_prec, lr_w=lr_w, printGap=500)



#%%
savefile = expset.basepath + "gammaVI_" + str(theta_d) + ".npz"
np.savez(savefile, kl_trace = kl_trace, mu = mu_init, 
         Prec = Prec_init,  lr_w = lr_w,  ww = ww, nu = nu)

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
    print("-----------------")

# %%
