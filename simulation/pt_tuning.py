#%%
import matplotlib.pyplot as plt
from warpu.tempering import ptSampler
import numpy as np
import argparse
import sys



#%%
if len(sys.argv) > 2:
    parser = argparse.ArgumentParser()
    parser.add_argument('--setID', type=int, default=1)
    parser.add_argument('--samplerType', type=int, default=0)
    parser.add_argument('--theta_d', type=int, default=100)
    parser.add_argument('--vanilla', type=int, default=1)
    args = parser.parse_args()
    theta_d = args.theta_d
    setID = args.setID
    vanilla = True if args.vanilla == 1 else False
    samplerType = args.samplerType # 0 is HMC, 1 is WarpU
    phimixType = 1 # only Gaussian mixture is considered 
else:
    theta_d = 1000
    setID = 1
    vanilla = False
    samplerType = 0 # 0 is HMC, 1 is WarpU
    phimixType = 1

params = {"pt": True, "vanilla": vanilla, "phimixType": phimixType, "samplerType": samplerType}
#%%

if setID == 1:
    import params.eg01setup as expset
    import params.eg01mcmc as mcmcparams
    simu_setup = expset.simuSKT(theta_d)
elif setID == 2:
    import params.eg02setup as expset
    import params.eg02mcmc as mcmcparams
    simu_setup = expset.simuSKT(theta_d)

pi0 = mcmcparams.get_init_sampler(theta_d)
methodName = mcmcparams.get_methodName(params)
hmc_epsilon_ = mcmcparams.hmc_epsilon_
hmc_iterN_ = mcmcparams.hmc_iterN_
hmc_sigma_ = mcmcparams.hmc_sigma_
hmc_gamma_ = mcmcparams.hmc_gamma_
prop_sigma = mcmcparams.prop_sigma



targetD = simu_setup.get_targetD()
logq_fun = targetD.logphimix_batch
logq_grad = targetD.logphimix_val_grad_batch

phimix = mcmcparams.get_phimix(phimixType, expset.basepath, theta_d)
phimix.setlogq(logq_fun)


def find_next(tSeq, rho_init, numberIter, lr, tuning_iterMin=100, 
            tuning_iterGap=20, tuning_rhoGap=0.01): 
    np.random.seed(123)  

    cHMCObj = ptSampler(theta_d, prop_sigma,\
                True, hmc_epsilon_, hmc_iterN_, hmc_sigma_, hmc_gamma_)
    if not vanilla:        
        cHMCObj.set_targetT0(phimix)

    cHMCObj.set_tempering(tSeq)

    if samplerType == 1:
        cHMCObj.set_phimix(phimix) # warpU
    
    cHMCObj.setpi0(pi0)
    cHMCObj.setlogq(logq_fun)
    cHMCObj.set_logq_val_grad(logq_grad)

    cHMCObj.set_tuning(rho_init, lr, tuning_iterMin, tuning_iterGap)
    cHMCObj.set_tuning_rhoGap(tuning_rhoGap)
    cHMCObj.run_classical(numberIter)
    rhoS, accS = cHMCObj.get_tuningseq()
    return rhoS, accS



#%%
lr = 0.05
niter = 2000
rho_gap = 0.01
iterGap = 5
if vanilla:
    tseq = np.array([0.05])
    rho_gap = 0.05
    iterGap = 10
else:
    tseq = np.array([])
cMax = 0
while cMax < 0.95 and len(tseq) < 20:
    next_t = (1+cMax)/2
    rho_init = np.log(1/next_t - 1)
    rhoS, accS = find_next(tseq, rho_init, niter, lr, tuning_iterGap= iterGap, tuning_rhoGap=rho_gap)
    tSeqMCMC = 1/(1+np.exp(rhoS))
    cMax = tSeqMCMC[-1]
    tseq = np.append(tseq, cMax)
    print("tseq = ", tseq)
    # plt.plot(tSeq)
    # plt.plot(accS)
    # plt.show()

#%%
tseq[-1] = 1
print("tseq = ", tseq)
mcmcparams.save_ptSeq(tseq, methodName, theta_d)

# %%
