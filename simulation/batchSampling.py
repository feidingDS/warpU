#%%
#import matplotlib.pyplot as plt
from warpu.batchchain import batchchain
import time
import numpy as np
import argparse
import sys



#%%
isNueralSampler = False

if len(sys.argv) > 2:
    parser = argparse.ArgumentParser()
    parser.add_argument('--setID', type=int, default=1)
    parser.add_argument('--theta_d', type=int, default=100)
    parser.add_argument('--totalRep', type=int, default=10)
    parser.add_argument('--startI', type=int, default=0)
    parser.add_argument('--phimixType', type=int, default=1)
    args = parser.parse_args()
    theta_d = args.theta_d
    setID = args.setID
    samplerType = 1 # only WarpU
    # Phimix: 1 is Gaussian (K=2), 2 is Gaussian (K=6), 3 is nueral Gaussian
    phimixType = args.phimixType 
    totalRep = args.totalRep
    startI = args.startI
else:
    theta_d = 30
    setID = 3
    samplerType = 1 # 0 is HMC, 1 is WarpU
    phimixType = 4
    totalRep = 2
    startI = 0

#%%
if setID == 3:
    import params.eg03setup as expset
    KComp = expset.KComp
    import params.eg03mcmc as mcmcparams
    pi0 = mcmcparams.get_init_sampler(theta_d)
    simu_setup = expset.simuSshape(theta_d)
    targetD = simu_setup.get_targetD()
    logq_fun = targetD.logphimix_batch
    logq_grad = targetD.logphimix_val_grad_batch

    phimix = mcmcparams.get_phimix(phimixType, expset.basepath, theta_d)

    if phimixType >= 3:
        isNueralSampler = True
    phimix.setlogq(logq_fun)
    params = {"phimixType": phimixType, "samplerType": "WarpU"}
    methodName = mcmcparams.get_methodName(params)



#%%
def run_oneRep(repID):

    hmc_epsilon_ = mcmcparams.hmc_epsilon_
    hmc_iterN_ = mcmcparams.hmc_iterN_
    hmc_sigma_ = mcmcparams.hmc_sigma_
    hmc_gamma_ = mcmcparams.hmc_gamma_
    prop_sigma = mcmcparams.prop_sigma
    numberIter = mcmcparams.numberIter
    NChains = mcmcparams.NChains
    
    np.random.seed(repID)  

    cHMCObj = batchchain(theta_d, prop_sigma,\
                True, hmc_epsilon_, hmc_iterN_, hmc_sigma_, hmc_gamma_)
    
    cHMCObj.set_NChains(NChains)
    if isNueralSampler:
        cHMCObj.set_warpUGap(3)

    
    if samplerType == 1:
        cHMCObj.set_phimix(phimix) # warpU
        
    cHMCObj.setpi0(pi0)
    cHMCObj.setlogq(logq_fun)
    cHMCObj.set_logq_val_grad(logq_grad)

    be = time.time()
    cHMCObj.run_classical(numberIter)
    ed = time.time()
    timeCost = (ed - be)
    mcmcSamples = cHMCObj.get_theta1All()
    #return mcmcSamples, cHMCObj
    mcmcparams.save_results(mcmcSamples, timeCost, methodName, theta_d, repID)
    print("method: %s, theta_d: %d, RepID: %d, timeCost: %0.2f" % (methodName, theta_d, repID, timeCost))

#%%
for j in range(startI, totalRep):
    run_oneRep(j)


