#%%
#import matplotlib.pyplot as plt
from warpu.batchchain import batchchain
import time
import numpy as np
import numpy.random as rd
from warpu.gaussmixfull import GaussMixtureFull
from scipy.stats import norm as gauss
import argparse
import sys
import os
#%%
if len(sys.argv) > 2:
    parser = argparse.ArgumentParser()
    parser.add_argument('--setID', type=int, default=1)
    parser.add_argument('--theta_d', type=int, default=100)
    parser.add_argument('--totalRep', type=int, default=10)
    parser.add_argument('--phimix_type', type=int, default=1)
    parser.add_argument('--samplerType', type=int, default=1)
    args = parser.parse_args()
    theta_d = args.theta_d
    setID = args.setID
    samplerType = args.samplerType # only WarpU
    # Phimix: 1 is Gaussian (K=2), 2 is Gaussian (K=6), 3 is nueral Gaussian
    phimix_type = args.phimix_type 
    totalRep = args.totalRep
else:
    theta_d = 30
    setID = 3
    samplerType = 1 # 0 is HMC, 1 is WarpU
    phimix_type = 3
    totalRep = 2


if setID == 3:
    import params.eg03setup as expset
    KComp = expset.KComp
    import params.eg03mcmc as mcmcset
    pi0 = mcmcset.get_init_sampler(theta_d)
    simu_setup = expset.simuSshape(theta_d) # comment
    targetD = simu_setup.get_targetD()  # comment
    # input n samples, output n vector, log q for the target distribution
    logq_fun = targetD.logphimix_batch
    logq_grad = targetD.logphimix_val_grad_batch

    phimix = mcmcset.get_phimix(phimix_type, expset.basepath, theta_d)
    if phimix is not None:
        phimix.setlogq(logq_fun)
    


#%%
def run_oneRep_sampling(repID):
    np.random.seed(repID)
    hmc_epsilon_ = mcmcset.hmc_epsilon_
    hmc_iterN_ = mcmcset.hmc_iterN_
    hmc_sigma_ = mcmcset.hmc_sigma_
    hmc_gamma_ = mcmcset.hmc_gamma_
    prop_sigma = mcmcset.prop_sigma

    cHMCObj = batchchain(theta_d, prop_sigma,\
                True, hmc_epsilon_, hmc_iterN_, hmc_sigma_, hmc_gamma_)
    
    cHMCObj.set_NChains(NChains)    
    if samplerType == 1:
        cHMCObj.set_phimix(phimix) # warpU
    
    cHMCObj.set_warpUGap(10)
        
    cHMCObj.setpi0(pi0)
    cHMCObj.setlogq(logq_fun)
    cHMCObj.set_logq_val_grad(logq_grad)

    be = time.time()
    cHMCObj.run_classical(numberIter)
    ed = time.time()
    timeCost = (ed - be)
    #print(timeCost)
    mcmcSamples = cHMCObj.get_theta1All()
    return mcmcSamples, cHMCObj

def run_oneRep_sampling_HMIndep(repID):
     
    np.random.seed(repID)

    mcmcSamples = np.zeros(( NChains, numberIter, theta_d))

    for kI in range(NChains):
        propseq = phimix.drawsample(numberIter+10)
        logq_prop_seq = logq_fun(propseq)
        logphimix_prop_seq = phimix.logphimix_batch(propseq)
        
        mcmcSamples[kI,0,:] = pi0()
        logq_cur = logq_fun(mcmcSamples[kI,0,:])
        logphimix_cur = logphimix_prop_seq[0]
        for iter in range(1, numberIter):
            prop_theta = propseq[iter]
            prop_q = logq_prop_seq[iter]
            logq_ratio = prop_q - logq_cur + logphimix_cur - logphimix_prop_seq[iter]
            if np.log(np.random.rand()) < logq_ratio:
                mcmcSamples[kI,iter,:] = prop_theta
                logq_cur = prop_q
                logphimix_cur = logphimix_prop_seq[iter]
            else:
                mcmcSamples[kI,iter,:] = mcmcSamples[kI,iter-1,:]
    return mcmcSamples.transpose(1,0,2)

#%%
def compute(mcmcSamples, startI, base_scale):
    sampleFactor = 1
    NIter = mcmcSamples.shape[0]
    NChains = mcmcSamples.shape[1]
    ww = 1.0/NChains
    nRecord = (NIter-startI) 
    logphiMat = np.zeros((nRecord, NChains*sampleFactor))
    logqMat = np.zeros((nRecord, NChains*sampleFactor))

    basesample = np.empty((nRecord, NChains*sampleFactor, theta_d))
    j = 0
    for iterI in range(startI, NIter):
        phimix_w = GaussMixtureFull(NChains, theta_d)
        for kI in range(NChains):
            phimix_w.set_parameter(kI, mcmcSamples[iterI, kI], np.eye(theta_d)*base_scale, 1.0/NChains)
        basesample[j] = phimix_w.drawsample(NChains*sampleFactor)
        logqMat[j] = logq_fun(basesample[j])
        logphiMat[j] = phimix_w.logphimix_batch(basesample[j])
        j += 1
    return logphiMat, logqMat

#%%
def one_rep(repID, base_scale):
    np.random.seed(repID)
    if samplerType == 1:
        mcmcSamples, _ = run_oneRep_sampling(repID)
    else:
        mcmcSamples = run_oneRep_sampling_HMIndep(repID)
    logphiMat, logqMat = compute(mcmcSamples, startI, base_scale)
    weights = np.exp(logqMat - logphiMat)
    gap = 20
    cutseq = list()
    rhatseq = list()
    for x in range(gap-1, numberIter - startI, gap):
        cut = x + 1
        cutseq.append(cut*NChains)
        rhat = np.mean(weights[:cut])
        rhatseq.append(rhat)
    return np.array(rhatseq), np.array(cutseq)

# %%
results = list()
# totalRep = 2
base_scale = 1.1
numberIter = 300 + 50
NChains = 100
startI = 50

#%%

for i in range(totalRep):
    rhatseq, cutseq = one_rep(i, base_scale)
    results.append(rhatseq)

resultMat = np.zeros((len(results[0]), totalRep))
for i in range(totalRep):
    resultMat[:,i] = results[i]

#%%
# err = np.mean(np.abs(resultMat-1), axis=1)
# print(err)

# %%
savepath =  mcmcset.savepath + "bridge"
if not os.path.exists(savepath):
    os.makedirs(savepath)
if samplerType == 1:
    savepath = savepath + "/LAIS_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
else:
    savepath = savepath + "/LAIS_INDEP_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
np.savez(savepath, resultMat=resultMat, cutseq=cutseq)
# %%
