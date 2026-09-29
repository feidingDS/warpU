#%%
# Metroplis-Hastings with independent proposal
import time
import numpy as np
import argparse
import sys

#%%
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
    samplerType = 2 # 0 is HMC, 1 is WarpU, 2 is MH
    # Phimix: 1 is Gaussian (K=2), 2 is Gaussian (K=6), 3 is nueral Gaussian
    phimixType = args.phimixType 
    totalRep = args.totalRep
    startI = args.startI
else:
    theta_d = 30
    setID = 3
    samplerType = 2  # only MH
    phimixType = 3
    totalRep = 10
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
    phimix.setlogq(logq_fun)
    if phimixType == 1:
        methodName = "HMPIGaussK2"
    elif phimixType == 2:
        methodName = "HMPIGaussK6"


#%%
def run_oneRep(repID):

    numberIter = mcmcparams.numberIter
     
    np.random.seed(repID)  
    be = time.time()

    propseq = phimix.drawsample(numberIter+10)
    logq_prop_seq = logq_fun(propseq)
    logphimix_prop_seq = phimix.logphimix_batch(propseq)
    
    mcmcSamples = np.zeros((numberIter, theta_d))
    mcmcSamples[0,:] = pi0()
    logq_cur = logq_fun(mcmcSamples[0,:])
    logphimix_cur = logphimix_prop_seq[0]
    for iter in range(1, numberIter):
        prop_theta = propseq[iter]
        prop_q = logq_prop_seq[iter]
        logq_ratio = prop_q - logq_cur + logphimix_cur - logphimix_prop_seq[iter]
        if np.log(np.random.rand()) < logq_ratio:
            mcmcSamples[iter,:] = prop_theta
            logq_cur = prop_q
            logphimix_cur = logphimix_prop_seq[iter]
        else:
            mcmcSamples[iter,:] = mcmcSamples[iter-1,:]
    #return mcmcSamples
    timeCost = time.time() - be
    mcmcparams.save_results(mcmcSamples, timeCost, methodName, theta_d, repID)
    #print("RepID: ", repID, "Time: ", time.time()-be)

#%%
for j in range(startI, totalRep):
    run_oneRep(j)

