#%%
#import matplotlib.pyplot as plt
from warpu.WarpUSampling import WarpUSampling
from warpu.coupledGradMCMC import coupledGradMCMC
from warpu.tempering import ptSampler
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from threadpoolctl import threadpool_limits
import argparse
import sys

import os
import multiprocessing as mp

# set the max threads for each process
MAX_Threads = "2"
os.environ.setdefault("OMP_NUM_THREADS", MAX_Threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", MAX_Threads)
os.environ.setdefault("MKL_NUM_THREADS", MAX_Threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", MAX_Threads)


#%%
if len(sys.argv) > 2:
    parser = argparse.ArgumentParser()
    parser.add_argument('--setID', type=int, default=1)
    parser.add_argument('--samplerType', type=int, default=0)
    parser.add_argument('--theta_d', type=int, default=100)
    parser.add_argument('--totalRep', type=int, default=10)
    parser.add_argument('--vanilla', type=int, default=1)
    parser.add_argument('--startI', type=int, default=0)
    args = parser.parse_args()
    theta_d = args.theta_d
    setID = args.setID
    vanilla = True if args.vanilla == 1 else False
    samplerType = args.samplerType # 0 is HMC, 1 is WarpU
    phimixType = 1 # only Gaussian mixture is considered 
    totalRep = args.totalRep
    startI = args.startI
else:
    theta_d = 50
    setID = 1
    vanilla = True
    samplerType = 1 # 0 is HMC, 1 is WarpU
    phimixType = 1
    totalRep = 2
    startI = 0

params = {"pt": True, "vanilla": vanilla, "phimixType": phimixType, "samplerType": samplerType}
#%%
def run_oneRep(repID):
    with threadpool_limits(limits=int(MAX_Threads)): 
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
        numberIter = mcmcparams.numberIter
        NChains = mcmcparams.NChains

        numberIter = numberIter // 3 * 2 + 10

        
        targetD = simu_setup.get_targetD()
        logq_fun = targetD.logphimix_batch
        logq_grad = targetD.logphimix_val_grad_batch

        phimix = mcmcparams.get_phimix(phimixType, expset.basepath, theta_d)
        phimix.setlogq(logq_fun)
        #phimix.setlogq_grad(logq_grad)
        #phimix.set_backward_rep(300)
        np.random.seed(repID)  

        cHMCObj = ptSampler(theta_d, prop_sigma,\
                    True, hmc_epsilon_, hmc_iterN_, hmc_sigma_, hmc_gamma_)
        #tSeq = np.linspace(0, 1, NChains+1)[1:]
        tSeq = mcmcparams.get_ptSeq(methodName, theta_d)
        if not vanilla:        
            cHMCObj.set_targetT0(phimix)

        cHMCObj.set_tempering(tSeq)
        
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
    #print("method: %s, theta_d: %d, RepID: %d, timeCost: %0.1f" % (methodName, theta_d, repID, timeCost))
    return repID, timeCost




# %%
def main():
    max_workers = 12
    if theta_d >= 500:
        max_workers = 6
    num_workers = min(max_workers, os.cpu_count())
    print(f"The current setting is {setID}, {theta_d}, phimix = {phimixType}")
    ctx = mp.get_context("spawn")
    with threadpool_limits(limits=int(MAX_Threads)):
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            for rep_id, time_cost in ex.map(run_oneRep, range(startI, totalRep)):
                print(f"rep {rep_id} done, timeCost={time_cost:.1f}")

if __name__ == "__main__":
    mp.freeze_support()                
    mp.set_start_method("spawn", force=True) 
    main()
    print("Finished!")
    print()