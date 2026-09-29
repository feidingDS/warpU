
#%%
import numpy as np
from scipy.stats import norm
import os
from estimator.bridge import *
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
import argparse
import sys
#%%

parser = argparse.ArgumentParser()
parser.add_argument('--setID', type=int, default=1)
parser.add_argument('--theta_d', type=int, default=100)
parser.add_argument('--phimix_type', type=int, default=1)
parser.add_argument('--sample_type', type=int, default=1)
parser.add_argument('--parallel', type=int, default=0)
args, unknown = parser.parse_known_args()

if len(sys.argv) > 2:
    theta_d = args.theta_d
    setID = args.setID
    phimix_type = args.phimix_type
    sample_type = args.sample_type
    parallel = args.parallel
else:
    setID = 3
    phimix_type = 1
    sample_type = 7
    totalRep = 10
    theta_d = 30
    parallel = 1



if setID==2:
    import params.eg02setup as expset
    import params.eg02mcmc as mcmcset
    theta_d = 30
    simu_setup = expset.simuSKT(theta_d)
    #simu_setup.init_r()
elif setID==3:
    import params.eg03setup as expset
    import params.eg03mcmc as mcmcset
    theta_d = 30
    simu_setup = expset.simuSshape(theta_d)

base_path = expset.basepath
targetD = simu_setup.get_targetD() 
logq_fun = targetD.logphimix_batch
totalRep = mcmcset.get_totalRep()
print("The current setting is %d, %d" % (setID, theta_d))
print("The sample type and phimix type is %d, %d" % (sample_type, phimix_type))

#%%
def one_rep(repID):
    # draw iid samples
    if repID % 10 == 0:
        print(repID)
    thetaMCMC = mcmcset.get_sample(sample_type, theta_d, repID)
    # get kernel
    phimix = mcmcset.get_phimix(phimix_type, base_path, theta_d)
    estimatorSel = mcmcset.get_estimatorSel(phimix_type)
    phimix.set_backward_rep(100)
    phimix.setlogq(logq_fun)
    # compute cHat by bridge estimator
    seq = mcmcset.get_cutseq(thetaMCMC.shape[0])
    cHats = one_kernel_seq(thetaMCMC, phimix, logq_fun, seq, estimatorSel)
    return cHats


#%%
num_workers = min(8, os.cpu_count())
if parallel:
    threadnum = os.cpu_count() // num_workers
    threadpool_limits(limits=threadnum)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(one_rep, range(totalRep)))
else:
    results = list()
    for i in range(totalRep):
        results.append(one_rep(i))


savepath =  mcmcset.savepath + "bridge"
if not os.path.exists(savepath):
    os.makedirs(savepath)
savepath = savepath + "/sampler_" + str(sample_type) +\
      "_phimix_" + str(phimix_type) + "_theta_" + str(theta_d) + ".npz"
np.savez(savepath, results=results)

