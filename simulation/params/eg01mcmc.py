import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
import warpu.GaussSkewMix as gsm
import warpu.GaussGammaMix as ggm
from warpu.gaussmixfull import GaussMixtureFull
import os
# HMC parameters
NChains = 10
hmc_epsilon_ = 0.03  
hmc_iterN_ = 25
hmc_sigma_ = 1
hmc_gamma_ = 0
prop_sigma = 0.5

##########
numberIter = 6100
totalRep = 40
##########


savepath = "./results/eg01mcmc/"


def get_totalRep():
    return totalRep

def get_init_sampler(theta_d):
    def init_sampler():
        return rd.normal(0, 1, theta_d)
    return init_sampler


def get_phimix(phimixType, basepath, theta_d):
    phimix = None

    # if phimixType == 1:
    #     savefile = basepath + "gmm_" + str(theta_d) + ".npz"
    #     mixParams = np.load(savefile)
    #     mu = mixParams['means']
    #     covs = mixParams['covs']
    #     ww = mixParams['weights']
    #     KCompMix = mu.shape[0]
    #     phimix = GaussMixtureFull(KCompMix, theta_d)
    #     for iterI in range(KCompMix):
    #         Prec = np.linalg.inv(covs[iterI])
    #         PrecL = np.linalg.cholesky(Prec)
    #         phimix.set_parameter(iterI, mu[iterI], PrecL, ww[iterI])

    if phimixType == 1:
        savefile = basepath + "gaussVIV2_" + str(theta_d) + ".npz"
        mixParams = np.load(savefile)
        mu = mixParams["mu"]
        ww = mixParams["ww"]
        Prec = mixParams["Prec"]
        KCompMix = mu.shape[0]
        phimix =  GaussMixtureFull(KCompMix, theta_d) 
        for iterI in range(KCompMix):
            Prec_L = np.linalg.cholesky(Prec[iterI])
            phimix.set_parameter(iterI, mu[iterI], Prec_L,  ww[iterI])
            
    if phimixType == 2:
        savefile = basepath + "skewVI_" + str(theta_d) + ".npz"
        mixParams = np.load(savefile)
        mu = mixParams["mu"]
        alpha = mixParams["alpha"]
        ww = mixParams["ww"]
        Prec = mixParams["Prec"]
        KCompMix = mu.shape[0]
        phimix =  gsm.GaussSkewMix(KCompMix, theta_d) 
        for iterI in range(KCompMix):
            Prec_L = np.linalg.cholesky(Prec[iterI])
            phimix.set_parameter(iterI, mu[iterI],\
                                Prec_L,  ww[iterI],  alpha[iterI])
            
    if phimixType == 3:
        savefile = basepath + "gammaVI_" + str(theta_d) + ".npz"
        mixParams = np.load(savefile)
        mu = mixParams["mu"]
        nu = mixParams["nu"]
        ww = mixParams["ww"]
        Prec = mixParams["Prec"]
        KCompMix = mu.shape[0]
        phimix =  ggm.GaussGammaMix(KCompMix, theta_d)
        for iterI in range(KCompMix):
            Prec_L = np.linalg.cholesky(Prec[iterI])
            phimix.set_parameter(iterI, mu[iterI],\
                Prec_L,  ww[iterI],  nu/2, nu/2)
    return phimix


def save_results(thetas, rt,  methodName, theta_d, repID):
    dirpath = savepath + methodName + "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    savefile = dirpath +   "D" + str(theta_d) 
    savefile += "_R" + str(repID) + ".npz"
    firstmoment = np.mean(thetas, axis=1)
    secondmoment = np.mean(thetas**2, axis=1)
    savedim = np.arange(0, theta_d, 10)
    savedim = np.append(savedim, 9) #theta_d-1
    np.savez(savefile, thetas=thetas[:,savedim], rt = rt,
             fm = firstmoment, sm = secondmoment)
    
def save_ptSeq(tseq, methodName, theta_d):
    dirpath = savepath + methodName + "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    savefile = dirpath +   "ptSeq_D" + str(theta_d) + ".npz"
    np.savez(savefile, tseq = tseq)

def get_ptSeq(methodName, theta_d):
    dirpath = savepath + methodName + "/"
    savefile = dirpath +   "ptSeq_D" + str(theta_d) + ".npz"
    tmp = np.load(savefile)
    return tmp["tseq"]


def get_methodName(params):
    phimixType = params["phimixType"]
    if 'pt' in params:
        ptUsed = params["pt"]
        vanilla = params["vanilla"]
        samplerType = params["samplerType"]
    else:
        ptUsed = False
    if not ptUsed:
        if phimixType == 1:
            methodName = "gaussMix"
        elif phimixType == 2:
            methodName = "skewMix"
        elif phimixType == 3:
            methodName = "gammaMix"
    else:
        if vanilla:
            if samplerType == 0:
                methodName = "vanillaHMC"
            else:
                methodName = "vanillaWarpU"
        elif samplerType == 0:
            methodName = "ptHMC"
        else:
            methodName = "ptWarpU"

    return methodName

def get_sample(sample_type, theta_d, repID):
    if sample_type == 1:
        params = {"phimixType": 1}
    elif sample_type == 2:
        params = {"phimixType": 2}
    elif sample_type == 3:
        params = {"phimixType": 3}
    elif sample_type == 4:
        params = {"phimixType": 1, "pt": True, "vanilla": True, "samplerType": 0}
    elif sample_type == 5:
        params = {"phimixType": 1, "pt": True, "vanilla": True, "samplerType": 1}
    elif sample_type == 6:
        params = {"phimixType": 1, "pt": True, "vanilla": False, "samplerType": 0}
    elif sample_type == 7:
        params = {"phimixType": 1, "pt": True, "vanilla": False, "samplerType": 1}
    elif sample_type == 10:
        params = {"phimixType": 3, "samplerType": 2}

    mm = get_methodName(params)
    filePath = savepath + mm + "/D" + str(theta_d)
    filePath += "_R"  + str(repID) + ".npz"
    tmp = np.load(filePath)
    sample = tmp["thetas"][100:,[0,-1]]
    runtime = tmp["rt"]
    return sample, runtime