import numpy.random as rd
import numpy as np
from warpu.gaussmixfull import GaussMixtureFull
from neuralode.deepphimix import deepphimixClass
from neuralode.deepphimixv2 import deepphimixClass_v2
import params.eg03setup as expset
import os

# HMC parameters
hmc_epsilon_ = 0.03  
hmc_iterN_ = 25
hmc_sigma_ = 1
hmc_gamma_ = 0
prop_sigma = 0.5

# Neural ODE parameters
width_neural = 128
hidden_dim_neural = 128
t0_neural = 0
t1_neural = 1
ode_stepsize_neural = 0.1
lr_neural = 5e-4
niters_neural = 501
nSample_neural = 512*4

##########
numberIter = 10500
NChains = 100
totalRep = 1
##########


savepath = "./results/eg03mcmc/"

def get_init_sampler(theta_d):
    def init_sampler():
        return rd.normal(0, 1, theta_d)
    return init_sampler

def get_totalRep():
    return totalRep*NChains

def get_phimix(phimixType, basepath, theta_d):
    phimix = None

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
        savefile = basepath + "gaussVIV2_K6_" + str(theta_d) + ".npz"
        mixParams = np.load(savefile)
        mu = mixParams["mu"]
        ww = mixParams["ww"]
        Prec = mixParams["Prec"]
        KCompMix = mu.shape[0]
        phimix =  GaussMixtureFull(KCompMix, theta_d) 
        for iterI in range(KCompMix):
            Prec_L = np.linalg.cholesky(Prec[iterI])
            phimix.set_parameter(iterI, mu[iterI], Prec_L,  ww[iterI])
    elif phimixType == 3:
        phimixbase = get_phimix(1, basepath, theta_d)
        stName  = "sshape"
        phimix = deepphimixClass(theta_d, phimixbase)
        phimix.set_ode_param(t0_neural, t1_neural, ode_stepsize_neural)
        phimix.set_network(width_neural, hidden_dim_neural, basepath, stName)
    elif phimixType == 4:
        phimixbase = get_phimix(1, basepath, theta_d)
        stName  = "sshape_V2K2"
        phimix = deepphimixClass_v2(theta_d, phimixbase)
        phimix.set_ode_param(t0_neural, t1_neural, ode_stepsize_neural)
        phimix.set_network(width_neural, hidden_dim_neural, basepath, stName)
    return phimix

            



def save_results(thetas, rt,  methodName, theta_d, repID):
    dirpath = savepath + methodName + "/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    savefile = dirpath +   "D" + str(theta_d) 
    savefile += "_R" + str(repID) + ".npz"
    np.savez(savefile, thetas=thetas, rt = rt)
    
def get_methodName(params):
    sampler_type = params["samplerType"]
    phimixType = params["phimixType"]
    methodName = None
    if sampler_type == "WarpU":
        if phimixType == 1:
            methodName = "gaussianWarpUK2"
        elif phimixType == 2:
            methodName = "gaussianWarpUK6"
        elif phimixType == 3:
            methodName = "neuralWarpU"
        elif phimixType == 4:
            methodName = "neuralWarpUV2"
    else:
        if phimixType == 1:
            methodName = "HMPIGaussK2"
        elif phimixType == 2:
            methodName = "HMPIGaussK6"
        elif phimixType == 3:
            methodName = "HMPINeural"
    return methodName



def get_sample(sample_type, theta_d, repID):
    fileRepID = repID
    if sample_type == 0:
        return expset.get_iid_sample(theta_d, numberIter-500)

    isbatchSampler = False
    if sample_type == 1:
        params = {"phimixType": 1, "samplerType": "WarpU"}
        isbatchSampler = True
    elif sample_type == 2:
        params = {"phimixType": 2, "samplerType": "WarpU"}
        isbatchSampler = True
    elif sample_type == 3:
        params = {"phimixType": 3, "samplerType": "WarpU"}
        isbatchSampler = True
    elif sample_type == 4:
        params = {"phimixType": 1, "samplerType": "MH"}
    elif sample_type == 5:
        params = {"phimixType": 2, "samplerType": "MH"}
    elif sample_type == 6:
        params = {"phimixType": 3, "samplerType": "MH"}
    elif sample_type == 7:
        # for. prob. based on phimix(gauss), but for. map based on neural network
        params = {"phimixType": 4, "samplerType": "WarpU"}
        isbatchSampler = True
    if isbatchSampler:
        fileRepID = 0
    
    methodName = get_methodName(params)
    dirpath = savepath + methodName + "/"
    savefile = dirpath +   "D30_R" + str(fileRepID) + ".npz" 
    file = np.load(savefile)
    thetas = file["thetas"]
    if isbatchSampler:
        return thetas[500:,repID,:]
    else:
        return thetas[500:,:]

def get_cutseq(nsample):
    startI = nsample//100
    return np.linspace(startI, nsample, 30).astype(int)

    
def get_estimatorSel(phimixType):

    estimatorSel = np.array([1, 1, 0])

    return  estimatorSel

            
