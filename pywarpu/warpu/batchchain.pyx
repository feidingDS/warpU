import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
cimport warpu.coupledGradMCMC
import warpu.utils
import time

np.import_array()
ctypedef unsigned int uint


cdef class batchchain(warpu.coupledGradMCMC.coupledGradMCMC):
    cdef object logq_val_grad_batch, phimix
    cdef np.ndarray  logqChains
    cdef uint NChains,  mixK,  theta1Iter, warpUGap


    def __init__(self, uint d, double sigma, \
            bint useHMC_ = False, double hmc_epsilon_=1e-3, uint hmc_iterN_=100,\
            double hmc_sigma_=1, double gamma_=1):
        super().__init__(d, sigma, useHMC_, hmc_epsilon_, hmc_iterN_, hmc_sigma_, gamma_)

        self.logq_val_grad_batch = None
        self.NChains = 1

        self.phimix = None # phimix for warp-u transformation
        self.mixK = -1
        self.warpUGap = 1
    
    def set_warpUGap(self, g_):
        if g_ < 0:
            return
        self.warpUGap = g_
    
    def set_phimix(self, phimix_):
        self.phimix = phimix_
        self.mixK = phimix_.get_K()

    def set_NChains(self, n_):
        self.NChains = n_

    
    
    def set_logq_val_grad(self, logq_val_grad_):
        self.logq_val_grad_batch = logq_val_grad_
    
    def get_theta1All(self):
        # return the recorded parameters of the target chain
        return self.theta1All[:self.theta1Iter]

    cpdef np.ndarray eval_logq_pt(self, np.ndarray theta1Star):
        return self.logq_fun(theta1Star).flatten()
    
    cdef np.ndarray eval_logq_grad_pt(self, np.ndarray theta1Star):
        cdef np.ndarray grad
        _, grad = self.logq_val_grad_batch(theta1Star)
        return grad

    cdef void leapfrog_pt(self, np.ndarray v,  np.ndarray x):
        cdef double stepsize
        cdef unsigned int step
        cdef np.ndarray grad

        stepsize = self.hmc_epsilon
        grad = self.eval_logq_grad_pt(x)
        v +=  (stepsize*0.5) * grad
        for step in range(1, self.hmc_iterN+1):
            x +=  stepsize * v
            grad = self.eval_logq_grad_pt(x)
            if (step != self.hmc_iterN):
                v +=  stepsize * grad
        v += (stepsize*0.5) * grad


    cpdef void coupledTrans(self):
        pass


    cpdef void singleTrans(self):
        # single step of transition with HMC
        cdef np.ndarray current_v, proposed_v, theta1Star
        cdef np.ndarray logq1Star, Uvar, diff1, currentE, proposedE

        ##
        # (warpU-)HMC update for the rest of the chains
        current_v = rd.normal(0, self.hmc_sigma, (self.NChains, self.d) )
        proposed_v = np.copy(current_v)
        theta1Star = np.copy(self.theta1) 
        self.leapfrog_pt(proposed_v, theta1Star)
        logq1Star = self.eval_logq_pt(theta1Star) 
        # proposed and current energy
        currentE = np.sum(current_v**2, axis = 1) /(2.*self.hmc_sigma**2)
        proposedE = np.sum(proposed_v**2, axis = 1) /(2.*self.hmc_sigma**2)

        Uvar = rd.uniform(0,1, self.NChains)
        diff1 = logq1Star - proposedE - (self.logqChains - currentE)
        diff1 = np.clip(diff1, -1e20, 0)
        acceptIdx = Uvar < np.exp(diff1)
        self.theta1 = np.where(acceptIdx[:,None], theta1Star, self.theta1)
        self.logqChains = np.where(acceptIdx, logq1Star, self.logqChains)

        if self.phimix is not None and self.mcmcIter % self.warpUGap == 0:
            self.warpu_update()
        
        # record the target chain
        self.record_theta1()

    cpdef void record_theta1(self):
        self.theta1Iter += 1
        self.theta1All[self.theta1Iter - 1] = np.copy(self.theta1)

    cpdef warpu_update(self):
        cdef np.ndarray probv, thetanew1, thetaStar, qValsStar, qt0Star
        cdef uint psi0
        # forward transformation and record result
        thetaStar = np.empty((self.NChains, self.d))
        probv, thetanew1 = self.phimix.forward_info_batch(self.theta1, False)
        for j in range(self.NChains):
            psi0 = rd.choice(range(self.mixK), 1, False, probv[j])[0]
            thetaStar[j] = np.copy(thetanew1[j, psi0, :])

        # backward transformation
        probv_log, thetanew1, qVals= self.phimix.backward_info_batch(thetaStar)


        probv_max = np.max(probv_log, axis = 1)
        probv = probv_log - probv_max[:,None]
        probv = np.exp(probv)
        probv = probv / np.sum(probv, axis = 1)[:,None]
        for j in range( self.NChains):
            psi1 = rd.choice(range(self.mixK), 1, False, probv[j])[0]
            self.theta1[j] = np.copy(thetanew1[j, psi1, :])
            self.logqChains[j] = qVals[j, psi1]


   

    cpdef void classical_prepare(self, unsigned int m):
        self.m = m

        self.theta1 = np.empty((self.NChains, self.d))
        for i in range(self.NChains):
            self.theta1[i] = self.pi0()
        self.logqChains = self.eval_logq_pt(self.theta1) 

        self.theta1All = np.zeros((m+2, self.NChains, self.d))
        self.theta1All[0] = np.copy(self.theta1) 
        self.mcmcIter = 1
        self.theta1Iter = 1
        

    cpdef void run_classical(self, unsigned int m):
        cdef unsigned int iter = 0
        self.classical_prepare(m)
        for iter in range(2, m+1):
            self.singleTrans()
            self.mcmcIter += 1
            #self.thetaAllLevels[iter-2, j] = np.copy(self.theta1)
        

