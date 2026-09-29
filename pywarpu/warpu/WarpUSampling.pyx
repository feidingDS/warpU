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


cdef class WarpUSampling(warpu.coupledGradMCMC.coupledGradMCMC):
    cdef np.ndarray qTildeValsHist
    cdef np.ndarray thetaStarHist, factor_StoWU, psiHist, transCount
    cdef object phimix
    cdef unsigned int mixK

    def __init__(self,unsigned int d, double sigma,\
            bint useHMC_ = False, double hmc_epsilon_=1e-3, unsigned int hmc_iterN_=100,\
            double hmc_sigma_=1, double gamma_=1):
        super().__init__(d, sigma, useHMC_, hmc_epsilon_, hmc_iterN_, hmc_sigma_,
                        gamma_)
        self.mixK = -1
        self.phimix = None

    def set_phimix(self, phimix_):
        self.phimix = phimix_
        self.mixK = phimix_.get_K()

    # the next three functions are for the evaluation of
    # (stochastic) Warp-U Bridge Estimation
    def get_qvalsHist(self):
        return self.qTildeValsHist

    def get_thetaStarHist(self):
        return self.thetaStarHist
    
    def get_factor_StoWU(self):
        return self.factor_StoWU

    def get_psiHist(self):
        return self.psiHist
    
    def get_transCount(self):
        return self.transCount


    cpdef void singleTrans(self):
        cdef np.ndarray thetaStar, probv, thetanew1, qVals, phiMixVals 
        cdef double qtilde
        cdef uint iterK, psi0, psi1

        warpu.coupledGradMCMC.coupledGradMCMC.singleTrans(self)

        # forward transformation and record result
        probv, thetanew1 = self.phimix.forward_info(self.theta1)
        psi0 = rd.choice(range(self.mixK), 1, False, probv)[0]
        thetaStar = np.copy(thetanew1[psi0,:])
        self.psiHist[self.mcmcIter-1] = psi0
        #thetaStar = self.phimix.draw_forward(self.theta1)
        self.thetaStarHist[self.mcmcIter-1, :] = thetaStar

        #qtilde = self.phimix.warped_logq(thetaStar)

        # Compute log probability: true
        probv_log, thetanew1, qVals  = self.phimix.backward_info(thetaStar)

        probv_log_max = np.max(probv_log)
        probv = probv_log - probv_log_max
        probv = np.exp(probv)
        probv = probv / np.sum(probv)

        qtilde = np.sum(np.exp(probv_log - probv_log_max)) + probv_log_max
        self.qTildeValsHist[self.mcmcIter-1] = qtilde

        psi1 = rd.choice(range(self.mixK), 1, False, probv)[0]
        self.theta1 = np.copy(thetanew1[psi1,:])
        self.logq1 = qVals[psi1]

        if psi1 != psi0:
            self.transCount[self.mcmcIter-1] = 1

        # record the value of Phi_mix at S_psi * thetaStar + mu_psi
        #self.factor_StoWU[self.mcmcIter-1] = qVals[psi0] - phiMixVals[psi0]


    cpdef void run_classical(self, unsigned int m):
        self.qTildeValsHist = np.zeros(m) # q_tilde(theta^*)
        self.factor_StoWU = np.zeros(m) # storage for stochastic sampling
        self.psiHist = np.zeros(m)
        self.thetaStarHist = np.zeros( (m, self.d) )
        warpu.coupledMH.coupledMH.run_classical(self, m)

    cpdef void classical_prepare(self, unsigned int m):
        warpu.coupledMH.coupledMH.classical_prepare(self, m)
        self.qTildeValsHist = np.zeros(m) # q_tilde(theta^*)
        #self.factor_StoWU = np.zeros(m) # storage for stochastic sampling
        self.psiHist = np.zeros(m)
        self.transCount = np.zeros(m)
        self.thetaStarHist = np.zeros( (m, self.d) )