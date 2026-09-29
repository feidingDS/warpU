import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
cimport warpu.coupledMH
import warpu.utils
import time

np.import_array()
ctypedef unsigned int uint


# Gradient related MCMC method. 
cdef class coupledGradMCMC(warpu.coupledMH.coupledMH):

    def __init__(self, unsigned int d, double sigma, \
            bint useHMC_ = False, double hmc_epsilon_=1e-3, unsigned int hmc_iterN_=100,\
            double hmc_sigma_=1, double gamma_=1):
        super().__init__(d,sigma)
        self.useHMC = useHMC_
        self.hmc_epsilon = hmc_epsilon_
        self.hmc_iterN = hmc_iterN_
        self.hmc_sigma = hmc_sigma_
        self.gamma = gamma_ # the probability of choosing the HMC kernel or the random walk kenrel

    cpdef void setlogqgrad(self, object logqgrad_):
        self.logq_grad = logqgrad_
    
    cdef void leapfrog(self, np.ndarray v,  np.ndarray x):
        cdef double stepsize
        cdef unsigned int step, dimJ
        cdef np.ndarray grad
        stepsize = self.hmc_epsilon
        grad = self.logq_grad(x) 
        v +=  (stepsize*0.5) * grad

        for step in range(1, self.hmc_iterN+1):
            x +=  stepsize * v
            grad = self.logq_grad(x) 
            if (step != self.hmc_iterN):
                    v +=  stepsize * grad
    
        v += (stepsize*0.5) * grad


    cpdef void coupledTrans(self):
        if self.useHMC:
            if rd.uniform(0,1) < self.gamma:
                self.coupledMH()
            else:
                self.coupledHMC()
        else:
            self.coupledMALA()



    cdef void coupledHMC(self):
        """
        Github reference: https://github.com/pierrejacob/debiasedhmc/blob/master/R/hmc_kernel.R
        Reference: https://arxiv.org/pdf/1709.00404.pdf
        """
        cdef np.ndarray current_v, proposed_v1, theta1Star, proposed_v2, theta2Star
        cdef double logq1Star, Uvar1, diff1, currentE, proposedE1
        cdef double logq2Star, Uvar2, diff2, proposedE2

        current_v = rd.normal(0, self.hmc_sigma, self.d)
        currentE = np.sum(current_v**2) *0.5/self.hmc_sigma**2

        proposed_v1 = np.copy(current_v)
        theta1Star = np.copy(self.theta1)
        self.leapfrog(proposed_v1, theta1Star)
        logq1Star = self.logq_fun(theta1Star)
        proposedE1 = np.sum(proposed_v1**2) *0.5/self.hmc_sigma**2

        proposed_v2 = np.copy(current_v)
        theta2Star = np.copy(self.theta2) 
        self.leapfrog(proposed_v2, theta2Star)
        logq2Star = self.logq_fun(theta2Star)
        proposedE2 = np.sum(proposed_v2**2) *0.5/self.hmc_sigma**2


        Uvar = rd.uniform(0,1)
        diff1 = logq1Star - self.logq1 + currentE - proposedE1

        diff1 = np.min([diff1, 0])
        diff1 = np.max([diff1, -1e20])
        if Uvar < np.exp(diff1):
            self.theta1 = np.copy(theta1Star)
            self.logq1 = logq1Star
            self.acceptCount1[self.mcmcIter-1] = 1.0

        diff2 = logq2Star - self.logq2 + currentE - proposedE2

        diff2 = np.min([diff2, 0])
        diff2 = np.max([diff2, -1e20])
        if Uvar < np.exp(diff2):
            self.theta2 = np.copy(theta2Star)
            self.logq2 = logq2Star
            self.acceptCount2[self.mcmcIter-1] = 1.0

    cpdef void singleTrans(self):
        if self.useHMC:
            if rd.uniform(0,1) < self.gamma:
                self.singleMH()
            else:
                self.singleHMC()
        else:
            self.singleMALA()

    cpdef void singleHMC(self):
        cdef np.ndarray current_v, proposed_v, theta1Star
        cdef double logq1Star, Uvar, diff1, currentE, proposedE
        current_v = rd.normal(0, self.hmc_sigma, self.d)

        proposed_v = np.copy(current_v)
        theta1Star = np.copy(self.theta1) 
        self.leapfrog(proposed_v, theta1Star)
        logq1Star = self.logq_fun(theta1Star) 
        # proposed and current energy
        currentE = np.sum(current_v**2) *0.5/self.hmc_sigma**2
        proposedE = np.sum(proposed_v**2) *0.5/self.hmc_sigma**2
        Uvar = rd.uniform(0,1)
        diff1 = logq1Star - proposedE - (self.logq1 - currentE)
        diff1 = np.min([diff1, 0])
        diff1 = np.max([diff1, -1e20])
        if Uvar < np.exp(diff1):
            self.theta1 = np.copy(theta1Star)
            self.logq1 = logq1Star
            self.acceptCount1[self.mcmcIter-1] = 1.0
            #self.theta2 = np.copy(theta1Star)
            #self.logq2 = logq1Star


    # Metropolis Adjusted Langevin Algorithm
    #  See Algorithm 11 of Biswas, Jacob and Vanetti.
    cdef void coupledMALA(self):
        cdef np.ndarray grad1, grad2, mu1, mu2
        cdef np.ndarray theta1Star, theta2Star, mu1Star, mu2Star
        cdef bint flag1, flag2, flag3
        cdef double logq1Star, logq2Star

        grad1 = self.logq_grad(self.theta1 )
        grad2 = self.logq_grad(self.theta2 )
        
        mu1 = self.theta1 + (self.sigma**2/2) * grad1
        mu2 = self.theta2 + (self.sigma**2/2) * grad2

        theta1Star, theta2Star, flag1 = \
            warpu.utils.reflectionGaussDraw(mu1, mu2, self.sigma)
        #maximxalGaussDraw
        #theta1Star, theta2Star, flag1 = \
        #    core.utils.maximxalGaussDraw(mu1, mu2, self.sigma)
        #theta1Star = mu1 + rd.normal(0, self.sigma, self.d)
        #theta2Star = mu2 + rd.normal(0, self.sigma, self.d)
        #flag1 = False


        grad1 = self.logq_grad(theta1Star)
        grad2 = self.logq_grad(theta2Star )
        mu1Star = theta1Star + (self.sigma**2/2) * grad1
        mu2Star = theta2Star + (self.sigma**2/2) * grad2


        logq1Star = self.logq_fun(theta1Star)
        logq2Star = self.logq_fun(theta2Star)
        Uvar = rd.uniform(0,1)
        flag2 = False
        flag3 = False
        diff1 = logq1Star - self.logq1
        diff1 +=  np.sum( gauss.logpdf( (self.theta1 - mu1Star)/self.sigma ))
        diff1 -=  np.sum( gauss.logpdf( (theta1Star - mu1)/self.sigma ))
        diff1 = np.min([diff1, 0])
        diff1 = np.max([diff1, -1e20])

        diff2 = logq2Star - self.logq2
        diff2 +=  np.sum( gauss.logpdf( (self.theta2 - mu2Star)/self.sigma ))
        diff2 -=  np.sum(  gauss.logpdf( (theta2Star - mu2)/self.sigma ) )
        diff2 = np.min([diff2, 0])
        diff2 = np.max([diff2, -1e20])
        if Uvar < np.exp(diff1):
            self.theta1 = np.copy(theta1Star)
            self.logq1 = logq1Star
            flag2 = True
            self.acceptCount1[self.mcmcIter-1] = 1.0
        if Uvar < np.exp(diff2):
            self.theta2 = np.copy(theta2Star)
            self.logq2 = logq2Star
            flag3 = True
            self.acceptCount2[self.mcmcIter-1] = 1.0
        self.flag = flag1 and flag2 and flag3

    cdef void singleMALA(self):
        cdef np.ndarray theta1Star, mu1Star, grad1, mu1
        cdef double logq1Star

        grad1 = self.logq_grad(self.theta1) 
        mu1 = self.theta1 + (self.sigma**2/2) * grad1 
        theta1Star = mu1 + rd.normal(0, self.sigma, self.d)

        grad1 = self.logq_grad(theta1Star)
        mu1Star = theta1Star + (self.sigma**2/2) * grad1

        logq1Star = self.logq_fun(theta1Star) 
        Uvar = rd.uniform(0,1)
        diff1 = logq1Star - self.logq1
        #print(diff1)
        diff1 +=  np.sum( gauss.logpdf( (self.theta1 - mu1Star)/self.sigma ))
        diff1 -=  np.sum( gauss.logpdf( (theta1Star - mu1)/self.sigma ))
        #print(diff1)
        #print()
        diff1 = np.min([diff1, 0])
        diff1 = np.max([diff1, -1e20])
        if Uvar < np.exp(diff1):
            self.theta1 = np.copy(theta1Star)
            self.logq1 = logq1Star
            self.acceptCount1[self.mcmcIter-1] = 1.0
            #self.theta2 = np.copy(theta1Star)
            #self.logq2 = logq1Star
 
