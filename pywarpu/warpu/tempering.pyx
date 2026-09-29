import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
cimport warpu.coupledGradMCMC
import warpu.utils
import time
import math

np.import_array()
ctypedef unsigned int uint


cdef class ptSampler(warpu.coupledGradMCMC.coupledGradMCMC):
    cdef object target_T0, logq_val_grad_batch, phimix
    cdef np.ndarray tseq, logqChains, level2Index
    cdef uint NChains, shift, mixK, sI, theta1Iter, theta0Iter
    cdef np.ndarray theta0All, logQ_at0, logPi0_at0, thetaAllLevels
    cdef double rho, lr, acc_ave, tuning_rhoGap # tuning parameter for swap
    cdef uint tuning, rhoIter, tuning_iterGap, tuning_iterMin # tuning parameter for swap 
    cdef np.ndarray rhoseq, accAveSeq # tuning parameter for swap

    def __init__(self, uint d, double sigma, \
            bint useHMC_ = False, double hmc_epsilon_=1e-3, uint hmc_iterN_=100,\
            double hmc_sigma_=1, double gamma_=1):
        super().__init__(d, sigma, useHMC_, hmc_epsilon_, hmc_iterN_, hmc_sigma_, gamma_)

        self.target_T0 = None # target distribution at temperature 0
        self.logq_val_grad_batch = None
        self.NChains = 1
        self.tseq = np.array([[1]])

        self.phimix = None # phimix for warp-u transformation
        self.mixK = -1
        self.shift = 0
        self.sI = 0
        self.tuning = False
        self.lr = 0.1
        self.tuning_rhoGap = 0.01

    def set_tuning_rhoGap(self, rhog):
        self.tuning_rhoGap = rhog
    
    def set_phimix(self, phimix_):
        self.phimix = phimix_
        self.mixK = phimix_.get_K()

    def set_tempering(self, np.ndarray tseq_):
        # if tseq_[0] <= 0:
        #     print("The first value should be positive!")
        #     return
        # if tseq_[-1] != 1:
        #     print("The last value should be one!")
        #     return
        self.tseq = tseq_
        self.NChains = len(tseq_) # the number of chains excluding inv. temp. at 0

    def set_tuning(self, double rho_init, double lr, 
                uint tuning_iterMin, uint tuning_iterGap):
        self.rho = rho_init
        self.tuning = True
        new_t = 1 / (1+math.exp(rho_init))
        self.tseq = np.hstack((self.tseq, new_t))
        self.NChains += 1
        self.lr = lr
        self.tuning_iterMin = tuning_iterMin
        self.tuning_iterGap = tuning_iterGap

    
    def set_targetT0(self, target_T0_):
        self.target_T0 = target_T0_
        self.sI = 1
    
    def set_logq_val_grad(self, logq_val_grad_):
        self.logq_val_grad_batch = logq_val_grad_
    
    def get_theta1All(self):
        # return the recorded parameters of the target chain
        return self.theta1All[:self.theta1Iter]

    cpdef np.ndarray eval_logq_pt(self, theta1Star):
        logq1Star = np.zeros((self.NChains, 3 ))
        logq1Star[:,1] = self.logq_fun(theta1Star).flatten()
        if self.target_T0 is not None:
            logq1Star[:,2] = self.target_T0.logphimix_batch(theta1Star).flatten()
        logq1Star[:,0] = logq1Star[:,1] * self.tseq + logq1Star[:,2] * (1-self.tseq)
        return logq1Star
    
    cdef np.ndarray eval_logq_grad_pt(self, np.ndarray theta1Star):
        cdef np.ndarray grad, tmp
        _, tmp = self.logq_val_grad_batch(theta1Star)
        grad = tmp * self.tseq[:,np.newaxis]
        if self.target_T0 is not None:
            _ , tmp = self.target_T0.logphimix_val_grad_batch(theta1Star)
            grad += tmp * (1-self.tseq[:,np.newaxis])
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
        cdef uint J

       
        self.swap_update()

        ###
        # single out chain 0 at classical_prepare()
        # by drawing iid samples
        ###

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
        diff1 = logq1Star[:,0] - proposedE - (self.logqChains[:,0] - currentE)
        diff1 = np.clip(diff1, -1e20, 0)
        acceptIdx = Uvar < np.exp(diff1)
        self.theta1 = np.where(acceptIdx[:,None], theta1Star, self.theta1)
        self.logqChains = np.where(acceptIdx[:,None], logq1Star, self.logqChains)

        if self.phimix is not None:
            self.warpu_update()
        
        # record the target chain
        self.record_theta1()

    cpdef void record_theta1(self):
        J = self.level2Index[-1]
        self.theta1Iter += 1
        self.theta1All[self.theta1Iter - 1] = np.copy(self.theta1[J])

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

        ## adjust the qVals based on temperature
        probv_log -= qVals
        if self.target_T0 is not None:
            qt0Star = self.target_T0.logphimix_batch(thetanew1.reshape(-1, self.d)).reshape(-1, self.mixK)
            qValsStar = qt0Star * (1-self.tseq[:,np.newaxis]) + qVals * self.tseq[:,np.newaxis]
        else:
            qValsStar = qVals * self.tseq[:,np.newaxis]
        probv_log +=  qValsStar 
        ##

        probv_max = np.max(probv_log, axis = 1)
        probv = probv_log - probv_max[:,None]
        probv = np.exp(probv)
        probv = probv / np.sum(probv, axis = 1)[:,None]
        for j in range( self.NChains):
            psi1 = rd.choice(range(self.mixK), 1, False, probv[j])[0]
            self.theta1[j] = np.copy(thetanew1[j, psi1, :])
            self.logqChains[j,0] = qValsStar[j, psi1]
            self.logqChains[j,1] = qVals[j, psi1]
            if self.target_T0 is not None:
                self.logqChains[j,2] = qt0Star[j, psi1]

    cpdef void swap_update(self):
        cdef np.ndarray logq1Star, Uvar
        cdef double diff1, before, after, after_tmp
        cdef uint J, A, B

        Uvar = rd.uniform(0,1, self.NChains+1)

        # swap update for chain 0 and chain 1
        if self.target_T0 is not None:
            self.swap_update_T0(Uvar[0])


        if self.NChains == 1:
                return

        # swap update for the rest of the chains
        if self.NChains > 2:
            self.shift = (self.shift + 1) % 2

        logq1Star = np.zeros((self.NChains, 3))
        for j in range(self.shift + 1, self.NChains, 2):
            A = self.level2Index[j]
            B = self.level2Index[j+1]
            logq1Star[A] = self.logqChains[B]
            logq1Star[B] = self.logqChains[A]
        logq1Star[:,0] = logq1Star[:,1] * self.tseq.flatten() + logq1Star[:,2] * (1-self.tseq.flatten())
   

        for j in range(self.shift+1, self.NChains, 2):
            A = self.level2Index[j]
            B = self.level2Index[j+1]
            before = self.logqChains[A,0] + self.logqChains[B,0]
            after = logq1Star[A,0] + logq1Star[B,0]
            diff1 = after - before
            diff1 = np.clip(diff1, -1e20, 0)
            if Uvar[j] < np.exp(diff1):
                self.level2Index[j] = B
                self.level2Index[j+1] = A
                self.tseq[A], self.tseq[B] = self.tseq[B], self.tseq[A]
                self.logqChains[A,0] = logq1Star[B,0]
                self.logqChains[B,0] = logq1Star[A,0]
        
        ######
        ## record the target chain if being swapped
        if j == self.NChains - 1:
            self.record_theta1()

    cpdef swap_update_T0(self, double Uvar):
        # swap update for chain 0 and chain 1
        cdef double diff1, before, after, after_tmp
        cdef uint B

        B = self.level2Index[1]
        before = self.logqChains[B,0] + self.logPi0_at0[self.theta0Iter]

        after_tmp = self.logQ_at0[self.theta0Iter] * self.tseq[B] +\
            self.logPi0_at0[self.theta0Iter] * (1-self.tseq[B])
        after = self.logqChains[B,2] + after_tmp
        diff1 = after - before
        diff1 = np.clip(diff1, -1e20, 0)
        if Uvar < np.exp(diff1):
            self.theta1[B] = self.theta0All[self.theta0Iter]
            self.logqChains[B,0] = after_tmp
            self.logqChains[B,1] = self.logQ_at0[self.theta0Iter]
            self.logqChains[B,2] = self.logPi0_at0[self.theta0Iter]
        self.theta0Iter += 1

    cpdef void classical_prepare(self, unsigned int m):
        self.m = m
        
        #self.timecost = np.zeros((m,1))

        if self.target_T0 is not None:
            self.theta0All = self.target_T0.drawsample(m+5)
            self.logPi0_at0 = self.target_T0.logphimix_batch(self.theta0All).flatten()
            self.logQ_at0 = self.logq_fun(self.theta0All).flatten()
            self.theta0Iter = 0

        self.theta1 = np.empty((self.NChains, self.d))
        for i in range(self.NChains):
            self.theta1[i] = self.pi0()
        self.logqChains = self.eval_logq_pt(self.theta1) 

        self.theta1All = np.zeros((2*m+10, self.d))
        self.theta1All[0] = np.copy(self.theta1[-1]) # X1
        self.mcmcIter = 1
        self.theta1Iter = 1
        
        # main chains with HMC starts from 1
        # 0 map to -1, but not used
        # 1 map to 0
        # 2 map to 1
        # ...
        self.level2Index = np.arange(self.NChains+1) - 1

        self.thetaAllLevels = np.zeros((m, self.NChains, self.d))
        if self.tuning:
            self.rhoseq = np.zeros(m+10)
            self.accAveSeq = np.zeros(m+10)
            self.acc_ave = 0.0
            self.rhoIter = 0

    cpdef void run_classical(self, unsigned int m):
        cdef unsigned int iter = 0
        self.classical_prepare(m)
        self.shift = 0
        for iter in range(2, m+1):
            self.singleTrans()
            if self.tuning and iter > self.tuning_iterMin and iter % self.tuning_iterGap == 1:
                self.update_rho(iter)
            # for j in range(self.NChains):
            #     B = self.level2Index[j+1]
            #     self.thetaAllLevels[iter-2, j] = np.copy(self.theta1[B])
        
    # def get_thetaAllLevels(self, unsigned int level):
    #     return self.thetaAllLevels[:, level,:]

    cpdef update_rho(self, uint iter):
        cdef double diff1, tupdate, before, after, after_tmp
        cdef np.ndarray logq1Star, tseqStar
        cdef uint A, B, j
        logq1Star = np.zeros((2, 3))
        tseqStar = np.zeros(2)
        j = self.NChains - 1
        if j > 0:
            A = self.level2Index[j]
            B = self.level2Index[j+1]
            logq1Star[0] = self.logqChains[B]
            logq1Star[1] = self.logqChains[A]
            tseqStar[0] = self.tseq[A]
            tseqStar[1] = self.tseq[B]
            logq1Star[:,0] = logq1Star[:,1] * tseqStar + logq1Star[:,2] * (1-tseqStar)
            before = self.logqChains[A,0] + self.logqChains[B,0]
            after = logq1Star[0,0] + logq1Star[1,0]
            diff1 = after - before
            diff1 = np.clip(diff1, -1e20, 0)
        elif j==0:
            B = self.level2Index[1]
            before = self.logqChains[B,0] + self.logPi0_at0[self.theta0Iter]
            after_tmp = self.logQ_at0[self.theta0Iter] * self.tseq[B] +\
                self.logPi0_at0[self.theta0Iter] * (1-self.tseq[B])
            after = self.logqChains[B,2] + after_tmp
            diff1 = after - before
            diff1 = np.clip(diff1, -1e20, 0)

        if iter == 2:
            self.acc_ave = np.exp(diff1)
        else:
            self.acc_ave = self.acc_ave * 0.5 + np.exp(diff1) * 0.5
        
        self.rho -= (self.lr) * (self.acc_ave - 0.23)

        
        tupdate = 1 / (1+math.exp(self.rho))
        if tupdate <= tseqStar[0] + self.tuning_rhoGap:
            tupdate = tseqStar[0] + self.tuning_rhoGap # the minimum gap is 0.01
            if tupdate >= 0.999:
                tupdate = 0.999
            self.rho = math.log(1/tupdate - 1)
        self.tseq[B] = tupdate


        self.rhoseq[self.rhoIter] = self.rho
        self.accAveSeq[self.rhoIter] = self.acc_ave
        self.rhoIter += 1



    def get_tuningseq(self):
        return self.rhoseq[:self.rhoIter-1], self.accAveSeq[:self.rhoIter-1]
        