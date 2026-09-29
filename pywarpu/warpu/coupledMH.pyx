import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
from warpu.utils import  maximxalGaussDraw, estimateH_oneSample
from time import time

np.import_array()
#DTYPE = np.double
#ctypedef np.double_t DTYPE_t



@cython.boundscheck(False)
@cython.wraparound(False)
cdef class coupledMH:

    def __init__(self, int d, double  sigma):
        self.d = d
        self.sigma = sigma
        self.timeOut = 1e12
        self.verbose = False

    cpdef void set_verbose(self, bint v_):
        self.verbose = v_

    cpdef void setpi0(self, object pi0_):
        self.pi0 = pi0_

    cpdef void setlogq(self, object logq_):
        self.logq_fun = logq_

    cpdef void settimeOut(self, double timeOut_):
        self.timeOut = timeOut_

    cpdef unsigned int gettau(self):
        return self.tau



    def get_theta1All(self):
        return self.theta1All

    def get_theta2All(self):
        return self.theta2All

    def get_timecost(self):
        return self.timecost[:,0]

    def get_mFinal(self):
        return self.mFinal

    def get_acceptCount1(self):
        return self.acceptCount1
    
    def get_acceptCount2(self):
        return self.acceptCount2

    def get_theta1(self):
        return self.theta1

    cpdef void run(self, unsigned int k, unsigned int m):
        cdef unsigned int iter = 0
        cdef double startT

        self.k = k 
        self.m = m
        self.tau = -1
        self.theta1All = np.zeros((m,self.d))
        self.theta2All = np.zeros((m,self.d))
        self.timecost = np.zeros((m,1))
        self.acceptCount1 = np.zeros((m,1))
        self.acceptCount2 = np.zeros((m,1))

        startT = time()
        self.mcmcIter = 1
        self.theta1 = self.pi0()
        self.theta2 = np.copy(self.theta1)
        self.logq1 = self.logq_fun(self.theta1)
        self.logq2 = self.logq1
        self.singleTrans()

        self.theta2 = self.pi0()
        self.theta1All[0,:] = self.theta1 # X1
        self.theta2All[0,:] = self.theta2 # Y0
        self.logq1 = self.logq_fun(self.theta1)
        self.logq2 = self.logq_fun(self.theta2)
        self.flag = False # if the two chains have meet?

        for iter in range(2,m+1): # actual time stamp for X, from 1
            self.mcmcIter = iter # mcmcIter for X, from 1
            self.coupledTrans()
            self.theta1All[iter-1,:] = self.theta1
            self.theta2All[iter-1,:] = self.theta2
            self.timecost[iter-1,0] = time() - startT
            if self.flag: # self.tau > self.m and
                self.tau = iter # record meeting time, starting from 1,..
                break
            if self.timecost[iter-1,0] > self.timeOut:
                break
        
        if self.tau < self.m:
            for iter in range(self.tau+1, self.m+1):
                self.mcmcIter = iter
                self.singleTrans()
                #print(self.logq1)
                self.theta1All[iter-1,:] = self.theta1
                #self.theta2All[iter-1,:] = self.theta2
                self.timecost[iter-1,0] = time() - startT
                if self.timecost[iter-1,0] > self.timeOut:
                    break
        
        self.mFinal = iter
        
    cpdef void run_classical(self, unsigned int m):
        cdef unsigned int iter = 0
        cdef double startT
        startT = time()
        self.classical_prepare(m)
        #for iter in range(2, self.m+1):
        self.mcmcIter = 2
        for iter in range(2, m+1):
            # if self.verbose:
            #     print(iter)
            self.singleTrans()
            self.theta1All[self.mcmcIter-1] = self.theta1
            # self.timecost[self.mcmcIter-1,0] = time() - startT
            # if self.timecost[self.mcmcIter-1,0] > self.timeOut:
            #     break
            self.mcmcIter += 1

        self.mFinal = iter
        

    def eval(self, h_fun, step = -1):
        if step <= 0:
            return estimateH_oneSample(h_fun, self.theta1All, self.theta2All,\
                        self.k, self.mFinal, self.tau)
        else:
            result = np.zeros(self.m)
            for iter in range(self.k, self.mFinal+1, step):
                result[iter] = estimateH_oneSample(h_fun, self.theta1All, self.theta2All,\
                                    self.k, iter, self.tau)
            result = np.vstack((self.timecost[:,0], result))
            return result[:, range(self.k, self.mFinal+1, step)]
                    
    cpdef void singleTrans(self):
        self.singleMH()

    cpdef void  coupledTrans(self):
        self.coupledMH()
    
    cpdef void coupledMH(self):
        cdef np.ndarray theta1Star, theta2Star
        cdef bint flag1, flag2, flag3
        cdef double logq1Star, logq2Star, Uvar
        cdef double diff1, diff2
        theta1Star, theta2Star, flag1 = \
            maximxalGaussDraw(self.theta1, self.theta2, self.sigma)


        logq1Star = self.logq_fun(theta1Star)
        logq2Star = self.logq_fun(theta2Star)
        Uvar = rd.uniform(0,1)
        flag2 = False
        flag3 = False
        diff1 = logq1Star - self.logq1
        diff1 = np.min([diff1, 0])
        diff1 = np.max([diff1, -1e20])
        diff2 = logq2Star - self.logq2
        diff2 = np.min([diff2, 0])
        diff2 = np.max([diff2, -1e20])
        if Uvar < np.exp(diff1):
            self.theta1 = theta1Star
            self.logq1 = logq1Star
            flag2 = True
            self.acceptCount1[self.mcmcIter-1] = 1.0
        if Uvar < np.exp(diff2):
            self.theta2 = theta2Star
            self.logq2 = logq2Star
            flag3 = True
            self.acceptCount2[self.mcmcIter-1] = 1.0
        self.flag = flag1 and flag2 and flag3
    

    cpdef void singleMH(self):
        cdef np.ndarray theta1Star
        cdef double logq1Star, Uvar, diff1

        theta1Star = self.theta1 + rd.normal(0, self.sigma, self.d)
        logq1Star = self.logq_fun(theta1Star) 
        Uvar = rd.uniform(0,1)
        diff1 = logq1Star - self.logq1
        diff1 = np.min([diff1, 0])
        diff1 = np.max([diff1, -1e20])
        if Uvar < np.exp(diff1):
            self.theta1 = np.copy(theta1Star)
            self.logq1 = logq1Star
            self.acceptCount1[self.mcmcIter-1] = 1.0
            #self.theta2 = np.copy(theta1Star)
            #self.logq2 = logq1Star
 

    def get_logq1(self):
        return self.logq1

    cpdef void classical_prepare(self, unsigned int m):
        self.k = 0
        self.m = m
        self.tau = -1
        self.theta1All = np.zeros((m,self.d))
        #self.theta2All = np.zeros((m,self.d))
        self.timecost = np.zeros((m,1))

        
        self.theta1 = self.pi0()
        self.theta2 = np.copy(self.theta1)
        self.logq1 = self.logq_fun(self.theta1) 
        self.logq2 = self.logq1

        self.theta1All[0,:] = self.theta1 # X1
        self.acceptCount1 = np.zeros((m,1))
        #self.theta2All[0,:] = self.theta2 # Y0
        self.flag = False # if the two chains have meet?

        
    cpdef double pt_proceed0(self):
        self.mcmcIter += 1
        self.singleTrans()
        self.theta1All[self.mcmcIter-1,:] = self.theta1
        return self.logq1
    
    cpdef void swap_update0(self, np.ndarray theta_, double logq_):
        self.mcmcIter += 1
        self.theta1All[self.mcmcIter-1,:] = np.copy(theta_)
        self.theta1 = np.copy(theta_)
        self.logq1 = logq_