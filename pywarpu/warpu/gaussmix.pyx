import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss


np.import_array()
ctypedef unsigned int uint


#%%
## Gauss Mixture for Sampling
cdef class GaussMixture:
    cdef unsigned int K, d
    cdef np.ndarray mu, Smat, SmatInv, logdets, w, w_log

    def __init__(self, K,  d):
        self.K = K
        self.d = d
        self.mu = np.zeros((K,d))
        self.Smat = np.zeros((K,d,d))
        self.SmatInv = np.zeros((K,d,d))
        self.logdets = np.zeros(K)
        for iterK in range(K):
            self.Smat[iterK,:,:] = np.identity(d)
            self.SmatInv[iterK,:,:] = np.identity(d)
        self.w = np.ones(K)/K
        self.w_log = np.log(self.w)


    cpdef uint get_K(self):
        return self.K

    cpdef void set_parameter(self, uint compI,np.ndarray mu,\
                     np.ndarray CovMat, double w_):
        cdef np.ndarray w, v, wSqrt

        w, v = np.linalg.eigh(CovMat)
        wSqrt = np.sqrt(w)
        self.mu[compI,:] = mu
        self.Smat[compI,:,:] = (v * wSqrt) @ v.T # matrix square root
        self.SmatInv[compI,:,:] = (v * (1/wSqrt)) @ v.T # inverse matrix square root
        self.logdets[compI] = np.sum(np.log(w)) / 2.0
        self.w[compI] = w_
        self.w_log[compI] = np.log(w_)


    cpdef void update_w(self, np.ndarray wAll_):
        self.w = wAll_
        self.w_log = np.log(wAll_)

    cpdef void update_mu(self, uint compI, np.ndarray mu):
        self.mu[compI,:] = mu

    cpdef  logpdf(self, np.ndarray theta_):
        # theta is a n-by-p matrix
        cdef np.ndarray probv, thetaw, pmax, psum, theta, result
        cdef uint iterK, n
        if theta_.ndim == 1:
            theta = np.copy(theta_).reshape(1,-1)
        else:
            theta = np.copy(theta_)
        n = theta.shape[0]
        probv = np.zeros((n, self.K))
        for iterK in range(self.K):
            # thetaw is p-by-n matrix
            thetaw = self.SmatInv[iterK,:,:] @ (theta - self.mu[iterK,:]).T
            probv[:,iterK] = np.sum(gauss.logpdf(thetaw),axis=0) - self.logdets[iterK]
        probv = probv + self.w_log
        pmax = np.max(probv, axis = 1).reshape(-1,1)
        psum = np.sum(np.exp(probv - pmax), axis = 1)
        result =  (pmax.flatten() + np.log(psum))
        if theta_.ndim == 1:
            return result[0]
        else:
            return result


    cpdef np.ndarray loggrad(self, np.ndarray theta):
        # theta is a p-vector
        cdef np.ndarray probv, thetaw
        cdef uint iterK, n
        cdef double pmax, psum
        n = theta.shape[0]
        probv = np.zeros(self.K)
        thetaw = theta - self.mu # mu is K-by-p
        for iterK in range(self.K):
            thetaw[iterK,:] = self.SmatInv[iterK,:,:] @ thetaw[iterK,:]

        probv = np.sum(gauss.logpdf(thetaw),axis=1) - self.logdets
        probv = probv + self.w_log
        pmax = np.max(probv)
        probv = np.exp(probv - pmax)
        probv = probv/np.sum(probv)

        for iterK in range(self.K):
            thetaw[iterK,:] = self.SmatInv[iterK,:,:] @ thetaw[iterK,:]
        return - (thetaw.T@probv)

    cpdef moment12(self):
        cdef np.ndarray m1, m2
        m1 = self.mu.T @ self.w
        m2 = (self.mu*self.mu).T @ self.w
        for iterK in range(self.K):
            tmp = self.Smat[iterK,:,:]
            m2 += np.diag(tmp@tmp) * self.w[iterK]
        return m1, m2

    
    cpdef np.ndarray drawsample(self, uint n):
        cdef np.ndarray result, aList, cList 
        cdef uint iter,j 

        result = np.random.normal(0,1,(n, self.d))
        #aList = np.arange(self.K)
        #print(np.sum(self.w))
        cList = np.random.choice(self.K, n, True, self.w)

        for iter in range(n):
            j = cList[iter]
            result[iter,:] = self.Smat[j,:,:] @ result[iter,:] + self.mu[j,:]

        return result