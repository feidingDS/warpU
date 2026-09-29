import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
import time

np.import_array()
#ctypedef double[:] ARR1D
#ctypedef double[:] ARR2D
ctypedef unsigned int uint


#%% 
# The Auxillary Distribution with Warp-U transformation
# Guass Mixture with Diagonal Scaling Matrices
cdef class GaussMixtureDiag:
    cdef unsigned int K, d
    cdef np.ndarray mu, Smat, SmatInv, logdets, w, w_log
    cdef object logq_fun, logq_grad
    cdef double stN_const

    
    def __init__(self, K,  d):
        self.K = K
        self.d = d
        self.stN_const = -0.5 * self.d * np.log(2*np.pi)
        self.mu = np.zeros((K,d))
        self.Smat = np.ones((K,d))
        self.SmatInv = np.ones((K,d))
        self.logdets = np.zeros(K)
        self.w = np.ones(K)/K
        self.w_log = np.log(self.w)

    cpdef void setlogq(self, logq_fun_):
        self.logq_fun = logq_fun_
    
    cpdef void setlogq_grad(self, logq_grad_):
        self.logq_grad = logq_grad_

    cpdef void set_parameter(self, uint compI,np.ndarray mu,\
                     np.ndarray CovDiag, double w_):
        cdef double ss, logdet
        self.mu[compI,:] = mu
        self.Smat[compI,:] = np.sqrt(CovDiag)
        self.SmatInv[compI,:] = 1/np.sqrt(CovDiag)
        self.logdets[compI] = np.sum(np.log(CovDiag))/2.0
        self.w[compI] = w_

        self.w_log[compI] = np.log(w_)


    def get_weights(self):
        return self.w

    cpdef uint get_K(self):
        return self.K

    cpdef np.ndarray get_mu(self, uint compI):
        return self.mu[compI,:]
    
    cpdef np.ndarray get_scale(self, uint compI):
        return self.Smat[compI,:]

    cpdef np.ndarray get_invScale(self, uint compI):
        return self.SmatInv[compI,:]

    cpdef double get_logdet(self, uint compI):
        return self.logdets[compI]
    
    cpdef double warped_logq(self, np.ndarray theta):
        """
        Compute the Warp-U transfomed target density.
        """
        cdef np.ndarray qVals, phiVals, probv, thetanew
        cdef uint iterK
        cdef double result, pmax
        

        thetanew = self.Smat * theta + self.mu

        phiVals = np.empty(self.K)  
        #qVals = np.zeros(self.K)   #########
        for iterK in range(self.K):
            phiVals[iterK] = self.logphimix(thetanew[iterK,:])
        qVals = self.logq_fun(thetanew) #######

        phiVals = qVals - phiVals + self.w_log

        pmax = np.max(phiVals)
        phiVals = np.exp(phiVals - pmax)

        result = np.sum(gauss.logpdf(theta))
        result += pmax + np.log(np.sum(phiVals))
        return result

    cpdef double logq_slice(self, np.ndarray theta, uint psi):
        """
        Compute the log target density for a single observation.
        """
        cdef np.ndarray probv
        qVals = self.logq_fun(theta)
        probv = self.forward_prob_log(theta)
        return qVals + probv[psi]

    cpdef np.ndarray logq_slice_grad(self, np.ndarray theta, uint psi):
        """
        Compute the gradient of the log target density for a single observation.
        """
        cdef np.ndarray probv, thetanew, result_grad
        cdef double pmax, psum
        result_grad = self.logq_grad(theta)
        result_grad -= self.logphimix_grad(theta)
        result_grad -= self.SmatInv[psi,:]*self.SmatInv[psi,:]*(theta - self.mu[psi,:])
        return  result_grad

    cpdef myGaussLogpdf(self, np.ndarray theta):
        result = -0.5 * np.sum(theta * theta, axis = 1) 
        result += self.stN_const
        return result

    cpdef double logphimix(self, np.ndarray theta):
        """
        Compute the logarithm of the auxilary phi_mix for a single observation.
        """
        cdef np.ndarray probv, thetanew
        cdef double pmax, psum
        thetanew = self.SmatInv * (theta - self.mu)
        #probv = np.sum(gauss.logpdf(thetanew), axis = 1) - self.logdets
        probv = self.myGaussLogpdf(thetanew) - self.logdets
        probv = probv + self.w_log
        pmax = np.max(probv)
        psum = np.sum(np.exp(probv - pmax))
        return (pmax + np.log(psum))


    
    cpdef  logphimix_val_grad(self, np.ndarray theta):
        """
        Compute the logarithm of the auxilary phi_mix.
        Return both the value and the gradient.
        """
        cdef np.ndarray probv, thetanew, result_grad
        cdef double pmax, psum, result_val
        thetanew = self.SmatInv * (theta - self.mu)
        #probv = np.sum(gauss.logpdf(thetanew), axis = 1) - self.logdets
        probv = self.myGaussLogpdf(thetanew) - self.logdets
        probv = probv + self.w_log
        pmax = np.max(probv)
        probv = np.exp(probv - pmax)

        psum = np.sum(probv)
        result_val = (pmax + np.log(psum))
        result_grad = (self.SmatInv * thetanew).T @ (probv/(-psum))
        return result_val, result_grad



    cpdef np.ndarray logphimix_grad(self, np.ndarray theta):
        """
        Compute the gradient of the logarithm of the auxilary phi_mix.
        """
        cdef np.ndarray probv, thetanew, result_grad
        cdef double pmax, psum
        thetanew = self.SmatInv * (theta - self.mu)
        #probv = np.sum(gauss.logpdf(thetanew), axis = 1) - self.logdets
        probv = self.myGaussLogpdf(thetanew) - self.logdets
        probv = probv + self.w_log
        pmax = np.max(probv)
        probv = np.exp(probv - pmax)

        psum = np.sum(probv)
        result_grad = (self.SmatInv * thetanew).T @ (probv/(-psum))
        return  result_grad


    cpdef forward_info(self, np.ndarray theta):
        """
        thetanew: K-by-d matrics, each row is one candidate 
              theta value after the Warp-U transformation. 
        probv: the probability of selecting each candidate theta
        """
        cdef np.ndarray probv, thetanew

        thetanew = self.SmatInv * (theta - self.mu)

        #probv = np.sum(gauss.logpdf(thetanew), axis = 1) - self.logdets + self.w_log
        probv = self.myGaussLogpdf(thetanew) - self.logdets + self.w_log
        probv = np.exp(probv - np.max(probv)) 
        probv = probv / np.sum(probv)
        return probv, thetanew

    cpdef np.ndarray forward_prob_log(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew, probv2
        cdef double logsumexp, pmax

        thetanew = self.SmatInv * (theta - self.mu)
        #probv = np.sum(gauss.logpdf(thetanew), axis = 1) - self.logdets + self.w_log
        probv = self.myGaussLogpdf(thetanew) - self.logdets + self.w_log
        pmax = np.max(probv)
        probv2 = np.exp(probv - pmax)
        logsumexp = pmax + np.log(np.sum(probv2))
        probv = probv - logsumexp
        return probv




    cpdef backward_info(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew, thetaw, qVals
        cdef uint iterK

        thetanew = self.Smat * theta + self.mu
        qVals = self.logq_fun(thetanew)
        probv = self.logdets + qVals
        for iterK in range(self.K):
            probv[iterK] += self.forward_prob_log(thetanew[iterK,:])[iterK]

        return (probv, thetanew, qVals)

    cpdef  draw_forward(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew
        cdef uint psi
    
        probv,thetanew = self.forward_info(theta)
        psi = rd.choice(range(self.K), 1, False, probv)[0]
        return np.copy(thetanew[psi,:]), psi


    cpdef drawsample(self, uint n):
        cdef np.ndarray result, aList, cList 
        cdef uint iter,j 

        result = np.random.normal(0,1,(n, self.d))
        #aList = np.arange(self.K)
        #print(np.sum(self.w))
        cList = np.random.choice(self.K, n, True, self.w)

        for iter in range(n):
            j = cList[iter]
            result[iter] = self.Smat[j] * result[iter] + self.mu[j]

        return result, cList

    cpdef np.ndarray draw_backward(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew, tmp
        cdef uint psi
        cdef double pmax
        probv, thetanew, tmp = self.backward_info(theta)
        pmax = np.max(probv)
        probv = np.exp(probv - pmax)
        probv = probv / np.sum(probv)
        psi = rd.choice(range(self.K), 1, False, probv)[0]
        return np.copy(thetanew[psi,:])
    
    # cpdef np.ndarray backward_trans(self, np.ndarray theta, uint psi):
    #     cdef np.ndarray thetaw
    #     thetaw = self.Smat[psi,:] * theta + self.mu[psi,:]
    #     return thetaw
    #cpdef np.ndarray forward_trans(self, np.ndarray theta, uint psi):
    #    cdef np.ndarray thetaw
    #    thetaw = self.SmatInv[psi,:] * (theta - self.mu[psi,:])
    #    return thetaw

    #cpdef np.ndarray forward_prob(self, np.ndarray theta):
    #    cdef np.ndarray probv
    #    probv, _ = self.forward_info(theta)
    #    return probv
    

    cpdef logphimix_val_grad_batch(self, np.ndarray theta):
        cdef np.ndarray probv, gradMat
        cdef uint i, n 
        n = theta.shape[0]
        probv = np.zeros(n)
        gradMat = np.zeros((n, self.d))
        for i in range(n):
            probv[i], gradMat[i] = self.logphimix_val_grad(theta[i])
        return probv, gradMat

    cpdef logphimix_batch(self, np.ndarray theta):
        if theta.ndim == 1:
            return self.logphimix(theta)
        cdef np.ndarray result
        cdef uint iter, n
        n = theta.shape[0]
        result = np.empty(n)
        for iter in range(n):
            result[iter] = self.logphimix(theta[iter])
        return result