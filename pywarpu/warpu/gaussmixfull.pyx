import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
#from scipy.sparse import csr_matrix
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
np.import_array()
#ctypedef unsigned int uint


#%%
# The Auxillary Distribution with Warp-U transformation
# Guass Mixture with Full Covariance Matrices
cdef class GaussMixtureFull:
    
    def __init__(self, K,  d):
        self.K = K
        self.d = d
        self.stN_const = -0.5 * self.d * np.log(2*np.pi)
        self.mu = np.zeros((K,d))
        self.Smat = np.zeros((K,d,d))
        self.logdets = np.zeros(K)

        self.w = np.ones(K)/K
        self.w_log = np.log(self.w)
        self.backward_rep = 500
        self.LtMat = [np.identity(d) for iterK in range(K)]
        self.LMat = [np.identity(d) for iterK in range(K)]

    def set_backward_rep(self, uint rep):
        self.backward_rep = rep
    
    cpdef uint get_K(self):
        return self.K

    cpdef np.ndarray get_mu(self, uint compI):
        return self.mu[compI,:]
    
    cpdef np.ndarray get_scale(self, uint compI):
        return np.linalg.inv(self.LtMat[compI]) #self.Smat[compI,:,:]
    
    cpdef np.ndarray get_invScale(self, uint compI):
        return self.LtMat[compI]
    
    cpdef double get_logdet(self, uint compI):
        return self.logdets[compI]

    def set_parameter(self, uint compI,np.ndarray mu,\
                     np.ndarray Lmat_, double w_):
        cdef double ss, logdet
        self.mu[compI] = mu

        self.w[compI] = w_
        self.w_log[compI] = np.log(w_)
        self.LMat[compI] = np.tril(Lmat_) 
        self.LtMat[compI] = self.LMat[compI].T

        cdef np.ndarray diagw
        diagw = np.diag(Lmat_)
        self.logdets[compI] = -np.sum(np.log(diagw))

    def get_weights(self):
        return self.w

    cpdef void setlogq(self, logq_fun_):
        self.logq_fun = logq_fun_

    cpdef void setlogq_grad(self, logq_grad_):
        self.logq_grad = logq_grad_

    cpdef np.ndarray backward_trans(self, np.ndarray theta, uint psi):
        cdef np.ndarray thetaw
        thetaw = solve_triangular(self.LtMat[psi], theta, lower = False)
        return thetaw + self.mu[psi] 

    cpdef np.ndarray forward_trans(self, np.ndarray theta, uint psi):
        cdef np.ndarray thetaw
        thetaw = self.LtMat[psi] @ (theta - self.mu[psi])
        return thetaw
    
    cpdef double warped_logq(self, np.ndarray theta):
        cdef np.ndarray thetanew, qVals, phiVals, probv
        cdef uint iterK
        cdef double result, pmax
        
        phiVals = np.empty(self.K)
        thetanew = np.empty( (self.K, self.d) )
        for iterK in range(self.K):
            thetanew[iterK,:] = self.backward_trans(theta, iterK)
            phiVals[iterK] = self.logphimix(thetanew[iterK,:])
        qVals = self.logq_fun(thetanew)

        phiVals = qVals - phiVals + self.w_log
        pmax = np.max(phiVals)
        phiVals = np.exp(phiVals - pmax)

        result = self.myGaussLogpdf(theta)
        result += pmax + np.log(np.sum(phiVals))
        return result

    cpdef myGaussLogpdf(self, np.ndarray theta):
        result = self.stN_const
        if theta.ndim == 1:
            result += -0.5 * np.sum(theta * theta) 
        else:
            result += -0.5 * np.sum(theta * theta, axis = 1) 
        
        return result

    cpdef double logphimix(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew
        cdef uint iterK
        cdef double pmax, psum
        
        thetanew = np.empty( (self.K, self.d) )
        for iterK in range(self.K):
            thetanew[iterK] = self.forward_trans(theta, iterK)

        probv = self.myGaussLogpdf(thetanew) - self.logdets + self.w_log
        pmax = np.max(probv)
        psum = np.sum(np.exp(probv - pmax))
        return (pmax + np.log(psum))
    
    cpdef logphimix_batch(self, np.ndarray theta):
        if theta.ndim == 1:
            return self.logphimix(theta)
        
        cdef np.ndarray probv, thetanew
        cdef uint iterK, nSample
        nSample = theta.shape[0]
        thetanew = np.empty((self.K, nSample, self.d))
        for iterK in range(self.K):
            thetanew[iterK] =  (theta - self.mu[iterK]) @ self.LMat[iterK] 

        probv = self.myGaussLogpdf(thetanew.reshape(-1,self.d)).reshape(self.K,nSample) 
        probv += -self.logdets[:,np.newaxis] + self.w_log[:,np.newaxis]
        result = logsumexp(probv, axis = 0)
        return result


    cpdef  logphimix_val_grad(self, np.ndarray theta):
        """
        Compute the gradient of the logarithm of the auxilary phi_mix.
        """
        cdef np.ndarray probv, thetanew, result_grad, thetanew2
        cdef double pmax, psum, val

        thetanew = np.empty( (self.K, self.d) )
        thetanew2 = np.empty( (self.K, self.d) )
        for iterK in range(self.K):
            thetanew[iterK] = self.forward_trans(theta, iterK)
            thetanew2[iterK] = self.LMat[iterK] @ thetanew[iterK]

        probv = self.myGaussLogpdf(thetanew) - self.logdets + self.w_log
        
        pmax = np.max(probv)
        probv = np.exp(probv - pmax)

        psum = np.sum(probv)
        result_grad = thetanew2.T @ (probv/(-psum))
        val = pmax + np.log(psum)
        return  val, result_grad

    cpdef logphimix_val_grad_batch(self, np.ndarray theta):
        cdef np.ndarray probv, result_grad, thetanew, thetanew2
        cdef uint i, nSample

        if theta.ndim == 1:
            return self.logphimix_val_grad(theta)

        nSample = theta.shape[0]
        probv = np.zeros((self.K, nSample))
        thetanew = np.empty((self.K, nSample, self.d))
        thetanew2 = np.empty((self.K, nSample, self.d))

        for iterK in range(self.K):
            thetanew[iterK] =  (theta - self.mu[iterK]) @ self.LMat[iterK] 
            thetanew2[iterK] =  thetanew[iterK] @ self.LtMat[iterK]
        
        probv = self.myGaussLogpdf(thetanew.reshape(-1,self.d)).reshape(self.K, nSample) 
        probv += -self.logdets[:,np.newaxis] + self.w_log[:,np.newaxis]
        val = logsumexp(probv, axis = 0)

        pmax = np.max(probv, axis = 0)
        probv = np.exp(probv - pmax)
        probv /= np.sum(probv, axis = 0)
        #result_grad = thetanew2.T @ (probv/(-psum))
        thetanew2 = thetanew2 * probv[...,np.newaxis]
        result_grad = -np.sum(thetanew2, axis = 0)

        return  val, result_grad


    cpdef forward_info(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew
        cdef uint iterK

        thetanew = np.empty((self.K, self.d))
        for iterK in range(self.K):
            thetanew[iterK] = self.forward_trans(theta, iterK)

        probv = self.myGaussLogpdf(thetanew) - self.logdets + self.w_log
        probv = np.exp(probv - np.max(probv)) 
        probv = probv / np.sum(probv)
        return probv, thetanew

    cpdef forward_info_batch(self, np.ndarray theta, uint returnLog):
        cdef np.ndarray probv, thetanew, probvMax, phimixVals
        cdef uint iterK, nSample
        if theta.ndim == 1:
            return self.forward_info(theta)
        nSample = theta.shape[0]

        thetanew = np.empty((self.K,nSample, self.d))
        for iterK in range(self.K):
            thetanew[iterK] =  (theta - self.mu[iterK]) @ self.LMat[iterK] 

        probv = self.myGaussLogpdf(thetanew.reshape(-1,self.d)).reshape(self.K,nSample) 
        probv += -self.logdets[:,np.newaxis] + self.w_log[:,np.newaxis]
        
        phimixVals = logsumexp(probv, axis = 0)
        if returnLog == 0:        
            probvMax = np.max(probv, axis = 0)
            probv = np.exp(probv - np.max(probv, axis = 0))
            probv = probv / np.sum(probv, axis = 0)
        else:
            probv -= phimixVals
        
        return probv.T, thetanew.transpose(1,0,2)

    cpdef np.ndarray forward_prob_log(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew, probv2
        cdef uint iterK
        cdef double pmax, lse

        thetanew = np.empty((self.K, self.d))
        for iterK in range(self.K):
            thetanew[iterK] = self.forward_trans(theta, iterK)
                
        probv = self.myGaussLogpdf(thetanew)  - self.logdets + self.w_log
        
        pmax = np.max(probv)
        probv2 = np.exp(probv - pmax)
        lse = pmax + np.log(np.sum(probv2))
        probv -= lse
        return probv

    
    cpdef  draw_forward(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew
        cdef uint psi

        probv,thetanew = self.forward_info(theta)
        psi = rd.choice(range(self.K), 1, False, probv)[0]
        return np.copy(thetanew[psi]), psi



    cpdef backward_info(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew, qVals
        cdef uint iterK
        probv = np.empty(self.K)
        thetanew = np.empty((self.K, self.d))

        for iterK in range(self.K):
            thetanew[iterK,:] = self.backward_trans(theta, iterK)
            probv[iterK] = self.forward_prob_log(thetanew[iterK,:])[iterK]            

        qVals = self.logq_fun(thetanew)
        probv = probv + self.logdets + qVals
        # probv = np.exp(probv - np.max(probv))
        # probv = probv / np.sum(probv)
        return (probv, thetanew, qVals)
    
    cpdef backward_info_batch(self, np.ndarray theta):
        cdef np.ndarray probv, thetanew, qVals, tmpArr
        cdef uint iterK, nsample
        if theta.ndim == 1:
            return self.backward_info(theta)
        nsample = theta.shape[0]
        probv = np.empty((self.K, nsample))
        thetanew = np.empty((self.K, nsample, self.d))

        for iterK in range(self.K):
            thetanew[iterK] = solve_triangular(self.LtMat[iterK], theta.T, lower = False).T + self.mu[iterK]

        tmpArr, _ = self.forward_info_batch(thetanew.reshape(-1,self.d), True)

        qVals = self.logq_fun(thetanew.reshape(-1,self.d))     
        probv = qVals.reshape(self.K, nsample)  + self.logdets[:,np.newaxis]
        for iterK in range(self.K):
            # range = iterK*nSample:(iterK+1)*nSample
            idx = slice(iterK*nsample, (iterK+1)*nsample)
            probv[iterK] += tmpArr[idx,iterK]
        
        probv = probv.T
        thetanew = thetanew.transpose(1,0,2)
        qVals = qVals.reshape(self.K, nsample).T
    
        return (probv, thetanew, qVals)

    cpdef np.ndarray logq_slice(self, np.ndarray theta, uint psi):
        """
        Compute the log target density for a single observation.
        """
        cdef np.ndarray probv
        qVals = self.logq_fun(theta)
        probv,_ = self.forward_info_batch(theta,1)
        return qVals + probv[:,psi]

    cpdef np.ndarray logq_slice_grad(self, np.ndarray theta, uint psi):
        """
        Compute the gradient of the log target density for a single observation.
        """
        cdef np.ndarray probv, thetanew, result_grad, tmp
        cdef double pmax, psum
        result_grad = self.logq_grad(theta)
        _, tmp = self.logphimix_val_grad_batch(theta)
        result_grad -= tmp
        result_grad -= (theta - self.mu[psi]) @ self.LMat[psi] @ self.LtMat[psi]
        return  result_grad

    # cpdef np.ndarray draw_backward(self, np.ndarray theta):
    #     cdef np.ndarray probv, thetanew
    #     cdef uint psi
    #     probv, thetanew = self.backward_info(theta)
    #     psi = rd.choice(range(self.K), 1, False, probv)[0]
    #     #thetaw = self.backward_trans(theta, psi)
    #     return np.copy(thetanew[psi,:])
    

    def  drawsample(self, uint n, uint returnIndex=0):
        cdef np.ndarray result, cList 
        cdef uint iter,j 
        self.w /= np.sum(self.w)
        result = np.random.normal(0,1,(n, self.d))
        cList = np.random.choice(self.K, n, True, self.w)

        for iter in range(n):
            j = cList[iter]
            result[iter] = self.backward_trans(result[iter], j)

        if returnIndex == 1:
            return result, cList
        return result

    def  drawsample_from(self, uint n, uint psi):
        cdef np.ndarray result
        cdef uint iter 
        result = np.random.normal(0,1,(n, self.d))
        for iter in range(n):
            result[iter] = self.backward_trans(result[iter], psi)

        return result