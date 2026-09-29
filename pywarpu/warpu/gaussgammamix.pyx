import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
import math
from scipy.stats import norm as gauss
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
cimport warpu.gaussmixfull

np.import_array()
ctypedef unsigned int uint


#%%
# The Auxillary Distribution with Warp-U transformation
# Guass Mixture with Gamma Scale
cdef class GaussGammaMix(warpu.gaussmixfull.GaussMixtureFull):
    cdef np.ndarray  alphaVec, betaVec

    def __init__(self, K,  d):
        super().__init__(K,d)
        self.alphaVec = np.zeros(K)
        self.betaVec = np.zeros(K)
    

    def set_parameter(self, uint compI, np.ndarray mu,\
                    np.ndarray Lmat, double w_,
                    double a_, double b_):
        warpu.gaussmixfull.GaussMixtureFull.set_parameter(self, compI, mu, Lmat, w_)
        self.alphaVec[compI] = a_
        self.betaVec[compI] = b_

    cpdef component_logprob_info(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar, thetaStarMat, aVec, bVec
        cdef uint psi
        cdef double sbar, alpha, beta, maxP, result, a, b

        aVec = np.zeros(self.K)
        bVec = np.zeros(self.K)
        thetaStarMat = np.zeros((self.K, self.d))
        probv = self.w_log - 0.5*self.d*np.log(2*3.1415926) - self.logdets
        for psi in range(self.K):
            alpha = self.alphaVec[psi]
            beta = self.betaVec[psi]
            a = alpha + 0.5 * self.d
            thetaStar = self.forward_trans(theta, psi)
            b = beta + 0.5 * np.sum(thetaStar*thetaStar)
            probv[psi] += -math.lgamma(alpha) + alpha*math.log(beta)
            probv[psi] += math.lgamma(a) - (a) * math.log(b)
            aVec[psi] = a
            bVec[psi] = b
            thetaStarMat[psi,:] = thetaStar
        return probv, aVec, bVec, thetaStarMat
    
    cpdef component_logprob_info_batch(self, np.ndarray theta):
        nsample = theta.shape[0]
        result = np.zeros(nsample)

        probConst = self.w_log - 0.5*self.d*np.log(2*3.1415926) - self.logdets   
        probv = np.empty((self.K, nsample))
        thetaStarMat = np.zeros((self.K, nsample, self.d))
        aMat = np.zeros((self.K, nsample))
        bMat = np.zeros((self.K, nsample))   
        for psi in range(self.K):
            alpha = self.alphaVec[psi]
            beta = self.betaVec[psi]
            
            thetaStar = (theta - self.mu[psi,:]) @ self.LMat[psi]
            a = alpha + 0.5 * self.d
            bVec = beta + 0.5 * np.sum(thetaStar*thetaStar, axis = 1)
            aMat[psi] = a
            bMat[psi] = bVec
            thetaStarMat[psi] = thetaStar
            probv[psi] =  -a * np.log(bVec)
            probConst[psi] += -math.lgamma(alpha) + alpha*math.log(beta)+\
                math.lgamma(a)
        
        probv += probConst[:,np.newaxis]
        #result = logsumexp(probv, axis = 0)
        return probv, aMat, bMat, thetaStarMat
    

    cpdef double logphimix(self, np.ndarray theta):
        cdef np.ndarray probv
        probv, _, _, _ = self.component_logprob_info(theta)
        maxP = np.max(probv)
        result = np.log(np.sum(np.exp(probv - maxP))) + maxP
        return result
    
    cpdef  logphimix_batch(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar,   bVec, result, probvMax
        cdef uint psi, nsample
        cdef double  alpha, beta,  a

        if theta.ndim == 1:
            theta = theta.reshape(1,-1)
        nsample = theta.shape[0]
        result = self.component_logprob_info_batch(theta)[0]
        result = logsumexp(result, axis = 0)
        if nsample == 1:
            return result[0]

        return result

    cpdef logphimix_val_grad(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar
        cdef uint psi
        cdef double sbar, alpha, beta, maxP, result, a, b, logPi_val
        cdef np.ndarray logPi_grad, gradMat

        gradMat = np.zeros((self.K, self.d))
        probv = self.w_log - 0.5*self.d*np.log(2*3.1415926) - self.logdets
        for psi in range(self.K):
            alpha = self.alphaVec[psi]
            beta = self.betaVec[psi]
            a = alpha + 0.5 * self.d
            #self.SmatInv[psi,:,:] @ (theta - self.mu[psi,:])
            thetaStar = self.forward_trans(theta, psi)
            b = beta + 0.5 * np.sum(thetaStar*thetaStar)
            probv[psi] += - math.lgamma(alpha) + alpha*math.log(beta)
            probv[psi] += math.lgamma(a) - (a) * math.log(b)
            gradMat[psi] = - a/b * (self.LMat[psi] @ thetaStar)
        
        maxP = np.max(probv)
        probv = np.exp(probv - maxP)
        logPi_val = np.log(np.sum(probv)) + maxP
        probv /= np.sum(probv)
        logPi_grad = np.sum(gradMat * probv.reshape(-1,1), axis = 0)
        return logPi_val, logPi_grad


    cpdef logphimix_val_grad_batch(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar,   bVec, result, probvMax, gradMat
        cdef uint psi, nsample
        cdef double  alpha, beta,  a

        if theta.ndim == 1:
            theta = theta.reshape(1,-1)
        nsample = theta.shape[0]
        result = np.zeros(nsample)

        probConst = self.w_log - 0.5*self.d*np.log(2*3.1415926) - self.logdets   
        probv = np.empty((self.K, nsample)) 

        gradMat = np.zeros((self.K, nsample, self.d))
        for psi in range(self.K):
            alpha = self.alphaVec[psi]
            beta = self.betaVec[psi]
            
            thetaStar = (theta - self.mu[psi,:]) @ self.LMat[psi]
            a = alpha + 0.5 * self.d
            bVec = beta + 0.5 * np.sum(thetaStar*thetaStar, axis = 1)
            probv[psi] =  -a * np.log(bVec)
            probConst[psi] += -math.lgamma(alpha) + alpha*math.log(beta)+\
                math.lgamma(a)
            # - a/b * (self.LMat[psi] @ thetaStar)
            gradMat[psi] = -a/bVec.reshape(-1,1) * (thetaStar @ self.LtMat[psi])
        
        probv += probConst[:,np.newaxis]

        probMax = np.max(probv, axis = 0)
        probNorm = np.exp(probv - probMax)
        probNorm /= np.sum(probNorm, axis = 0)
        gradResult = np.sum(gradMat * probNorm[..., np.newaxis], axis = 0)

        result = logsumexp(probv, axis = 0)
        if nsample == 1:
            return result[0], gradResult

        return result, gradResult

    
    cpdef  draw_forward(self, np.ndarray theta):
        cdef np.ndarray probv,  thetaStar,thetaStarMat, aVec, bVec
        cdef uint psi
        cdef double sbar, a, b, sigmaSqInv

        probv, aVec, bVec, thetaStarMat = self.component_logprob_info(theta)

        # sample the component index
        probv = np.exp(probv - np.max(probv))
        probv /= np.sum(probv)
        psi = rd.choice(range(self.K), 1, False, probv)[0]
        thetaStar = thetaStarMat[psi]

        # sample 1/sigmaSq from gamma, i.e., sigmaSq from inverse gamma
        a = aVec[psi]
        b = bVec[psi]
        sigmaSqInv = rd.gamma(a, 1.0/b,size = 1)[0] # 1/b is the scale parameter
        thetaStar *= math.sqrt(sigmaSqInv) #############

        return thetaStar, psi
    
    cpdef forward_info(self, np.ndarray theta):
        cdef np.ndarray probv,  thetaStar,thetaStarMat, aVec, bVec
        cdef uint psi
        cdef double sbar, a, b, sigmaSqInv

        probv, aVec, bVec, thetaStarMat = self.component_logprob_info(theta)

        probv = np.exp(probv - np.max(probv))
        probv /= np.sum(probv)

        for psi in range(self.K):
            a = aVec[psi]
            b = bVec[psi]
            sigmaSqInv = rd.gamma(a, 1.0/b,size = 1)[0] # 1/b is the scale parameter
            thetaStarMat[psi] *= math.sqrt(sigmaSqInv) #############
            
        return probv, thetaStarMat

    cpdef forward_info_batch(self, np.ndarray theta,  uint returnLog):
        cdef np.ndarray probv, thetaStarMat, aMat, bMat,  a, b, sigmaSqInv
        cdef uint psi

        probv, aMat, bMat, thetaStarMat = self.component_logprob_info_batch(theta)
        phimixVals = logsumexp(probv, axis = 0)
        if returnLog == 0:        
            probvMax = np.max(probv, axis = 0)
            probv = np.exp(probv - np.max(probv, axis = 0))
            probv = probv / np.sum(probv, axis = 0)
        else:
            probv -= phimixVals
        #probv = logsumexp(probv, axis = 0)
        for psi in range(self.K):
            a = aMat[psi]
            b = bMat[psi]
            
            sigmaSqInv = rd.gamma(a, 1.0/b,size = theta.shape[0] ) # 1/b is the scale parameter
            thetaStarMat[psi] *= np.sqrt(sigmaSqInv).reshape(-1,1) #############
            
        return probv.T, thetaStarMat.transpose(1,0,2)

    # cpdef backward_info_batch(self, np.ndarray theta):
    #     cdef np.ndarray probs, thetas, qVals
    #     cdef uint psi, nSample

    #     nSample = theta.shape[0]
    #     probs = np.zeros((nSample, self.K))
    #     thetas = np.zeros((nSample, self.K, self.d))
    #     qVals = np.zeros((nSample, self.K))
    #     for n in range(nSample):
    #         probs[n], thetas[n], qVals[n] = self.backward_info(theta[n])
    #     return probs, thetas, qVals

    cpdef  backward_info(self, np.ndarray theta):
        # use importance sampling to draw backward transformation
        cdef np.ndarray thetanew, log_diff, sigmaSqInv, sigmaSqInvSqrt, probv, expR
        cdef np.ndarray qVals, thetanewMat, logprob_psi, thetatmp, probv_J 
        cdef uint psi
        cdef double maxall, ss
        
        cdef uint rep, j
        rep = self.backward_rep

        log_diff = np.zeros((self.K, rep))
        sigmaSqInv =  np.zeros((self.K, rep))
        sigmaSqInvSqrt = np.zeros((self.K, rep))

        log_diff += gauss.logpdf(theta).sum() - np.log(rep)
        log_diff += self.w_log[..., np.newaxis]

        thetanewMat = np.zeros((self.K, self.d))
        for psi in range(self.K):
            alpha = self.alphaVec[psi]
            beta = self.betaVec[psi] # scale parameter
            sigmaSqInv[psi] = rd.gamma(alpha, 1.0/beta, size =  rep ) # use scale parameter, not rate
            sigmaSqInvSqrt[psi] = np.sqrt(sigmaSqInv[psi,:])
            #log_diff[psi] += self.w_log[psi]
            thetatmp = solve_triangular(self.LtMat[psi], theta, lower = False)
            thetanew = thetatmp.reshape(1,-1) / sigmaSqInvSqrt[psi].reshape(-1,1) + self.mu[psi]
            log_diff[psi] += self.logq_fun(thetanew) - self.logphimix_batch(thetanew)
            # for j in range(rep):            
            #     thetanew =  thetatmp / sigmaSqInvSqrt[psi, j] + self.mu[psi]
            #     log_diff[psi, j] += self.logq_fun(thetanew) - self.logphimix(thetanew)

            # record one of the thetanew
            maxall = np.max(log_diff[psi])
            expR = np.exp(log_diff[psi] - maxall)
            probv_J = expR / np.sum(expR)
            j = rd.choice(range(rep), 1, False, probv_J)[0]
            thetanewMat[psi] = thetanew[j]


        logprob_psi = logsumexp(log_diff, axis = 1)
        qVals = self.logq_fun(thetanewMat)
          

        return logprob_psi, thetanewMat, qVals

    cpdef backward_info_batch2(self, np.ndarray theta):
        cdef np.ndarray probs, thetas, qVals
        cdef uint psi, nSample

        nSample = theta.shape[0]
        probs = np.zeros((nSample, self.K))
        thetas = np.zeros((nSample, self.K, self.d))
        qVals = np.zeros((nSample, self.K))
        for n in range(nSample):
            probs[n], thetas[n], qVals[n] = self.backward_info(theta[n])
        return probs, thetas, qVals

    cpdef  backward_info_batch(self, np.ndarray theta):
        # use importance sampling to draw backward transformation
        cdef np.ndarray thetanew, log_diff, sigmaSqInvSqrt, probv, expR
        cdef np.ndarray qVals, thetanewMat, logprob_psi, thetatmp, probv_J,maxall
        cdef uint psi, nSample
        cdef np.ndarray logq_fun_psi, logphimix_psi, log_diff_base, sigmaSqInvSqrtSelected

        if theta.ndim == 1:
            return self.backward_info(theta)
        
        cdef uint rep, j
        rep = self.backward_rep
        nSample = theta.shape[0]

        sigmaSqInvSqrt = np.empty(( rep, nSample))
        logq_fun_psi = np.empty((rep, nSample))
        logphimix_psi = np.empty((rep, nSample))
        log_diff_base = gauss.logpdf(theta).sum(axis = 1) - np.log(rep) 
        
        # for the output
        thetanewMat = np.zeros((self.K, nSample, self.d))
        logQ_mat = np.zeros((self.K, nSample))
        logprob_psi = np.zeros((self.K, nSample))

        for psi in range(self.K):
            log_diff = np.full((rep, nSample), self.w_log[psi])
            log_diff += log_diff_base  
            alpha = self.alphaVec[psi]
            beta = self.betaVec[psi] # scale parameter
            thetatmp = solve_triangular(self.LtMat[psi], theta.T, lower = False).T
            for j in range(rep):
                tmp = rd.gamma(alpha, 1.0/beta, size =  nSample ) # use scale parameter, not rate
                sigmaSqInvSqrt[j] = np.sqrt(tmp)
                thetanew = thetatmp / sigmaSqInvSqrt[j].reshape(-1,1) + self.mu[psi]
                logq_fun_psi[j] = self.logq_fun(thetanew)
                logphimix_psi[j] = self.logphimix_batch(thetanew)
                log_diff[j] += logq_fun_psi[j] - logphimix_psi[j]

            # record one of the thetanew
            maxall = np.max(log_diff, axis = 0)
            expR = np.exp(log_diff - maxall) #size rep x nSample
            probv_J = expR / np.sum(expR, axis = 0)
            sigmaSqInvSqrtSelected = np.zeros(nSample)
            for n in range(nSample):
                j = rd.choice(range(rep), 1, False, probv_J[:,n])[0]
                sigmaSqInvSqrtSelected[n] = sigmaSqInvSqrt[j,n]
                logQ_mat[psi,n] = logq_fun_psi[j,n]
            thetanewMat[psi] = thetatmp / sigmaSqInvSqrtSelected.reshape(-1,1) + self.mu[psi]
            logprob_psi[psi] = logsumexp(log_diff, axis = 0)

        return logprob_psi.T, thetanewMat.transpose(1,0,2), logQ_mat.T



    def drawsample(self, uint nsize, uint returnIndex = 0):
        cdef np.ndarray samples, cList
        cdef uint psi, j
        cdef double a, b, sigmaSqInv

        samples = rd.normal(size = (nsize, self.d))
        cList = np.random.choice(self.K, nsize, True, self.w)
        for j in range(nsize):
            psi = cList[j]
            a = self.alphaVec[psi]
            b = self.betaVec[psi] # scale parameter
            sigmaSqInv = rd.gamma(a, 1.0/b, size = 1)[0] # use scale parameter, not rate
            thetatmp = solve_triangular(self.LtMat[psi], samples[j], lower = False)
            samples[j,:] =  thetatmp / math.sqrt(sigmaSqInv) + self.mu[psi,:]

        if returnIndex == 1:
            return samples, cList
        return samples