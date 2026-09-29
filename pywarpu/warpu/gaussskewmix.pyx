import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
import math
from scipy.stats import norm as gauss
from scipy.stats import truncnorm
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
cimport warpu.gaussmixfull

np.import_array()
ctypedef unsigned int uint


#%%
# The Auxillary Distribution with Warp-U transformation
# Guass Mixture with a skewness parameter
cdef class GaussSkewMix(warpu.gaussmixfull.GaussMixtureFull):
    cdef np.ndarray  alphaVec, betaVec, adjustVec, adjustSigmaInv
    cdef np.ndarray adjust_logdets, aSaP1

    def __init__(self, K,  d):
        super().__init__(K,d)
        self.alphaVec = np.zeros((K,d))
        self.adjustVec = np.zeros((K,d))
        self.adjustSigmaInv = np.zeros((K,d,d))
        self.adjust_logdets = np.zeros(K)
        self.aSaP1 = np.zeros(K)


    def set_parameter(self, uint compI, np.ndarray mu_,\
                    np.ndarray Lmat_, double w_,
                    np.ndarray alpha_):
        cdef double logabsdet, sign
        warpu.gaussmixfull.GaussMixtureFull.set_parameter(self, compI, mu_, Lmat_, w_)

        self.alphaVec[compI] = alpha_
        #Omega@alpha = L@L^T@alpha
        self.adjustVec[compI] = self.LMat[compI] @ (self.LtMat[compI] @ alpha_)
        
        self.aSaP1[compI] = 1 + np.sum(alpha_ * self.adjustVec[compI])
        self.adjustVec[compI] /= math.sqrt(self.aSaP1[compI])

        tmpMat = np.outer(self.adjustVec[compI], self.adjustVec[compI])
        self.adjustSigmaInv[compI] = self.LMat[compI]@self.LtMat[compI] - tmpMat # L@Lt - tmpMat
        self.adjust_logdets[compI] = math.log(self.aSaP1[compI]) + 2*self.logdets[compI]



    cpdef  component_logprob_info(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar, thetaMat, muVec, PhiOut
        cdef uint psi

        thetaMat = theta - self.mu
        muVec = np.sum(thetaMat*self.adjustVec, axis = 1)
        PhiOut = gauss.logcdf(muVec)
        muVec /= np.sqrt(self.aSaP1)

        probv = self.w_log - 0.5*self.d*np.log(2*np.pi) - 0.5*self.adjust_logdets
        for psi in range(self.K):
            thetaStar = self.adjustSigmaInv[psi,:,:] @ thetaMat[psi]
            probv[psi] -= 0.5*np.sum(thetaMat[psi] * thetaStar) 
        probv += PhiOut + math.log(2)
        return probv, muVec
    

    cpdef double logphimix(self, np.ndarray theta):
        cdef np.ndarray probv
        probv,_ = self.component_logprob_info(theta)
        maxP = np.max(probv)
        result = np.log(np.sum(np.exp(probv - maxP))) + maxP
        return result

    cpdef logphimix_val_grad(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar, thetaMat, muVec, PhiOut, gradMat
        cdef uint psi

        #gradMat = np.zeros((self.K, self.d))
        thetaMat = theta - self.mu
        muVec = np.sum(thetaMat * self.adjustVec, axis = 1)
        PhiOut = gauss.logcdf(muVec)
        dphiOut = gauss.logpdf(muVec) 

        # the gradient of each log component
        gradMat = np.exp(dphiOut - PhiOut).reshape(-1,1) * self.adjustVec
        thetaStar = np.zeros((self.K, self.d))
        for psi in range(self.K):
            thetaStar[psi] = self.adjustSigmaInv[psi,:,:] @ thetaMat[psi]
            gradMat[psi] -= thetaStar[psi]

        probv = self.w_log - 0.5*self.d*np.log(2*np.pi) - 0.5*self.adjust_logdets        
        probv -= 0.5*np.sum(thetaMat * thetaStar, axis = 1)  
        probv += PhiOut + math.log(2)
        #gradMat = np.exp(probv + dphiOut).reshape(-1,1) * self.adjustVec
        #gradMat -= np.exp(probv).reshape(-1,1) * thetaStar

        maxP = np.max(probv)
        probv = np.exp(probv - maxP)
        logPi_val = np.log(np.sum(probv)) + maxP
        probv /= np.sum(probv)
        logPi_grad = np.sum(gradMat * probv.reshape(-1,1), axis = 0)
        return logPi_val, logPi_grad
    
    cpdef component_logprob_info_batch(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar, thetaDeMean, muVec, PhiOut, gradMat
        cdef uint psi, nSample
        cdef double constPsi

        nSample = theta.shape[0]
        probv = np.empty((self.K, nSample))
        muVec = np.zeros((self.K, nSample))
        for psi in range(self.K):
            constPsi = self.w_log[psi] + math.log(2) - 0.5*self.d*np.log(2*np.pi) - 0.5*self.adjust_logdets[psi]
            thetaDeMean = theta - self.mu[psi]
            muVec[psi] = np.sum(thetaDeMean * self.adjustVec[psi], axis = 1)
            PhiOut = gauss.logcdf(muVec[psi])
            muVec[psi] /= np.sqrt(self.aSaP1[psi])
            thetaStar = thetaDeMean @  self.adjustSigmaInv[psi,:,:]
            probv[psi] = PhiOut + constPsi - 0.5*np.sum(thetaDeMean * thetaStar, axis = 1)

        return probv, muVec
    
    cpdef logphimix_batch(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar, thetaDeMean, muVec, PhiOut, gradMat
        cdef uint psi, nSample
        cdef double constPsi
        if theta.ndim == 1:
            return self.logphimix(theta)

        probv,_ = self.component_logprob_info_batch(theta)

        maxP = np.max(probv, axis = 0)
        probv = np.exp(probv - maxP)
        logPi_val = np.log(np.sum(probv, axis = 0)) + maxP
        return logPi_val

    cpdef logphimix_val_grad_batch(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStar, thetaDeMean, muVec, PhiOut, gradMat
        cdef uint psi, nSample
        nSample = theta.shape[0]

        # the gradient of each log component
        #gradMat = np.exp(dphiOut - PhiOut).reshape(-1,1) * self.adjustVec
        probv = np.empty((self.K, nSample))
        gradMat = np.zeros((self.K, nSample, self.d))
        for psi in range(self.K):
            constPsi = self.w_log[psi] + math.log(2) - 0.5*self.d*np.log(2*np.pi) - 0.5*self.adjust_logdets[psi]
            thetaDeMean = theta - self.mu[psi]
            muVec = np.sum(thetaDeMean * self.adjustVec[psi], axis = 1)
            PhiOut = gauss.logcdf(muVec)
            dphiOut = gauss.logpdf(muVec) 
            phiRatio = np.exp(dphiOut - PhiOut)
            thetaStar = thetaDeMean @  self.adjustSigmaInv[psi,:,:]

            probv[psi] = PhiOut + constPsi - 0.5*np.sum(thetaDeMean * thetaStar, axis = 1)
            gradMat[psi] = np.outer(phiRatio, self.adjustVec[psi]) - thetaStar


        maxP = np.max(probv, axis = 0)
        probv = np.exp(probv - maxP)
        logPi_val = np.log(np.sum(probv, axis = 0)) + maxP
        probv /= np.sum(probv, axis = 0)
        logPi_grad = np.sum(gradMat * probv[..., np.newaxis], axis = 0)
        return logPi_val, logPi_grad


    
    cpdef draw_forward(self, np.ndarray theta):
        cdef np.ndarray probv,  thetaStar
        cdef uint psi

        probv, thetaStarMat = self.forward_info(theta)
        psi = rd.choice(range(self.K), 1, False, probv)[0]
        thetaStar = np.copy(thetaStarMat[psi])

        return thetaStar, psi
    
    cpdef forward_info(self, np.ndarray theta):
        cdef np.ndarray probv, thetaStarMat
        cdef uint psi
        cdef double a, b,ww, a_transformed

        probv, muVec = self.component_logprob_info(theta)
        probv = np.exp(probv - np.max(probv))
        probv /= np.sum(probv)

        a = 0
        b = 1e10
        thetaStarMat = np.zeros((self.K, self.d))
        for psi in range(self.K):
            loc = muVec[psi]
            scale = 1/math.sqrt(self.aSaP1[psi])
            a_transformed = (a - loc) / scale
            rv = truncnorm(a_transformed, b, loc=loc, scale=scale)
            ww = rv.rvs(size = 1)[0]
            thetaStarMat[psi] = theta - ww * self.alphaVec[psi] - self.mu[psi]
            thetaStarMat[psi] = self.LtMat[psi] @ thetaStarMat[psi]
            #thetaStarMat[psi] = self.forward_trans(theta, psi, ww)
            
        return probv, thetaStarMat



    cpdef forward_info_batch(self, np.ndarray theta, uint returnLog):
        cdef np.ndarray probv, thetaStarMat, ww, phimixVals
        cdef uint psi, nSample
        cdef double a, b, a_transformed

        if theta.ndim == 1:
            return self.forward_info(theta)

        nSample = theta.shape[0]
        probv, muVec = self.component_logprob_info_batch(theta)
        phimixVals = logsumexp(probv, axis = 0)
        if returnLog == 0:            
            probv_max = np.max(probv, axis = 0)
            probv = np.exp(probv - probv_max)
            probv /= np.sum(probv, axis = 0)
        else:
            probv -= phimixVals

        a = 0
        b = 1e10
        thetaStarMat = np.zeros((nSample, self.K, self.d))
        ww = np.zeros(nSample)
        for psi in range(self.K):
            scale = 1/math.sqrt(self.aSaP1[psi])
            for n in range(nSample):
                loc = muVec[psi, n]
                a_transformed = (a - loc) / scale
                rv = truncnorm(a_transformed, b, loc=loc, scale=scale)
                ww[n] = rv.rvs(size = 1)[0]
            thetaStarMat[:,psi,:] = theta - np.outer(ww, self.alphaVec[psi]) - self.mu[psi]
            thetaStarMat[:,psi,:] =  thetaStarMat[:,psi,:] @ self.LMat[psi]
            #thetaStarMat[psi] = self.forward_trans(theta, psi, ww)
            
        return probv.T, thetaStarMat

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

    cpdef backward_info_batch(self, np.ndarray theta):
             # use importance sampling to draw backward transformation
        cdef np.ndarray thetanew, logprob, probv, expR, log_diff, maxall
        cdef np.ndarray qVals, thetanewMat, logprob_psi, probv_J, logQ_mat, log_diff_base
        cdef uint psi, nSample, n, rep, j
        
        rep = self.backward_rep
        nSample = theta.shape[0]
             

        sigmaSqInvSqrt = np.empty(( rep, nSample))
        logq_fun_psi = np.empty((rep, nSample))
        logprob_tmp = np.empty((rep, nSample))
        log_diff_base = gauss.logpdf(theta).sum(axis = 1) - np.log(rep) 
        
        # for the output
        thetanewMat = np.zeros((self.K, nSample, self.d))
        logQ_mat = np.zeros((self.K, nSample))
        logprob_psi = np.zeros((self.K, nSample))

        logprob = np.zeros((self.K, rep))   
        for psi in range(self.K):
  
            thetatmp = solve_triangular(self.LtMat[psi], theta.T, lower = False).T
            thetatmp = thetatmp + self.mu[psi]

            log_diff = np.full((rep, nSample), self.w_log[psi])
            log_diff += log_diff_base 

            ww = rd.normal(size = (rep, nSample)) 
            ww = np.abs(ww)
            for j in range(rep):
                thetanew = thetatmp + np.outer(ww[j], self.alphaVec[psi])
                logq_fun_psi[j] = self.logq_fun(thetanew)
                log_diff[ j] += logq_fun_psi[j] - self.logphimix_batch(thetanew)
            
            ww_selected = np.empty(nSample)
            maxall = np.max(log_diff, axis = 0)
            expR = np.exp(log_diff - maxall)
            probv_J = expR / np.sum(expR, axis = 0)
            for n in range(nSample):
                j = rd.choice(range(rep), 1, False, probv_J[:,n])[0]
                ww_selected[n] = ww[j, n]
                logQ_mat[psi, n] = logq_fun_psi[j, n]

            thetanewMat[psi] = thetatmp + np.outer(ww_selected, self.alphaVec[psi])
            logprob_psi[psi] = logsumexp(log_diff, axis = 0)

        return logprob_psi.T, thetanewMat.transpose(1,0,2), logQ_mat.T

    cpdef  backward_info(self, np.ndarray theta):
        # use importance sampling to draw backward transformation
        cdef np.ndarray thetanew, logprob, probv, expR, log_diff
        cdef np.ndarray qVals, thetanewMat, logprob_psi, probv_J
        cdef uint psi
        cdef double maxall
        
        cdef uint rep, j
        rep = self.backward_rep

        logprob = np.zeros((self.K, rep))        
        ww = rd.normal(size =  (self.K, rep)) 
        ww = np.abs(ww)
        for psi in range(self.K):
            thetanew = solve_triangular(self.LtMat[psi], theta, lower = False)
            thetanew = thetanew + self.mu[psi] + np.outer(ww[psi], self.alphaVec[psi])
            logprob[psi] = self.w_log[psi] + self.logq_fun(thetanew) - self.logphimix_batch(thetanew)
            #for j in range(rep):            
                #logprob[psi, j] -=  self.logphimix(thetanew[j,:])
        
        logprob += gauss.logpdf(theta).sum() - np.log(rep)
        logprob_psi = logsumexp(logprob, axis = 1)


        thetanewMat = np.zeros((self.K, self.d))
        for psi in range(self.K):           
            maxall = np.max(logprob[psi,:])
            expR = np.exp(logprob[psi,:] - maxall)
            probv_J = expR / np.sum(expR)
            j = rd.choice(range(rep), 1, False, probv_J)[0]
            thetanew = solve_triangular(self.LtMat[psi], theta, lower = False)
            thetanewMat[psi] =  thetanew + self.mu[psi] + ww[psi,j] * self.alphaVec[psi]
        
        qVals = self.logq_fun(thetanewMat)

        return logprob_psi, thetanewMat, qVals


    def drawsample(self, uint nsize, uint returnIndex = 0):
        cdef np.ndarray samples, cList
        cdef uint psi, j
        cdef double a, b, sigmaSqInv

        samples = rd.normal(size = (nsize, self.d))
        cList = np.random.choice(self.K, nsize, True, self.w)
        ww = rd.normal(size = nsize)
        for j in range(nsize):
            psi = cList[j]
            thetatmp = solve_triangular(self.LtMat[psi], samples[j], lower = False)
            samples[j,:] =  thetatmp + self.mu[psi,:] + ww[j] * self.alphaVec[psi]

        if returnIndex == 1:
            return samples, cList
        return samples
