"""
Bridge-sampling estimators of a normalizing constant from Warp-U output.

  bridgeEst            - bridge sampling estimator (iterative, Meng & Wong 1996)
  stochastic_bridgeEst - stochastic Warp-U bridge sampling estimator
  one_kernel_seq       - convenience wrapper used in the paper's simulation study

This module is the same code as simulation/estimator/bridge.py (kept there so the
paper's simulation scripts run unchanged), exposed here as `warpu.bridge`.
"""
import numpy as np
import numpy.random as rd
from scipy.stats import norm
from scipy.special import logsumexp
import math
import time

def bridgeEst(logPhiMCMC, loqQMCMC, logPhiStd, loqQStd):
    cHat = 1
    for i in range(5):
        cHat = bridgeEst_internel(logPhiMCMC, loqQMCMC, logPhiStd, loqQStd, cHat)
    return cHat

def bridgeEst_internel(logPhiMCMC, loqQMCMC, logPhiStd, loqQStd, cHatInit):
    n1 = len(logPhiMCMC)
    n2 = len(logPhiStd)
    s1 = math.log(n1/(n1+n2))
    s2 = math.log(n2/(n1+n2))
    cHat_log = math.log(cHatInit)
    logMax1 = np.max([np.max(logPhiMCMC), np.max(loqQMCMC)])
    logPhiMCMC = logPhiMCMC - logMax1
    loqQMCMC = loqQMCMC - logMax1
    logAlphaMat = np.vstack([s2+cHat_log+logPhiMCMC, s1+loqQMCMC])
    logalphaMCMC = logsumexp(logAlphaMat, axis=0)


    logMax2 = np.max([np.max(logPhiStd), np.max(loqQStd)])
    logPhiStd = logPhiStd - logMax2
    loqQStd = loqQStd - logMax2
    logAlphaMat = np.vstack([s2+cHat_log+logPhiStd, s1+loqQStd])
    logalphaStd = logsumexp(logAlphaMat, axis=0)

    num = np.mean(np.exp(loqQStd - logalphaStd))
    den = np.mean(np.exp(logPhiMCMC - logalphaMCMC))
    return num/den

def stochastic_bridgeEst(logPhiMCMC, logQMatMCMC, logPhiStd, logQMatStd, psiAll):
    KComp = logQMatMCMC.shape[1]
    cHats = np.zeros(KComp)
    for psi in range(KComp): 
        selected = (psiAll==psi)
        if np.sum(selected) == 0:
            continue
        logPhiMCMC_sub = logPhiMCMC[selected]
        loqQMCMC_sub = logQMatMCMC[selected,psi]
        logPhiStd_sub = logPhiStd#[selected]
        loqQStd_sub = logQMatStd[:,psi]#[selected,psi]

        cHats[psi] = bridgeEst(logPhiMCMC_sub, loqQMCMC_sub, logPhiStd_sub, loqQStd_sub)
    return np.sum(cHats)


def stochastic_bridgeEst_v2(logPhiMCMC, quvMCMC, logPhiStd, quvStd, psiAll, KComp):
    cHats = np.zeros(KComp)
    for psi in range(KComp): 

        selected = (psiAll==psi)
        if np.sum(selected) == 0:
            continue
        logPhiMCMC_sub = logPhiMCMC[selected]
        loqQMCMC_sub = quvMCMC[selected]
        logPhiStd_sub = logPhiStd#[selected]
        loqQStd_sub = quvStd[:,psi]

        cHats[psi] = bridgeEst(logPhiMCMC_sub, loqQMCMC_sub, logPhiStd_sub, loqQStd_sub)
    return np.sum(cHats)


def compute_values(phimix, thetaSamples):
    KComp = phimix.get_K()
    nSample = len(thetaSamples)

    logPhi = np.sum(norm.logpdf(thetaSamples), axis=1)

    logprobMat, _, _ = phimix.backward_info_batch(thetaSamples)

    # for j in range(nSample):
    #     logprobv, _, _ = phimix.backward_info(thetaSamples[j])
    #     logprobMat[j] = logprobv

    logprobMax = np.max(logprobMat, axis=1)
    logp1 = logprobMat - logprobMax[:,None]
    logprobVec = np.log(np.sum(np.exp(logp1), axis=1)) + logprobMax
    return logPhi, logprobVec, logprobMat


#%%

def one_kernel_seq(thetaMCMC,  phimix, logq_fun, cutseq, estimatorSel):
    num_samples, theta_d = thetaMCMC.shape
    thetaStd = np.random.randn(num_samples, theta_d)
    thetaStars = np.zeros((num_samples, theta_d))
    psiAll = np.zeros(num_samples)
    probv, thetanew1 = phimix.forward_info_batch(thetaMCMC, False)
    KComp = phimix.get_K()
    weights = phimix.get_weights()
    weights_sel = np.zeros(num_samples)
    for j in range(num_samples):
        psi0 = rd.choice(range(KComp), 1, False, probv[j])[0]
        psiAll[j] = psi0
        thetaStars[j] = np.copy( thetanew1[j,psi0,:] )
        weights_sel[j] = weights[psi0]
    lenCut = len(cutseq)
    resultMat = np.full((lenCut, 3), np.nan)

    if estimatorSel[0] == 1 or estimatorSel[1] == 1:
        phimix.set_backward_rep(50)
        logPhiStd, logQStd, logQMatStd = compute_values(phimix, thetaStd)
        logPhiMCMC, logQMCMC, logQMatMCMC = compute_values(phimix, thetaStars)        


        for i, cut in enumerate(cutseq):
            logPhiMCMC_sub = logPhiMCMC[:cut]
            logQMCMC_sub = logQMCMC[:cut]
            logQMatMCMC_sub = logQMatMCMC[:cut]
            logPhiStd_sub = logPhiStd[:cut]
            logQStd_sub = logQStd[:cut]
            logQMatStd_sub = logQMatStd[:cut]
            resultMat[i,0] = bridgeEst(logPhiMCMC_sub, logQMCMC_sub, logPhiStd_sub, logQStd_sub)
            resultMat[i,1] = stochastic_bridgeEst(logPhiMCMC_sub, logQMatMCMC_sub, logPhiStd_sub, logQMatStd_sub, psiAll[:cut])

    stime = time.time()
    if estimatorSel[2] == 1:
        # third way to compute cHat
        phimix.set_backward_rep(1)
        logQMCMC = logq_fun(thetaMCMC)
        if estimatorSel[0] == 0:
            phimixValsMCMC = phimix.logphimix_batch(thetaMCMC)
            logPhiMCMC = np.sum(norm.logpdf(thetaStars), axis=1)
        quvMCMC = logPhiMCMC + logQMCMC - phimixValsMCMC + np.log(weights_sel)
        logPhiStd, logQStd, quvStd = compute_values(phimix, thetaStd)
        etime = time.time()
        for i, cut in enumerate(cutseq):
            logPhiMCMC_sub = logPhiMCMC[:cut]
            quvMCMC_sub = quvMCMC[:cut]
            logPhiStd_sub = logPhiStd[:cut]
            quvStd_sub = quvStd[:cut]
            resultMat[i,2] = stochastic_bridgeEst_v2(logPhiMCMC_sub, quvMCMC_sub, logPhiStd_sub, quvStd_sub, psiAll[:cut], KComp)
        etime2 = time.time()
    return resultMat
