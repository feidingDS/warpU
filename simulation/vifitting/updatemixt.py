# update the mixture of skew Gaussian distribution
#%%
import numpy as np
import math
import numpy.random as rd
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.linalg import  solve_triangular
#%%

def get_samples_gamma_gauss(gA, gB, mu, PrecL, nSamples):
    Dim = len(mu)
    gauss_noise = np.random.randn(Dim, nSamples)
    sigma = 1/ math.sqrt(rd.gamma(gA, 1.0/gB,size = 1)[0])
    theta = sigma* scipy.linalg.solve_triangular(PrecL.T, gauss_noise, lower = False).T
    SigmaInvThetaMu = sigma* (PrecL@gauss_noise)
    theta += mu
    return  theta, SigmaInvThetaMu


def gamma_cvi_general(phimix, target_val_grad_batch,  nSamples,\
                      maxIters, init_m, init_P, nu, ww,  lr, \
                      lr_prec, lr_w = None,
                    printGap = 1, seed = 1):
    np.random.seed(seed)
    # Initialize
    KComp = phimix.get_K()
    Dim = init_m.shape[1]
    mu = init_m
    Prec = init_P
    Prec_L = np.zeros((KComp, Dim, Dim))
    gSigma_ave = np.zeros((KComp, Dim, Dim))
    gMuAve = np.zeros((KComp, Dim))
    gW_ave = np.zeros(KComp)
    for psi in range(KComp):
        Prec_L[psi] = np.linalg.cholesky(Prec[psi])

    kl_accum = 0
    kl_trace = np.zeros(maxIters)
    log_phimix_all = np.zeros(KComp*nSamples)
    logq_target_all = np.zeros(KComp*nSamples)
    if isinstance(lr_prec, (int, float)):
        lr_prec = np.ones(KComp) * lr_prec
        
    for t in range(maxIters):

        for psi in range(KComp):
            theta_samples,  SNoise =\
                  get_samples_gamma_gauss(nu/2, nu/2, mu[psi], Prec_L[psi], nSamples)
            
            logq_target, grad_target = target_val_grad_batch(theta_samples)
            log_phimix, grad_phimix = phimix.logphimix_val_grad_batch(theta_samples)
            log_phimix_all[psi*nSamples:(psi+1)*nSamples] = log_phimix
            logq_target_all[psi*nSamples:(psi+1)*nSamples] = logq_target

            grad = grad_target - grad_phimix
            gMu = np.mean(grad, axis=0)
            gSigma = SNoise @ grad
            gSigma = (gSigma + gSigma.T) / (2*nSamples)

            gWtmp = np.mean(log_phimix - logq_target)
            if t==0:
                gSigma_ave[psi] = gSigma
                gMuAve[psi] = gMu
                gW_ave[psi] = gWtmp
            else:
                gSigma_ave[psi] = 0.5 * gSigma_ave[psi] + 0.5 * gSigma
                gMuAve[psi] = 0.5 * gMuAve[psi] + 0.5 * gMu
                gW_ave[psi] = 0.5 * gW_ave[psi] + 0.5 * gWtmp

            try:
                Gmat = solve_triangular(Prec_L[psi].T, gSigma_ave[psi], lower = False)
                Prec_trial = Prec[psi] - lr_prec[psi] * gSigma_ave[psi] + (0.5*lr_prec[psi]**2) * (Gmat.T @ Gmat)
                Prec_L[psi] = np.linalg.cholesky(Prec_trial)
            except np.linalg.LinAlgError:
                print('Cholesky failed! Stepsize too large for component %d' % psi)
                return 
                
            
            Prec[psi] = Prec_trial.copy()
            mu[psi] = mu[psi] + lr * gMuAve[psi]

        # Update the weights  
        if lr_w is not None:
            ww[:] = ww - lr_w * gW_ave
            ww[:] /= np.sum(ww)
        for psi in range(KComp):
            #phimix.set_parameter(psi, mu[psi], Prec_L[psi], ww[psi])
            phimix.set_parameter(psi, mu[psi], Prec_L[psi], ww[psi], nu/2, nu/2)
            
        kl = np.mean(logq_target_all - log_phimix_all)
        if t ==0:
            kl_accum = kl
        else:
            kl_accum = 0.9 * kl_accum + 0.1 * kl
        kl_trace[t] = kl_accum
        if printGap > 1 and t % printGap == 0:
            print("Iter: %d KL: %0.3f" % (t, kl_accum))
            # plt.scatter(logq_target_all, log_phimix_all, s = 3)
            # xx = np.linspace(np.min(logq_target_all), np.max(logq_target_all), 100)
            # plt.plot(xx, xx, 'r')
            # plt.show()

    return  kl_trace

