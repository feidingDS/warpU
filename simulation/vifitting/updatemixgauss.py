# update the mixture of skew Gaussian distribution
#%%
import numpy as np
import scipy.linalg
from scipy.linalg import  solve_triangular
import matplotlib.pyplot as plt
#%%

def get_samples_gauss( mu, PrecL, nSamples):
    Dim = len(mu)
    gauss_noise = np.random.randn(Dim, nSamples)
    theta = mu  + scipy.linalg.solve_triangular(PrecL.T, gauss_noise, lower = False).T
    return  theta,  PrecL@gauss_noise


def gauss_cvi_general(phimix, target_val_grad_batch,  nSamples,\
                      maxIters, init_m, init_P, ww, 
                        lr, lr_prec, lr_w = None, printGap = 1, seed = 1):
    np.random.seed(seed)
    # Initialize
    KComp = phimix.get_K()
    d = init_m.shape[1]
    mu = init_m
    Prec = init_P
    Prec_L = np.zeros((KComp, d, d))
    for psi in range(KComp):
        Prec_L[psi] = np.linalg.cholesky(Prec[psi])
    
    if isinstance(lr_prec, (int, float)):
        lr_prec = np.ones(KComp) * lr_prec

    kl_accum = 0
    kl_trace = np.zeros(maxIters)
    log_phimix_all = np.zeros(KComp*nSamples)
    logq_target_all = np.zeros(KComp*nSamples)
    gSigma_ave = np.zeros((KComp, d, d))
    gMu_ave = np.zeros((KComp, d))
    gW_ave = np.zeros(KComp)
    for t in range(maxIters):

        for psi in range(KComp):
            theta_samples, SNoise =\
                  get_samples_gauss( mu[psi], Prec_L[psi], nSamples)
            logq_target, grad_target = target_val_grad_batch(theta_samples)
            log_phimix, grad_phimix = phimix.logphimix_val_grad_batch(theta_samples)
            grad = grad_target - grad_phimix
            log_phimix_all[psi*nSamples:(psi+1)*nSamples] = log_phimix
            logq_target_all[psi*nSamples:(psi+1)*nSamples] = logq_target

            gMu = np.mean(grad, axis=0)
            gSigma = SNoise @ grad
            gSigma = (gSigma + gSigma.T) / (2*nSamples)

            gWtmp = np.mean(log_phimix - logq_target)
            if t == 0:
                gSigma_ave[psi] = gSigma
                gMu_ave[psi] = gMu
                gW_ave[psi] = gWtmp
            else:
                gSigma_ave[psi] = 0.5 * gSigma_ave[psi] + 0.5 * gSigma
                gMu_ave[psi] = 0.5 * gMu_ave[psi] + 0.5 * gMu
                gW_ave[psi] = 0.5 * gW_ave[psi] + 0.5 * gWtmp


            try:
                Gmat = solve_triangular(Prec_L[psi].T, gSigma_ave[psi], lower = False)
                Prec_trial = Prec[psi] - lr_prec[psi] * gSigma_ave[psi] + (0.5*lr_prec[psi]**2) * (Gmat.T @ Gmat)
                Prec_L[psi] = np.linalg.cholesky(Prec_trial)
            except np.linalg.LinAlgError:
                print('Cholesky failed! Stepsize too large for component %d' % psi)
                return 
            
            Prec[psi] = Prec_trial.copy()
            mu[psi] = mu[psi] + lr * gMu_ave[psi]
        
        # Update the weights  
        if lr_w is not None:
            ww[:] = ww - lr_w * gW_ave
            ww[:] /= np.sum(ww)
        for psi in range(KComp):
            phimix.set_parameter(psi, mu[psi], Prec_L[psi], ww[psi])

        kl = np.mean(logq_target_all - log_phimix_all)
        if t ==0:
            kl_accum = kl
        else:
            kl_accum = 0.5 * kl_accum + 0.5 * kl
        if printGap > 1 and t % printGap == 0:
            print("Iter: %d KL: %0.3f" % (t, kl_accum))
            # plt.scatter(logq_target_all, log_phimix_all, s = 3)
            # xx = np.linspace(np.min(logq_target_all), np.max(logq_target_all), 100)
            # plt.plot(xx, xx, 'r')
            # plt.show()

        kl_trace[t] = kl_accum

    return  kl_trace



# %%
