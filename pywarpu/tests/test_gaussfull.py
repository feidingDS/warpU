#%%

import unittest
import numpy as np
import warpu.gaussmixfull as gmf
from scipy.stats import multivariate_t
#%%
KComp = 2
theta_d = 5

mu = np.random.rand(KComp, theta_d)
nu = 10
covs = np.zeros((KComp, theta_d, theta_d))
for k in range(KComp):
    UMat = np.random.rand(theta_d, theta_d)
    covs[k] = np.dot(UMat, UMat.T) + 0.1*np.eye(theta_d)
ww = np.ones(KComp)/KComp

phimix = gmf.GaussMixtureFull(KComp, theta_d)
for k in range(KComp):
    Prec = np.linalg.inv(covs[k])
    PrecL = np.linalg.cholesky(Prec)
    phimix.set_parameter(k, mu[k],PrecL, ww[k])

def logqfun(theta):
    if theta.ndim == 1:
        return -0.5*np.sum(theta**2)
    else:
        return -0.5*np.sum(theta**2, axis=1)

phimix.setlogq(logqfun)

# %%

class TestMyFunction(unittest.TestCase):
    def test_values(self):
        theta = np.random.rand(3, theta_d)
        a = phimix.logphimix(theta[0])
        b = phimix.logphimix(theta[1])
        logps = phimix.logphimix_batch(theta)
        self.assertAlmostEqual(a, logps[0], places=6)
        self.assertAlmostEqual(b, logps[1], places=6)
    
    def test_valgrad(self):
        theta = np.random.rand(3, theta_d)
        a, b = phimix.logphimix_val_grad(theta[0])
        c, d = phimix.logphimix_val_grad(theta[1])
        logps, lograds = phimix.logphimix_val_grad_batch(theta)
        self.assertAlmostEqual(a, logps[0], places=6)
        self.assertTrue(np.allclose(b, lograds[0], rtol=1e-05, atol=1e-05))
        self.assertAlmostEqual(c, logps[1], places=6)
        self.assertTrue(np.allclose(d, lograds[1], rtol=1e-05, atol=1e-05))
    
    def test_forward(self):
        theta = np.random.rand(3, theta_d)
        a, b = phimix.forward_info(theta[0])
        c, d = phimix.forward_info(theta[1])
        pmV = phimix.logphimix_batch(theta)
        # forward_info_batch returns (probabilities, transformed points); the mixture
        # density itself is available from logphimix_batch
        logps, thetaAlls = phimix.forward_info_batch(theta, False)
        self.assertEqual(pmV.shape, (theta.shape[0],))
        self.assertTrue(np.allclose(a, logps[0], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(c, logps[1], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(b, thetaAlls[0], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(d, thetaAlls[1], rtol=1e-05, atol=1e-05))
    
    def test_backward(self):
        theta = np.random.rand(3, theta_d)
        a, b, c = phimix.backward_info(theta[0])
        d, e, f = phimix.backward_info(theta[1])
        x, thetaAlls, qVals = phimix.backward_info_batch(theta)
        self.assertTrue(np.allclose(a, x[0], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(d, x[1], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(b, thetaAlls[0], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(e, thetaAlls[1], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(c, qVals[0], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(f, qVals[1], rtol=1e-05, atol=1e-05))
            
#%%
if __name__ == '__main__':
    unittest.main()
# %%

#%%
# if setID == 1:
#     import params.eg01setup as expset
#     KComp = expset.KComp
#     import params.eg01mcmc as mcmcparams
#     pi0 = mcmcparams.get_init_sampler(theta_d)

# simu_setup = expset.simuSKT(theta_d)
# targetD = simu_setup.get_targetD() 
# logq_fun = targetD.logphimix_batch
# logq_grad = targetD.loggrad

# phimix = mcmcparams.get_phimix(phimixType, expset.basepath, theta_d)
# phimix.setlogq(logq_fun)
# phimix.setlogq_grad(logq_grad)
# phimix.set_backward_rep(300)

#%%    
# nSamples = 1000
# import numpy.random as rd
# thetas = phimix.drawsample(nSamples)
# probv, thetaStars2 = phimix.forward_info_batch(thetas, False)
# thetaStars = np.zeros((nSamples, theta_d))
# for j in range(nSamples):
#     psi0 =  rd.choice(range(2), 1, False, probv[j])[0]
#     thetaStars[j] = thetaStars2[j][psi0]
# plt.hist(thetaStars[:,0], bins = 50)

# #%%
# probv_log, thetanew1, qVals  = phimix.backward_info_batch(thetaStars)
# probv_max = np.max(probv_log, axis = 1)
# probv = probv_log - probv_max[:,None]
# probv = np.exp(probv)
# probv = probv / np.sum(probv, axis = 1)[:,None]

# thetaFinal = np.zeros((nSamples, theta_d))
# for j in range(nSamples):
#     psi1 = rd.choice(range(2), 1, False, probv[j])[0]
#     thetaFinal[j] = np.copy(thetanew1[j, psi1, :])
# plt.hist(thetaFinal[:,9], bins = 50)