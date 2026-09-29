#%%
# import sys

# if "" in sys.path:
#     sys.path.remove("")

import unittest
import numpy as np
import warpu.GaussGammaMix as ggm
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

phimix = ggm.GaussGammaMix(KComp, theta_d)
for k in range(KComp):
    Prec = np.linalg.inv(covs[k])
    PrecL = np.linalg.cholesky(Prec)
    phimix.set_parameter(k, mu[k],PrecL, ww[k], nu/2, nu/2)

# %%

class TestMyFunction(unittest.TestCase):
    def test_values(self):
        theta = np.random.rand(5, theta_d)
        a = phimix.logphimix(theta[0])
        b = phimix.logphimix(theta[1])
        logps = phimix.logphimix_batch(theta)
        self.assertAlmostEqual(a, logps[0], places=6)
        self.assertAlmostEqual(b, logps[1], places=6)
    
    def test_valgrad(self):
        theta = np.random.rand(5, theta_d)
        a, b = phimix.logphimix_val_grad(theta[0])
        c, d = phimix.logphimix_val_grad(theta[1])
        logps, lograds = phimix.logphimix_val_grad_batch(theta)
        self.assertAlmostEqual(a, logps[0], places=6)
        self.assertTrue(np.allclose(b, lograds[0], rtol=1e-05, atol=1e-05))
        self.assertAlmostEqual(c, logps[1], places=6)
        self.assertTrue(np.allclose(d, lograds[1], rtol=1e-05, atol=1e-05))
    
    def test_single(self):
        thetas = np.random.rand(theta_d)
        p0 = multivariate_t(mu[0], covs[0], nu).logpdf(thetas)
        phimixt = ggm.GaussGammaMix(1, theta_d)
        k = 0
        Prec = np.linalg.inv(covs[k])
        PrecL = np.linalg.cholesky(Prec)
        phimixt.set_parameter(k, mu[k], PrecL, 1, nu/2, nu/2)
        p1 = phimixt.logphimix(thetas)
        self.assertAlmostEqual(p0, p1, places=7)
    
    def test_single_valgrad(self):
        theta = np.random.rand(theta_d)
        a, b = phimix.logphimix_val_grad(theta)
        theta[0] += 1e-6
        c, _ = phimix.logphimix_val_grad(theta)
        grad = (c-a)/1e-6
        self.assertAlmostEqual(grad, b[0], places=4)
    
    def test_forward(self):
        theta = np.random.rand(3, theta_d)
        # forward_info_batch returns (probabilities, transformed points)
        x, y = phimix.forward_info_batch(theta, 0)
        a,b = phimix.forward_info(theta[0])
        c,d = phimix.forward_info(theta[1])
        self.assertTrue(np.allclose(a, x[0], rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(c, x[1], rtol=1e-05, atol=1e-05))

    def test_component_info(self):
        theta = np.random.rand(3, theta_d)
        a, b, c, d = phimix.component_logprob_info_batch(theta)
        x1, y1, z1, q1 = phimix.component_logprob_info(theta[0])
        x2, y2, z2, q2 = phimix.component_logprob_info(theta[1])
        self.assertTrue(np.allclose(b[:,0], y1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(b[:,1], y2, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(c[:,0], z1, rtol=1e-05, atol=1e-05))
        self.assertTrue(np.allclose(c[:,1], z2, rtol=1e-05, atol=1e-05))
            
#%%
if __name__ == '__main__':
    unittest.main()
# %%
