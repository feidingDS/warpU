#%%
import numpy as np
import torch
from torchdiffeq import odeint
from neuralode.deepphimix import deepphimixClass

#%%

device = torch.device('cuda:' + str(0)
                       if torch.cuda.is_available() else 'cpu')


class deepphimixClass_v2(deepphimixClass):
    # the second version of the class
    # the forward map is selected by Gaussian Mixture
    def __init__(self, theta_d,  phimix):
        super().__init__(theta_d, phimix)



    def logphimix_batch(self, theta):
        return self.phimix.logphimix_batch(theta)



    def forward_info(self, theta):
        '''
        work for a single observation
        '''
        probv, _ = self.phimix.forward_info(theta)
        thetaRep = np.zeros((self.KComp, self.theta_d))
        for psi in range(self.KComp):
            self.func_list[psi].trace_compute = False
            a, _ = self.forward_once(theta, psi)
            thetaRep[psi] = a.flatten()
            self.func_list[psi].trace_compute = True
        return probv, thetaRep

    def forward_info_batch(self, theta, logprov = False, return_phimix = False):
        probv, _ = self.phimix.forward_info_batch(theta, logprov)
        nSample = theta.shape[0]
        thetaRep = np.zeros((nSample, self.KComp, self.theta_d))
        for psi in range(self.KComp):
            self.func_list[psi].trace_compute = False
            z_t, _ = self.forward_once(theta, psi)
            thetaRep[:, psi, :] = z_t
            self.func_list[psi].trace_compute = True
        if return_phimix:
            return probv, thetaRep, None
        return probv,  thetaRep

    def forward_once(self, theta, psi):
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        invScale = self.phimix.get_invScale(psi).T
        muVec = self.phimix.get_mu(psi)
        thetaC = (theta - muVec) @ invScale
        z_t, _ = self.ode_compute(thetaC, psi, self.ode_t0, self.ode_t1)
        #logDelta =  logDelta.flatten() - self.phimix.get_logdet(psi)
        return z_t, None

    def drawsample(self, num_samples):
        pass


# %%
