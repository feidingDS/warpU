#%%
import numpy as np
import torch
from .networks import CNF
import numpy.random as rd
#from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint
import scipy.stats
from scipy.special import logsumexp

#%%

device = torch.device('cuda:' + str(0)
                       if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')
class deepphimixClass:
    def __init__(self, theta_d,  phimix):
        self.phimix = phimix
        self.KComp = phimix.get_K()
        self.theta_d = theta_d
        self.func_list = []
        self.ode_t0 = 0
        self.ode_t1 = 1
        self.ode_stepsize = 1e-3

    def setlogq(self, logq_fun_):
        self.logq_fun = logq_fun_
    
    def set_backward_rep(self, backward_rep):
        pass

    def set_ode_param(self, t0, t1, stepsize):
        self.ode_t0 = t0
        self.ode_t1 = t1
        self.ode_stepsize = stepsize

    def set_network(self, width, hidden_dim, basepath = None, stName  = None):
        self.func_list = []
        for i in range(self.KComp):
            func = CNF(in_out_dim=self.theta_d, hidden_dim=hidden_dim, width=width).to(device)
            if basepath is None:
                func.training = True   
                func.trace_compute = True
            else:
                func.eval()
                func.training = False
                func.trace_compute = True
                path = f"{basepath}func_{stName}_{self.theta_d}_{i}.pth"
                func.load_state_dict(torch.load(path, weights_only=True))
            self.func_list.append(func)
    
    def save_network(self, basepath, stName):
        for i in range(self.KComp):
            path = f"{basepath}func_{stName}_{self.theta_d}_{i}.pth"
            torch.save(self.func_list[i].state_dict(), path)

    def get_K(self):
        return self.KComp
    
    def ode_compute(self, theta, psi, T0, T1):
        '''
        work for  observations
        '''
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        
        nSample = theta.shape[0]
        timerange = torch.tensor([T0, T1]).type(torch.float32).to(device)
        with torch.no_grad():
            x = torch.tensor(theta, dtype=torch.float32).to(device)
            logp_diff_t1 = torch.zeros(nSample, 1, dtype=torch.float32).to(device)
            states = (x,logp_diff_t1)
            z_t, logp_diff_t = odeint(
                self.func_list[psi], states, timerange,
                method='rk4', options=dict(step_size=self.ode_stepsize)
            )
            z_t1 = z_t[-1].detach().cpu().numpy()
            logp_diff_t0 = logp_diff_t[-1].detach().cpu().numpy()
        return z_t1, logp_diff_t0

    def logphimix_batch(self, theta):
        return self.forward_info_batch(theta, True, True)[2]

    def backward_once(self, theta, psi, return_logphimix = False):
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        scale_psi = self.phimix.get_scale(psi).T
        Zsample, logq_diff = self.ode_compute(theta, psi, 
                                            self.ode_t1, self.ode_t0)
        Zsample = Zsample @ scale_psi + self.phimix.get_mu(psi)
        qvals_psi = self.logq_fun(Zsample)
        logq_diff = logq_diff.flatten() + self.phimix.get_logdet(psi) ##?????
        logq_x = qvals_psi + logq_diff

        probv_forward, _, logphimix_deep = self.forward_info_batch(Zsample, True, True)
        # forward prob
        logprobv_psi = probv_forward[:, psi]
        # output:
        # Zsample: the backward transformed theta
        # logq_x: the log q of the transformed target at the input theta
        # qvals_psi: target log q value at the transformed theta
        if return_logphimix:
            return Zsample, logq_x, logprobv_psi, qvals_psi, logphimix_deep
        return Zsample, logq_x, logprobv_psi, qvals_psi
    
    def warped_logq(self, theta):
        return 0

    def get_weights(self):
        return self.phimix.get_weights()
    
    
    def backward_info(self, theta):
        '''
        work for a single observation
        '''
        logprobv = np.zeros(self.KComp)
        theta_all = np.zeros((self.KComp, self.theta_d))
        for psi in range(self.KComp):
            a, b, c, qVals = self.backward_once(theta, psi)
            theta_all[psi,:] = a.flatten()
            logprobv[psi] = b.flatten()[0] + c
        #qVals = self.logq_fun(theta_all)
        #logprobv = logprobv + qVals.flatten()
        return logprobv, theta_all, qVals
    
    def backward_info_batch(self, theta):
        
        nSample = theta.shape[0]
        logprobMat = np.zeros((nSample, self.KComp))
        theta_all = np.zeros((nSample, self.KComp, self.theta_d))
        qVals = np.zeros( (nSample, self.KComp) )
        for psi in range(self.KComp):
            a, b, c, qVals_psi = self.backward_once(theta, psi)
            theta_all[:, psi, :] = a
            qVals[:, psi] =  qVals_psi
            logprobMat[:, psi] = b + c.flatten() #+ qVals[:, psi]

        return logprobMat, theta_all, qVals
    


    def forward_once(self, theta, psi):
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        #nSample = theta.shape[0]
        invScale = self.phimix.get_invScale(psi).T
        muVec = self.phimix.get_mu(psi)
        thetaC = (theta - muVec) @ invScale
        #thetaC = self.phimix.get_invScale(psi) @ (theta - self.phimix.get_mu(psi))
        z_t, logDelta = self.ode_compute(thetaC, psi, self.ode_t0, self.ode_t1)
        logDelta =  logDelta.flatten() - self.phimix.get_logdet(psi)
        return z_t, logDelta

    def forward_once_training(self, theta, psi):
        invScale = self.phimix.get_invScale(psi).T
        muVec = self.phimix.get_mu(psi)
        nSample = theta.shape[0]    
        logdet_val = self.phimix.get_logdet(psi)
        logdet_val_tensor = torch.tensor(logdet_val, dtype=torch.float32).to(device)
        invScale = torch.tensor(invScale, dtype=torch.float32).to(device)
        muVec = torch.tensor(muVec, dtype=torch.float32).to(device)
        thetaC = (theta - muVec) @ invScale
        #z_t, logDelta = self.ode_compute(thetaC, psi, self.ode_t0, self.ode_t1)

        timerange = torch.tensor([self.ode_t0, self.ode_t1]).type(torch.float32).to(device)
        
        logp_diff_t1 = torch.zeros(nSample, 1, dtype=torch.float32).to(device)
        states = (thetaC, logp_diff_t1)
        z_t, logp_diff_t = odeint(
            self.func_list[psi], states, timerange,
            method='rk4', options=dict(step_size=self.ode_stepsize)
        )
        z_t1 = z_t[-1]#.detach().cpu().numpy()
        logp_diff_t0 = logp_diff_t[-1]#.detach().cpu().numpy()


        logDelta =  logp_diff_t0.flatten() - logdet_val_tensor
        return z_t1, logDelta
    
    def draw_forward(self, theta):
        if theta.ndim == 1:
            probv, thetaRep = self.forward_info(theta)
            psi = rd.choice(range(self.KComp), 1, False, probv)[0]
            return thetaRep[psi]

        probv, thetaRep = self.forward_info_batch(theta)
        nSample = theta.shape[0]
        thetaOut = np.zeros((nSample, self.theta_d))
        psiAll = np.zeros(nSample)
        for i in range(nSample):
            psi = rd.choice(range(self.KComp), 1, False, probv[i])[0]
            thetaOut[i] = thetaRep[i, psi]    
            psiAll[i] = psi
        return thetaOut, psiAll

    def forward_info(self, theta):
        '''
        work for a single observation
        '''
        probv = np.log( self.phimix.get_weights() )
        thetaRep = np.zeros((self.KComp, self.theta_d))
        for psi in range(self.KComp):
            a, b = self.forward_once(theta, psi)
            thetaRep[psi] = a.flatten()
            logStdN = scipy.stats.norm.logpdf(thetaRep[psi]).sum()
            probv[psi] +=  logStdN + b[0]
        probv = np.exp(probv - np.max(probv)) 
        probv = probv / np.sum(probv)
        return probv, thetaRep

    def forward_info_batch(self, theta, logprov = False, return_phimix = False):
        nSample = theta.shape[0]
        probv = np.zeros((nSample, self.KComp))
        thetaRep = np.zeros((nSample, self.KComp, self.theta_d))
        wws = self.phimix.get_weights()
        wws_log = np.log(wws)
        for psi in range(self.KComp):
            z_t, logDelta2 = self.forward_once(theta, psi)
            logStdN = scipy.stats.norm.logpdf(z_t).sum(axis=1)
            probv[:, psi] = logStdN + logDelta2.flatten() 
            thetaRep[:, psi, :] = z_t
        probv = probv + wws_log
        phimixVals = logsumexp(probv, axis=1)
        if not logprov:
            probv = np.exp(probv - np.max(probv, axis=1).reshape(-1, 1))
            probv = probv / np.sum(probv, axis=1).reshape(-1, 1)

        if return_phimix:
            return probv, thetaRep, phimixVals
        return probv,  thetaRep


    def draw_backward(self, theta):

        logprobv, theta_all, _ = self.backward_info(theta)
        probvmax = np.max(logprobv)
        probv = np.exp(logprobv - probvmax)
        probv = probv/np.sum(probv)
        psi = rd.choice(range(self.KComp), 1, False, probv)[0]
        return theta_all[psi,:]

    def drawsample(self, num_samples):
        thetabase = rd.randn(num_samples, self.theta_d)
        prob, theta, qvals = self.backward_info_batch(thetabase)
        theta_output = np.zeros((num_samples, self.theta_d))
        for i in range(num_samples):
            probVec = prob[i] - np.max(prob[i])
            probVec = np.exp(probVec)
            probVec = probVec / np.sum(probVec)
            psi = rd.choice(range(self.KComp), 1, False, probVec)[0]
            theta_output[i] = theta[i, psi]
        #psi = rd.choice(range(self.KComp), num_samples, True, self.phimix.get_weights())
        #for i in range(num_samples):
        #    theta[i] = self.draw_backward(theta[i-1])
        return theta_output


# %%
