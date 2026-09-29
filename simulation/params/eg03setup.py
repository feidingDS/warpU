#%%
import numpy as np
from warpu.gaussmixdiag import GaussMixtureDiag
#%%

basepath =  "./results/eg03fits/"

KComp = 2
theta_d = 30
#%%


def get_Xd(RotMat, shift, Xbase, n_center, theta_d):
    X1 = Xbase @ RotMat + shift
    Xd = np.zeros((n_center, theta_d))
    for j in range(theta_d // 2):
        Xd[:, 2*j] = X1[:, 0]
        Xd[:, 2*j+1] = X1[:, 1]
    return Xd


class simuSshape:
    def __init__(self, theta_d):
        self.theta_d = 30 # even number
        np.random.seed(42)
        self.n_center = 16
        self.KComp = 2  
        theta1 = np.linspace(0, np.pi, self.n_center//2)
        theta2 = np.linspace(np.pi, 2 * np.pi, self.n_center//2)
        x1 = np.sin(theta1)*0.7; y1 = theta1*0.8
        x2 = np.sin(theta2)*0.7; y2 = theta2*0.8
        Xbase = np.vstack((np.hstack((x1, x2)), np.hstack((y1, y2)))).T
        RotMatBase = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)

        shift = np.array([-1, 1]) * 3
        self.X4 = np.zeros((self.n_center*self.KComp, self.theta_d))
        #self.mu = np.zeros((self.KComp, self.theta_d))
        RotMat = np.eye(2)
        for j in range(self.KComp):
            Xd = get_Xd(RotMat, shift[j], Xbase, self.n_center, self.theta_d)
            RotMat = RotMat @ RotMatBase
            self.X4[j*self.n_center:(j+1)*self.n_center] = Xd 
            #self.mu[j] = np.mean(self.X4[j*self.n_center:(j+1)*self.n_center], axis=0)

    def get_targetD(self):
        KCompTarget = self.n_center * self.KComp
        precL = np.ones(self.theta_d)*0.5
        targetD = GaussMixtureDiag(KCompTarget, self.theta_d)

        wwTarget = np.ones(KCompTarget) / KCompTarget
        for i in range(KCompTarget):
            targetD.set_parameter(i, self.X4[i], precL, wwTarget[i])
        return targetD


def get_init_params(theta_d, phimix_type, phimixID):
    '''
    Get initial parameters for variational inference
    '''
    tmp = simuSshape(theta_d)
    KComp = 2
    mu_init = np.zeros((KComp, theta_d))
    for j in range(KComp):
        mu_init[j] = np.mean(tmp.X4[j*tmp.n_center:(j+1)*tmp.n_center], axis=0)

    if phimix_type==1 and phimixID == 1:
            savecode = "gaussVIV2_"
    else:
        savecode = "gaussVIV2_K6_"
        KComp = 2*3
        mu_init_6 = np.zeros((KComp, theta_d))
        mu_init_6[:2] = mu_init.copy()
        mu_init_6[2] = tmp.X4[3]
        mu_init_6[3] = tmp.X4[12]
        mu_init_6[4] = tmp.X4[19]
        mu_init_6[5] = tmp.X4[28]
        mu_init = mu_init_6
    ww_init = np.ones(KComp) / KComp
    Prec_init = np.zeros((KComp, theta_d, theta_d))
    for psi in range(KComp):
        Prec_init[psi] = np.eye(theta_d)
    result = {"KComp": KComp, "mu": mu_init, "ww": ww_init, "Prec": Prec_init}
    return result, savecode
# %%

def get_iid_sample(theta_d, n_sample):
    simu_setup = simuSshape(theta_d)
    targetD = simu_setup.get_targetD()
    sample = targetD.draw_sample(n_sample)
    return sample