
#%%
import numpy as np
from warpu.skewt import mixtureSKT

#%%
basepath = "./results/eg02fits/"

#%%
#mvtnorm = importr("mvtnorm")
#MASS = importr("MASS")

KComp = 10
#%%
class simuSKT:
    def __init__(self, theta_d):
        np.random.seed(3)
        self.KComp = KComp
        self.theta_d = theta_d
        theta_d_ = 100
        self.Smat = np.zeros((KComp, theta_d, theta_d))
        self.mu = np.random.uniform(-10, 10, (KComp, theta_d_))
        self.alpha = np.random.uniform(-5, 5, (KComp, theta_d_)) 
        
        self.ww = np.random.uniform(1,5, KComp)
        self.ww /= np.sum(self.ww) 
        self.nu = 10 ################## 

        scaleFactor = np.linspace(1, 5, KComp)
        for iterK in range(KComp): 
            eigv_num = 50
            tmp = np.random.normal(0,1 , (theta_d_, eigv_num))
            tmp = tmp/np.linalg.norm(tmp, axis=0)
            eigv = np.linspace(20, 1, eigv_num) /10
            diagvec = np.random.uniform(1, 3, theta_d_)/10
            Prec1000 = np.diag(diagvec) + (tmp*eigv)@tmp.T
            Prec1000 *= scaleFactor[iterK]
            covMat = np.linalg.inv(Prec1000)
            self.Smat[iterK] = np.linalg.inv(covMat[:theta_d,:theta_d])
        self.alpha = self.alpha[:,:theta_d]
        self.mu = self.mu[:,:theta_d]
            
    def get_targetD(self):
        targetD = mixtureSKT(self.theta_d)
        for iterK in range(self.KComp):
            targetD.appendComp(self.ww[iterK], self.mu[iterK].copy(),
                                np.linalg.inv(self.Smat[iterK]).copy(), 
                               self.alpha[iterK].copy(), self.nu)
        return targetD
    
    def init_r(self):
        from rpy2.robjects.packages import importr
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()

        robjects.r.assign('K', self.KComp)
        robjects.r.assign('p', self.theta_d)
        robjects.r.assign('ww', self.ww)
        robjects.r('''
            library(sn)
            #library(Matrix)
            mu_true = list()
            K_true = K
            prop_true = ww
            tdf = rep(20, K)
            sigma2_true = list()
            alpha_true = list()
            compI = 0
            ''')
        
        for iterK in range(self.KComp): 
            r_matrix = robjects.conversion.py2rpy(self.mu[iterK])
            robjects.r.assign('muVec', r_matrix)
            r_matrix = robjects.conversion.py2rpy(self.alpha[iterK])
            robjects.r.assign('alphaVec', r_matrix)
            SKInv = np.linalg.inv(self.Smat[iterK,:,:])
            r_matrix = robjects.conversion.py2rpy(SKInv)
            robjects.r.assign('SMat', r_matrix)
            robjects.r.assign('wwI', self.ww[iterK])
            robjects.r.assign('nuI', self.nu)
            robjects.r('''
                    params = list(muVec, SMat, alphaVec, nuI)
                    compI = compI + 1
                    mu_true[[compI]] = as.numeric(muVec)
                    sigma2_true[[compI]] = SMat
                    alpha_true[[compI]] = as.numeric(alphaVec)
                    tdf[compI] = nuI
                    ''')
        
        robjects.r('''
           sampler_rmst = function(n){
                psiall = sample(c(1:K_true),n,replace=T,prob=prop_true)
                
                for (iterK in 1:K_true){
                    nK = sum(psiall == iterK)
                    if (nK > 0){
                        mu = mu_true[[iterK]]
                        omega = sigma2_true[[iterK]]
                        omega <- round((omega + t(omega)) / 2, digits = 10)
                        alpha = alpha_true[[iterK]]
                        nu = tdf[iterK]
                        X = rmst(nK, Omega=omega, xi=mu, alpha=alpha, nu=nu)
                        if (iterK == 1){
                            Xall = X
                        }else{
                            Xall = rbind(Xall, X)
                        }
                    }
                }
                return(Xall)
            }
            ''')
        
    def draw_sample(self, n):
        from rpy2.robjects.packages import importr
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri
        v = robjects.r["sampler_rmst"](n)
        return np.array(v)



# %%
def get_init_params(theta_d, phimix_type, phimixID):
    '''
    Get initial parameters for variational inference
    '''
    tmp = simuSKT(theta_d)
    mu_init = tmp.mu
    ww_init = np.ones(KComp) / KComp
    Prec_init = np.zeros((KComp, theta_d, theta_d))
    for psi in range(KComp):
        Prec_init[psi] = np.eye(theta_d)
    result = {"KComp": KComp, "mu": mu_init, "ww": ww_init, "Prec": Prec_init}
    savecode = "gaussVIV2_"
    return result, savecode
# %%
