#%%
import numpy as np
cimport numpy as np
import cython
import math  
from scipy.stats import t as tdist
np.import_array()
ctypedef unsigned int uint

from scipy.special import betainc, betaln, log1p

def stable_log_cdf_t(x, df):
    """
    Compute the numerically stable log CDF of the t-distribution.
    
    Parameters:
        x (float or np.ndarray): Value(s) at which to evaluate the log CDF.
        df (float): Degrees of freedom for the t-distribution.
    
    Returns:
        float or np.ndarray: Log CDF of the t-distribution.
    """
    x = np.asarray(x)
    
    # Degrees of freedom as parameters for the Beta function
    a = 0.5 * df
    b = 0.5
    x_transformed = df / (df + x**2)
    
    # Compute regularized incomplete beta function
    betainc_values = betainc(a, b, x_transformed)
    
    # Clamp betainc values to avoid log(0)
    betainc_values = np.maximum(betainc_values, 1e-100)
    
    # Compute log(CDF) for x < 0 and log(1 - CDF) for x >= 0
    log_cdf = np.where(
        x < 0,
        np.log(0.5 * betainc_values),  # log(CDF) for x < 0
        log1p(-0.5 * betainc_values)  # log(1 - CDF) for x >= 0
    )
    
    return log_cdf

#%%
cdef class multiskewt:
    """
    Compute the density for Multivariate skew t-distribution. 

    Adapted from the R package *sn*, function *dmst*.
    Reference: A. K. Gupta (2003) Multivariate skew t-distribution, Statistics: A Journal of Theoretical and Applied Statistics, 37:4, 359-363,
    """
    cdef np.ndarray Omega, omegavec, OmegaBar, OmegaInv
    cdef double ologdet, nu, logconst, nuplusd
    cdef uint d
    cdef np.ndarray xi, alpha, alphaScaled

    def  __init__(self, xi_,  Omega_, alpha_,  nu_):
        """
        xi_ is the shift vector
        Omega_ is the scale matrix
        alpha_ is the skew direction?
        nu_ is the degree of freedom
        """
        cdef double logconst
        self.Omega = Omega_ # covariance matrix
        self.omegavec = np.sqrt(np.diag(Omega_))
        self.OmegaBar =  self.Omega / self.omegavec # correlation matrix
        self.OmegaBar /= self.omegavec.reshape(-1,1)
        self.OmegaInv = np.linalg.inv(Omega_)
        ss, logdet = np.linalg.slogdet(Omega_)
        self.ologdet = ss * logdet

        self.xi = xi_
        self.alpha = alpha_
        self.alphaScaled = alpha_/self.omegavec

        self.nu = nu_
        self.d = alpha_.size

        logconst = math.lgamma((nu_+ self.d)/2) - math.lgamma(nu_/2) 
        logconst -=  0.5 * self.d * math.log(nu_*math.pi)
        logconst -= 0.5 *  self.ologdet
        self.logconst = logconst + math.log(2) 
        self.nuplusd = self.nu + self.d

    cpdef  logpdf(self, np.ndarray theta_):
        cdef np.ndarray X, Q, logdmt, tval, logpdf
        cdef np.ndarray theta
        if theta_.ndim == 1:
            theta = np.copy(theta_).reshape(1,-1)
        else:
            theta = np.copy(theta_)
        X = theta - self.xi
        Q = np.sum((X @ self.OmegaInv) * X, axis = 1)

        logdmt = -0.5 * self.nuplusd * np.log(1 + Q/self.nu)
        tval = self.nuplusd / (Q + self.nu)
        
        # The value of L below is controversial. The R package uses the scaled version. 
        # The paper of A. K. Gupta (2003) used the unscaled version. 
        #L = X @ self.alpha
        tval = (X @ self.alphaScaled) * np.sqrt(tval)
        logpdf = self.logconst  + logdmt 
        logpdf += stable_log_cdf_t(tval, self.nuplusd) #tdist.logcdf(tval, self.nuplusd )

        if theta_.ndim == 1:
            return logpdf[0]
        else:
            return logpdf

    cpdef np.ndarray loggrad(self, np.ndarray theta):
        """
        Compute the gradient of the log density. 
        Theta only contains one sample
        """
        cdef np.ndarray X, OmegaInvX, Part1, Part2, Part3
        cdef double xOx, Q, tval, L, logTdf, logDe

        X = theta - self.xi
        OmegaInvX = self.OmegaInv @ X #
        xOx = np.sum(OmegaInvX * X)
        Q = 1 + xOx/self.nu
        Part1 = self.nuplusd/self.nu / Q *OmegaInvX

        tval = self.nuplusd / (xOx + self.nu)        
        L = np.sum(X * self.alphaScaled) 
        #L = X @ self.alpha
        tval = L * np.sqrt(tval)        
        logTdf = tdist.logpdf(tval, self.nuplusd) - stable_log_cdf_t(tval, self.nuplusd) #tdist.logcdf(tval, self.nuplusd)
        logTdf += 0.5 * np.log(self.nuplusd)
        logDe = np.log(self.nu + xOx)

        Part2 = np.exp(logTdf - 0.5 *logDe) * self.alphaScaled
        Part3 = np.exp(logTdf - 1.5 * logDe) * L * OmegaInvX
        return (-Part1+ Part2- Part3)

    cpdef np.ndarray loggrad_batch(self, np.ndarray theta):
        """
        Compute the gradient of the log density. 
        """
        cdef np.ndarray X, OmegaInvX, Part1, Part2, Part3
        cdef np.ndarray xOx, Q, tval, L, logTdf, logDe

        X = theta - self.xi
        OmegaInvX =   X @ self.OmegaInv # n x d
        xOx = np.sum(OmegaInvX * X, axis = 1)
        Q = 1 + xOx/self.nu # n x 1
        Part1 = (self.nuplusd/self.nu / Q)[..., np.newaxis] *OmegaInvX # n x d

        tval = self.nuplusd / (xOx + self.nu)        
        L = np.sum(X * self.alphaScaled, axis = 1) # n x 1
        #L = X @ self.alpha
        tval = L * np.sqrt(tval) # n x 1

        #logTdf = tdist.logpdf(tval, self.nuplusd) - tdist.logcdf(tval, self.nuplusd)
        logTdf = tdist.logpdf(tval, self.nuplusd) - stable_log_cdf_t(tval, self.nuplusd)
        logTdf += 0.5 * np.log(self.nuplusd)
        logDe = np.log(self.nu + xOx)

        dRatio = np.exp(logTdf - 0.5 *logDe)

        Part2 =  np.outer(dRatio, self.alphaScaled)
        dRatio = np.exp(logTdf - 1.5 * logDe) * L
        Part3 = dRatio[..., np.newaxis] * OmegaInvX
        return (-Part1+ Part2- Part3)

    def moment12(self):
       oa =  self.OmegaBar @ self.alpha
       q = 1 + np.sum(self.alpha * oa)
       delta = oa / math.sqrt(q)
       fa = math.sqrt(self.nu/math.pi) * math.gamma((self.nu-1)/2)
       fa /= math.gamma(self.nu/2)
       mu = delta * fa
       fmm = self.omegavec * mu
       firstm = self.omegavec * mu + self.xi
       secondm = self.nu / (self.nu-2) * self.Omega 
       secondm += np.outer(self.xi, fmm) + np.outer(fmm, self.xi)
       secondm += np.outer(self.xi, self.xi)
       return firstm, secondm


cdef class mixtureSKT():
    cdef uint K, d
    cdef object complist, w, logw

    def __init__(self,  d_) -> None:
        self.K = 0
        self.d = d_ # dimension
        self.complist = []
        self.w = []
        self.logw = []

    cpdef appendComp(self, double w_, np.ndarray xi_,\
                np.ndarray Omega_, np.ndarray alpha_, double nu_):
        """
        Append a mixture component
        """
        self.w.append(w_)
        self.logw.append(math.log(w_))
        stobj = multiskewt(xi_,Omega_,alpha_, nu_ )
        self.complist.append(stobj)
        self.K += 1

    cpdef  logphimix_batch(self, np.ndarray theta):
        cdef uint nsample
        cdef np.ndarray loga, maxlog, rowsum, result
        if theta.ndim == 1:
            theta = theta.reshape(1,-1)
        nsample = theta.shape[0]
        loga = np.array(self.logw)
        loga = np.tile(loga, (nsample, 1))
        for iterI in range(self.K):
            loga[:,iterI] += self.complist[iterI].logpdf(theta)
        maxlog = np.max(loga,axis = 1).reshape(-1,1)
        rowsum = np.sum(np.exp(loga - maxlog), axis = 1)
        result = maxlog.flatten() + np.log(rowsum)
        if nsample == 1:
            return result[0]
        else:
            return result

    cpdef loggrad(self, np.ndarray theta):
        cdef np.ndarray loga, thetaGradMat

        loga = np.copy(self.logw)
        thetaGradMat = np.zeros((self.d, self.K))
        for iterI in range(self.K):
            loga[iterI] += self.complist[iterI].logpdf(theta)
            thetaGradMat[:,iterI] = self.complist[iterI].loggrad(theta)
        loga -= np.max(loga)
        loga = np.exp(loga)
        loga /= np.sum(loga)
        return (thetaGradMat@loga)

    cpdef logphimix_val_grad_batch(self, np.ndarray theta):
        cdef np.ndarray loga, thetaGradMat
        cdef uint nsample

        if theta.ndim == 1:
            theta = theta.reshape(1,-1)
        nsample = theta.shape[0]

        loga = np.zeros((self.K, nsample))
        loga += np.copy(self.logw)[..., np.newaxis]
        thetaGradMat = np.zeros((self.K, nsample, self.d ))
        for iterI in range(self.K):
            loga[iterI] += self.complist[iterI].logpdf(theta)
            thetaGradMat[iterI] = self.complist[iterI].loggrad_batch(theta)
        loga_max = np.max(loga, axis = 0)
        loga = np.exp(loga - loga_max)
        logp_val = np.log(np.sum(loga, axis = 0)) + loga_max

        loga /= np.sum(loga, axis = 0)
        thetaGradMat = thetaGradMat * loga[..., np.newaxis]
        thetaGradMat = np.sum(thetaGradMat, axis = 0)
        return logp_val, thetaGradMat

    def moment12(self):
        firstm = np.zeros(self.d)
        secondm = np.zeros( (self.d, self.d) )
        for iterI in range(self.K):
            t1, t2 = self.complist[iterI].moment12()
            firstm += t1 * self.w[iterI]
            secondm += t2 * self.w[iterI]
        return firstm, secondm 

