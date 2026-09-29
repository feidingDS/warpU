import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
#from scipy.spatial import distance_matrix
#import cvxpy as cp

np.import_array()
#DTYPE = np.double
#ctypedef np.double_t DTYPE_t


#ctypedef np.ndarray[np.double_t, ndim=1] ARR1D
ctypedef double[:] ARR1D
ctypedef double[:,:] ARR2D
ctypedef unsigned int uint



# draw from N(theta1, sigma^2I) and N(theta2, sigma^2I) based on maximal coupling
cpdef maximxalGaussDraw(ARR1D theta1, ARR1D theta2, double sigma):
    cdef unsigned int d = theta1.size
    cdef np.ndarray thetaStar1
    cdef np.ndarray thetaStar2
    cdef double pX, qX, WW
    thetaStar1 = theta1 + rd.normal(0, sigma, d)
    pX = np.prod(gauss.pdf(thetaStar1-theta1, scale = sigma))
    qX = np.prod(gauss.pdf(thetaStar1-theta2, scale = sigma))
    WW = rd.uniform(0, pX)
    if WW < qX:
        thetaStar2 = np.copy(thetaStar1)
        return (thetaStar1, thetaStar2, True)
    while True:
        thetaStar2 = theta2 + rd.normal(0, sigma, d)
        qY = np.prod(gauss.pdf(thetaStar2-theta2, scale = sigma))
        pY = np.prod(gauss.pdf(thetaStar2-theta1, scale = sigma))
        WW = rd.uniform(0, qY)
        if WW > pY:
            break
    return (thetaStar1, thetaStar2, False)


# Relection Maximal Coupling. See Algorithm 4 of Biswas, Jacob and Vanetti.
cpdef reflectionGaussDraw(np.ndarray   mu1, np.ndarray mu2, double sigma):
    cdef unsigned int d = mu1.size
    cdef np.ndarray Z, E, thetaStar1, thetaStar2, Xdot, Ydot
    cdef double pRatio, WW, znorm
    cdef bint flag = True

    Z = mu1 - mu2
    Z = Z/sigma
    znorm = np.linalg.norm(Z)
    if znorm > 1e-8:
        E = Z / znorm
    else:
        Z = np.zeros(d)
        E = np.zeros(d)
    
    WW = rd.uniform(0, 1)
    Xdot =  rd.normal(0, 1, d)
    Ydot = Xdot + Z

    pRatio = np.sum(gauss.logpdf(Ydot)) - np.sum(gauss.logpdf(Xdot))
    pRatio = np.min([pRatio, 0])
    pRatio = np.max([pRatio, -1e20])
    if WW > np.exp(pRatio):
        flag = False
        Ydot = Xdot - 2*np.sum(E*Xdot) * E
    thetaStar1 = sigma*Xdot + mu1
    thetaStar2 = sigma*Ydot + mu2

    return (thetaStar1, thetaStar2, flag)




# One Sample Estimate
cpdef double estimateH_oneSample(object h_fun, ARR2D theta1All, ARR2D theta2All, \
                    unsigned int k, unsigned int m, unsigned int tau):
    cdef double h_hat
    cdef unsigned int iter
    h_hat = 0.0
    for iter in range(k,m+1):
        h_hat += eval_Hl(h_fun, iter, m, tau, theta1All, theta2All)
    h_hat /= (m - k + 1)
    return h_hat

cpdef double eval_Hl(object h_fun, unsigned int l, unsigned int m, unsigned int tau, \
                    ARR2D theta1All, ARR2D theta2All):
    cdef double h_hat
    cdef unsigned int iter
    h_hat = h_fun(theta1All[l - 1,:])
    if tau <= l+1:
        return h_hat
    for iter in range(l+1, min(tau, m+1)):
        h_hat += h_fun(theta1All[iter-1,:])
        h_hat -= h_fun(theta2All[iter-1,:])
    return h_hat

# Warp-U K sample estimate
cpdef double estimateH_kS(object h_fun, double[:,:,:] theta1All, double[:,:,:] theta2All, \
                double[:,:] prob1All, double[:,:] prob2All, uint k, uint m, uint tau):
    cdef double h_hat
    cdef uint iter
    h_hat = 0.0
    for iter in range(k,m+1):
        h_hat += eval_Hl_Ks(h_fun, iter, m, tau, theta1All,\
                     theta2All, prob1All, prob2All)
    h_hat /= (m - k + 1)
    return h_hat

cpdef double eval_Hl_Ks(object h_fun, uint l, uint m, uint tau, \
                    double[:,:,:] theta1All,\
                    double[:,:,:] theta2All, \
                    double[:,:] prob1All, double[:,:] prob2All):
    cdef double h_hat = 0.0
    cdef double tmp
    cdef uint iter1, iter2, phimixK
    phimixK = prob1All.shape[1]

    for iter2 in range(0,phimixK):
        tmp = h_fun(theta1All[l - 1, iter2,:])
        h_hat += tmp * prob1All[l - 1, iter2]
    
    # l+1 <= tau -1 ---> l+2 <= tau 
    # l+2>tau ---> l+1>=tau 
    if tau <= l+1: # both tau and l are actual time points for X
        return h_hat
    for iter1 in range(l+1, min(tau, m+1)):
        for iter2 in range(0,phimixK):
            h_hat += h_fun(theta1All[iter1-1,iter2,:]) * prob1All[iter1-1,iter2]
            h_hat -= h_fun(theta2All[iter1-1,iter2,:]) * prob2All[iter1-1,iter2]
    return h_hat

   