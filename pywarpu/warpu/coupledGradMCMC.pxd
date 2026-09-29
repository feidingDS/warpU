import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
cimport warpu.coupledMH
import warpu.utils

np.import_array()


cdef class coupledGradMCMC(warpu.coupledMH.coupledMH):
    #cdef np.ndarray probv1Hist, probv2Hist, thetanew1Hist, thetanew2Hist
    #cdef object phimix
    #cdef unsigned int mixK
    cdef object logq_grad
    cdef double hmc_epsilon, hmc_sigma, gamma
    cdef unsigned int hmc_iterN
    cdef bint useHMC

    cpdef void setlogqgrad(self, object)

    cdef void leapfrog(self, np.ndarray, np.ndarray )

    cpdef void coupledTrans(self)
    cdef void coupledHMC(self)
    cpdef void singleTrans(self)
    cpdef void singleHMC(self)

    cdef void coupledMALA(self)
    cdef void singleMALA(self)
    ## To be updated 2
    #cpdef void run(self, unsigned int k, unsigned int m)
    #cpdef void run_classical(self, unsigned int m)
