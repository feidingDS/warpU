import cython
import numpy as np
cimport numpy as np

np.import_array()
#DTYPE = np.double
#ctypedef np.double_t DTYPE_t

ctypedef unsigned int uint


cdef class coupledMH:
    # mFinal: the true value m if time out. 
    cdef unsigned int d, k, m, mFinal, mcmcIter
    cdef long int tau
    cdef double sigma, logq1, logq2 # inverse temperature for parallel tempering
    cdef np.ndarray acceptCount1, acceptCount2
    cdef np.ndarray theta1, theta2
    cdef np.ndarray theta1All, theta2All, timecost
    cdef double timeOut

    cdef bint flag, verbose
    cdef object pi0 #initial sampling distribution
    cdef object logq_fun #log density of the target 
    cpdef void setpi0(self, object)
    cpdef void setlogq(self, object)
    cpdef void settimeOut(self, double)
    cpdef void set_verbose(self, bint v_)
    cpdef unsigned int gettau(self)
    cpdef void singleMH(self)
    cpdef void singleTrans(self)
    cpdef void coupledTrans(self)
    cpdef void run(self, unsigned int k, unsigned int m)
    cpdef void run_classical(self, unsigned int m)
    cpdef void coupledMH(self)
    cpdef void singleMH(self)
    cpdef void classical_prepare(self, unsigned int m)

    cpdef double pt_proceed0(self)
    cpdef void swap_update0(self, np.ndarray theta_, double logq_)
    #cpdef double eval(self, object)
    #def __init__(self, int d, double sigma)

    

    


 
