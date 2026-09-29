import numpy as np
cimport numpy as np
import cython
import numpy.random as rd
from scipy.stats import norm as gauss
from scipy.sparse import csr_matrix
import warpu.utils

np.import_array()
ctypedef unsigned int uint

cdef class GaussMixtureFull:
    cdef unsigned int K, d, backward_rep
    cdef np.ndarray mu, Smat, SmatInv,SmatInvSq, logdets, w, w_log
    cdef object logq_fun, logq_grad
    cdef double stN_const
    cdef list LtMat # the transpose of L, which Omega = L*L^T = Sigma^{-1}
    cdef list LMat # non-transpose version of L

    cpdef uint get_K(self)
    cpdef np.ndarray get_mu(self, uint compI)
    cpdef np.ndarray get_scale(self, uint compI)
    cpdef np.ndarray get_invScale(self, uint compI)
    cpdef double get_logdet(self, uint compI)


    cpdef void setlogq(self, logq_fun_)
    cpdef void setlogq_grad(self, logq_grad_)


    
    cpdef double warped_logq(self, np.ndarray theta)

    cpdef myGaussLogpdf(self, np.ndarray theta)

    cpdef double logphimix(self, np.ndarray theta)
    cpdef logphimix_batch(self, np.ndarray theta)

    cpdef logphimix_val_grad(self, np.ndarray theta)
    cpdef logphimix_val_grad_batch(self, np.ndarray theta)

    cpdef np.ndarray backward_trans(self, np.ndarray theta, uint psi)
    cpdef np.ndarray forward_trans(self, np.ndarray theta, uint psi)
    cpdef forward_info(self, np.ndarray theta)
    cpdef forward_info_batch(self, np.ndarray theta, uint returnLog)
    cpdef  draw_forward(self, np.ndarray theta)

    cpdef np.ndarray forward_prob_log(self, np.ndarray theta)
    
    

    cpdef backward_info(self, np.ndarray theta)
    cpdef backward_info_batch(self, np.ndarray theta)

    cpdef np.ndarray logq_slice(self, np.ndarray theta, uint psi)
    cpdef np.ndarray logq_slice_grad(self, np.ndarray theta, uint psi)

    #def drawsample(self, uint n, uint returnIndex=0)
    #def  drawsample_from(self, uint n, uint psi)