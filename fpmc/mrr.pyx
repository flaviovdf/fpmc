#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False
from __future__ import division, print_function

from cython.parallel cimport prange

import numpy as np

def compute(int[:, ::1] HSDs, double[:, ::1] VUI_uk, double[:, ::1] VIU_ok, \
        double[:, ::1] VLI_ok, double[:, ::1] VIL_ok):

    cdef double[::1] aux = np.zeros(VIU_ok.shape[0], dtype='d')
    cdef double[::1] rrs = np.zeros(HSDs.shape[0], dtype='d')
    cdef int i, h, s, d, candidate_d, k
    
    for i in xrange(HSDs.shape[0]):
        h = HSDs[i, 0]
        s = HSDs[i, 1]
        d = HSDs[i, 2]
        for candidate_d in prange(VIU_ok.shape[0], schedule='static', nogil=True):
            aux[candidate_d] = 0.0

        for k in xrange(VIL_ok.shape[1]):
            for candidate_d in prange(VIU_ok.shape[0], schedule='static', nogil=True):
                aux[candidate_d] += VUI_uk[h, k] * VIU_ok[d, k] + VIL_ok[d, k] * VLI_ok[s, k] 
            
        for candidate_d in prange(VIU_ok.shape[0], schedule='static', nogil=True):
            if aux[candidate_d] >= aux[d]:
                rrs[i] += 1
        rrs[i] = 1.0 / rrs[i]

    return np.array(rrs)
