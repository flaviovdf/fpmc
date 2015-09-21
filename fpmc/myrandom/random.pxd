#-*- coding: utf8
# cython: boundscheck = False
# cython: cdivision = True
# cython: initializedcheck = False
# cython: wraparound = False
# cython: nonecheck = False

from __future__ import division, print_function

cdef extern from 'randomkit.h':
    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss
 
cdef class RNG:
    cdef rk_state *rng_state
    cdef void set_seed(self, unsigned long seed) nogil
    cdef double rand(self) nogil

cdef void set_seed(unsigned long seed) nogil
cdef double rand() nogil
