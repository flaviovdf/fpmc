#-*- coding: utf8
# cython: boundscheck = True
# cython: cdivision = True
# cython: initializedcheck = False
# cython: nonecheck = False
# cython: wraparound = False
from __future__ import print_function, division

import numpy as np

from fpmc.myrandom.random cimport rand

cdef extern from 'math.h':
    inline double exp(double)
    inline double log(double)
    inline double abs(double)

cdef inline double sigma(double z):
    return 1.0 / (1 + exp(-z))

cdef inline double compute_dist(int u, int l, int i, \
        double[:, ::1] VUI_uk, double[:, ::1] VIU_ok, \
        double[:, ::1] VLI_ok, double[:, ::1] VIL_ok):

    cdef double d = 0.0
    cdef int k
    for k in range(VUI_uk.shape[1]):
        d += VUI_uk[u, k] * VIU_ok[i, k] + VIL_ok[i, k] * VLI_ok[l, k]
    return d

cdef inline void update(int row, double[:, ::1] X, double[::1] update):
    cdef int k
    for k in range(X.shape[1]):
        X[row, k] += update[k]

cdef void do_iter(int[:, ::1] Trace, double[:, ::1] VUI_uk, \
        double[:, ::1] VIU_ok, double[:, ::1] VLI_ok, double[:, ::1] VIL_ok, \
        double rate, double regularization):

    cdef int idx, u, l, i, j
    cdef double z, sigma_z, delta
    
    cdef double[::1] update_vui_u = np.zeros(VUI_uk.shape[1], dtype='d')
    cdef double[::1] update_viu_i = np.zeros(VUI_uk.shape[1], dtype='d')
    cdef double[::1] update_viu_j = np.zeros(VUI_uk.shape[1], dtype='d')
    
    cdef double[::1] update_vil_i = np.zeros(VUI_uk.shape[1], dtype='d')
    cdef double[::1] update_vil_j = np.zeros(VUI_uk.shape[1], dtype='d')
    cdef double[::1] update_vli_l = np.zeros(VUI_uk.shape[1], dtype='d')
    
    for idx in xrange(Trace.shape[0]):
        u = Trace[idx, 0]
        l = Trace[idx, 1]
        i = Trace[idx, 2]

        j = <int> (VIU_ok.shape[0] * rand())
        while j == l:
            j = <int> (VIU_ok.shape[0] * rand())
            
        z = compute_dist(u, l, i, VUI_uk, VIU_ok, VLI_ok, VIL_ok)
        z -= compute_dist(u, l, j, VUI_uk, VIU_ok, VLI_ok, VIL_ok)
        sigma_z = sigma(z)
        delta = 1 - sigma_z

        for k in range(VUI_uk.shape[1]):
            update_vui_u[k] = rate * \
                    (delta * (VIU_ok[i, k] - VIU_ok[j, k]) - \
                     regularization * VUI_uk[u, k])
            
            update_viu_i[k] = rate * \
                    (delta * VUI_uk[u, k] - \
                     regularization * VIU_ok[i, k])
            
            update_viu_j[k] = rate * \
                    (-delta * VUI_uk[u, k] - \
                     regularization * VIU_ok[j, k])
        
        update(u, VUI_uk, update_vui_u)
        update(i, VIU_ok, update_viu_i)
        update(j, VIU_ok, update_viu_j)

        for k in range(VUI_uk.shape[1]):
            update_vil_i[k] = rate * \
                    (delta * VLI_ok[l, k] - \
                     regularization * VIL_ok[i, k])
            
            update_vil_j[k] = rate * \
                    (-delta * VLI_ok[l, k] - \
                     regularization * VIL_ok[j, k])
            
            update_vli_l[k] = rate * \
                    (delta * (VIL_ok[i, k] - VIL_ok[j, k]) - \
                     regularization * VLI_ok[l, k])
        
        update(i, VIL_ok, update_vil_i)
        update(j, VIL_ok, update_vil_j)
        update(l, VLI_ok, update_vli_l)
             
def compute_cost(int[:, ::1] Trace, double[:, ::1] VUI_uk, \
        double[:, ::1] VIU_ok, double[:, ::1] VLI_ok, double[:, ::1] VIL_ok, \
        double rate, double regularization, int num_examples=-1, \
        int num_candidates=-1):
    
    cdef int idx, idx_j, u, l, i, j
    cdef double z, sigma_z, cost, curr_cost
    cdef dict precomputed = {}
    
    cdef int[::1] idx_trace = np.arange(Trace.shape[0], dtype='i4')
    if num_examples > 0:
        np.random.shuffle(idx_trace)
        idx_trace = idx_trace[:num_examples]
    
    cdef int[::1] candidates = np.arange(VIU_ok.shape[0], dtype='i4')
    if num_candidates > 0:
        np.random.shuffle(candidates)
        candidates = candidates[:num_candidates]
    
    cost = 0.0
    for idx in xrange(idx_trace.shape[0]):
        u = Trace[idx_trace[idx], 0]
        l = Trace[idx_trace[idx], 1]
        i = Trace[idx_trace[idx], 2]
 
        if (u, l, i) in precomputed:
            cost += precomputed[u, l, i]
            continue

        curr_cost = 0.0
        for idx_j in xrange(candidates.shape[0]):
            j = candidates[idx_j]
            z = compute_dist(u, l, i, VUI_uk, VIU_ok, VLI_ok, VIL_ok)
            z -= compute_dist(u, l, j, VUI_uk, VIU_ok, VLI_ok, VIL_ok)
            sigma_z = sigma(z)
            curr_cost += log(sigma(z))

        precomputed[u, l, i] = curr_cost
        cost += curr_cost

    return cost

def sgd(int[:, ::1] Trace, double[:, ::1] VUI_uk, double[:, ::1] VIU_ok, \
        double[:, ::1] VLI_ok, double[:, ::1] VIL_ok, \
        double rate, double regularization, int[:, ::1] Trace_val):
    
    cost_train = 0.0
    cost_val = 0.0
    i = 0
    while i < 1000:
        do_iter(Trace, VUI_uk, VIU_ok, VLI_ok, VIL_ok, rate, regularization)
        i += 1
    
    cost_train = compute_cost(Trace, VUI_uk, VIU_ok, VLI_ok, VIL_ok, \
            rate, regularization, 10000, 10000)
    cost_val = compute_cost(Trace_val, VUI_uk, VIU_ok, VLI_ok, VIL_ok, \
            rate, regularization, 10000, 10000)
    return cost_train, cost_val
