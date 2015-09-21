#-*- coding: utf8
from __future__ import print_function, division

import dataio
import numpy as np
import os

from fpmc import sgd

def learn(trace_fpath, nk, rate, regularization, sigma, \
        from_=0, to=np.inf, validation=0.1):

    _, Trace, seen, hyper2id, obj2id = \
            dataio.initialize_trace(trace_fpath, from_, to)
    no = len(obj2id)
    nh = len(hyper2id)
    
    validation_from = int(len(Trace) - len(Trace) * validation)
    print('Using first %d of %d as train, rest is validation' \
            % (validation_from, len(Trace)))

    Trace_train = Trace[:validation_from]
    
    rnd_idx = np.arange(len(Trace_train))
    np.random.shuffle(rnd_idx)
    Trace_train = np.asanyarray(Trace_train[rnd_idx], dtype='i4', order='C')
    Trace_val = np.asanyarray(Trace[validation_from:], dtype='i4', order='C')

    VUI_uk = np.random.normal(0, sigma, (nh, nk))
    VIU_ok = np.random.normal(0, sigma, (no, nk))
    VLI_ok = np.random.normal(0, sigma, (no, nk))
    VIL_ok = np.random.normal(0, sigma, (no, nk))
    
    cost_train, cost_val = sgd(Trace, VUI_uk, VIU_ok, VLI_ok, VIL_ok, \
            rate, regularization, Trace_val)

    rv = {}
    rv['num_topics'] = np.asarray([nk])
    rv['trace_fpath'] = np.asarray([os.path.abspath(trace_fpath)])
    rv['rate'] = np.asarray([rate])
    rv['regularization'] = np.asarray([regularization])
    rv['from_'] = np.asarray([from_])
    rv['to'] = np.asarray([to])
    rv['hyper2id'] = hyper2id
    rv['obj2id'] = obj2id
    rv['cost_train'] = np.asarray([cost_train])
    rv['cost_val'] = np.asarray([cost_val])
    rv['VUI_uk'] = VUI_uk
    rv['VIU_ok'] = VIU_ok
    rv['VLI_ok'] = VLI_ok
    rv['VIL_ok'] = VIL_ok
    return rv
