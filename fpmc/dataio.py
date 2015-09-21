#-*- coding: utf8
from __future__ import division, print_function

from collections import OrderedDict

import numpy as np
import pandas as pd

def save_model(out_fpath, model):
    store = pd.HDFStore(out_fpath, 'w')
    for model_key in model:
        model_val = model[model_key]
        
        if type(model_val) == np.ndarray:
            store[model_key] = pd.DataFrame(model_val)
        else:
            store[model_key] = pd.DataFrame(model_val.items(), \
                    columns=['Name', 'Id'])
    store.close()

def initialize_trace(trace_fpath, from_=0, to=np.inf):
    
    hyper2id = OrderedDict()
    obj2id = OrderedDict()
    seen = {}

    Trace = []
    dts = []
    with open(trace_fpath, 'r') as trace_file:
        for i, line in enumerate(trace_file):
            if i < from_: 
                continue

            if i >= to:
                break
            dt, hyper_str, src_str, dest_str = line.strip().split('\t')

            if hyper_str not in hyper2id:
                hyper2id[hyper_str] = len(hyper2id)
            
            if src_str not in obj2id:
                obj2id[src_str] = len(obj2id)
            
            if dest_str not in obj2id:
                obj2id[dest_str] = len(obj2id)
            
            dt = float(dt)
            h = hyper2id[hyper_str]
            s = obj2id[src_str]
            d = obj2id[dest_str]
            
            if (h, s) not in seen:
                seen[h, s] = set()
            
            seen[h, s].add(d)
            Trace.append([h, s, d])
            dts.append(dt)
    
    dts = np.asanyarray(dts, order='C')
    Trace = np.asanyarray(Trace, dtype='i4', order='C')
    return dts, Trace, seen, hyper2id, obj2id
