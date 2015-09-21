#-*- coding: utf8
from __future__ import division, print_function

from fpmc import mrr

import pandas as pd
import plac
import numpy as np

def main(model, out_fpath):
    store = pd.HDFStore(model)
    
    from_ = store['from_'][0][0]
    to = store['to'][0][0]
    assert from_ == 0
    
    trace_fpath = store['trace_fpath'][0][0]

    VUI_uk = store['VUI_uk'].values
    VIU_ok = store['VIU_ok'].values
    VLI_ok = store['VLI_ok'].values
    VIL_ok = store['VIL_ok'].values

    hyper2id = dict(store['hyper2id'].values)
    obj2id = dict(store['obj2id'].values)
    
    HSDs = []
    tstamps = []

    with open(trace_fpath) as trace_file:
        for i, l in enumerate(trace_file): 
            if i < to:
                continue

            _, h, s, d = l.strip().split('\t')
            if h in hyper2id and s in obj2id and d in obj2id:
                HSDs.append([hyper2id[h], obj2id[s], obj2id[d]])
    
    num_queries = min(10000, len(HSDs))
    queries = np.random.choice(len(HSDs), size=num_queries)
    
    HSDs = np.array(HSDs, order='C', dtype='i4')
    rrs = mrr.compute(HSDs, VUI_uk, VIU_ok, VLI_ok, VIL_ok)
    
    np.savetxt(out_fpath, rrs)
    store.close()
    
plac.call(main)
