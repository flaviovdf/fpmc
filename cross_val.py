#-*- coding: utf8
from __future__ import division, print_function

from fpmc import dataio
from fpmc import learn

import argparse
import numpy as np
import pandas as pd
import os
import time

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('trace_fpath', help='The trace to learn topics from', \
            type=str)
    parser.add_argument('num_topics', help='The number of topics to learn', \
            type=int)
    parser.add_argument('model_fpath', \
            help='The name of the model file (a h5 file)', type=str)
    
    parser.add_argument('--leaveout', \
            help='The number of transitions to leave for test', type=float, \
            default=0.3)

    args = parser.parse_args()
    started = time.mktime(time.localtime())
    
    num_lines = 0
    with open(args.trace_fpath) as trace_file:
        num_lines = sum(1 for _ in trace_file)
        
    if args.leaveout > 0:
        leave_out = min(1, args.leaveout)
        if leave_out == 1:
            print('Leave out is 1 (100%), nothing todo')
            return
        from_ = 0
        to = int(num_lines - num_lines * leave_out)
    else:
        from_ = 0
        to = np.inf
 
    max_cost = float('-inf')
    best_model = None

    for rate in [0.00001, 0.0001, 0.001, 0.01]:
        for reg in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            for sigma in [0.001, 0.01, 0.1, 1.0]:
                rv = learn(args.trace_fpath, args.num_topics, rate, \
                        reg, sigma, from_, to)
                cost_val = rv['cost_val'][0]
                if cost_val > max_cost:
                    max_cost = cost_val
                    best_model = rv
                    print(max_cost)

    ended = time.mktime(time.localtime())
    best_model['training_time'] = np.array([ended - started])
    dataio.save_model(args.model_fpath, best_model)
    print('Learning took', ended - started, 'seconds')

if __name__ == '__main__':
    main()
