#-*- coding: utf8
from __future__ import division, print_function

from fpmc import dataio
from fpmc import learn

import argparse
import numpy as np
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
            default=0)

    parser.add_argument('--sigma', \
            help='The value for the sigma variable', \
            type=float, default=0.01)
    parser.add_argument('--learning_rate', \
            help='The learning rate for the algorithm', \
            type=float, default=0.005)
    parser.add_argument('--regularization', help='The regularization', \
            type=float, default=0.03)
    
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
    
    print('Learning')
    rv = learn(args.trace_fpath, args.num_topics, args.learning_rate, \
            args.regularization, args.sigma, from_, to)
    ended = time.mktime(time.localtime())
    rv['training_time'] = np.array([ended - started])
    dataio.save_model(args.model_fpath, rv)
    print('Learning took', ended - started, 'seconds')

if __name__ == '__main__':
    main()
