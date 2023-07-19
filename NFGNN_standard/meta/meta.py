#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
#
import numpy as np
import subprocess
import argparse
import hyperopt

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

min_y = 0
min_c = None
min_tst= 0
min_tst_c = None


def trial(hyperpm):
    global min_y, min_c, min_tst, min_tst_c
    # Plz set nbsz manually. Maybe a larger value if you have a large memory.
    cmd = 'python train_model.py --net NFGNN'
    #cmd = 'CUDA_VISIBLE_DEVICES=7 ' + cmd
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        
        if isinstance(v, str):
            cmd += ' %s' %v
        elif int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
    try:
        val, tst, tst_2, tst_3 = eval(subprocess.check_output(cmd, shell=True))
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val*100, tst*100, cmd))
    print('>>>>>>>>>> min tst now=%5.2f%% @ %s' % (-min_tst*100, min_tst_c))
    tst_score = -tst/100
    score = -val/100
    if score < min_y:
        min_y, min_c = score, cmd
        f= open("./result/logger-{}-2.txt".format(args.dataset),"a+")
        f.write('>>>>>>>>>> min val now=%5.2f%% @ %s\n' % (-min_y*100, min_c))
        f.close()

        min_tst, min_tst_c = tst_score, cmd
        f= open("./result/logger-{}.txt".format(args.dataset),"a+")
        f.write('>>>>>>>>>> min tst now=%5.2f%% tst now=%5.2f%% tst now=%5.2f%% @ %s\n' % (-min_tst*100, tst_2*100, tst_3*100, min_tst_c))
        f.close()
    return {'loss': score, 'status': hyperopt.STATUS_OK}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    help='dataset name')
parser.add_argument('--train_rate', type=float, default=0.025,
                    help='training set rate')
parser.add_argument('--val_rate', type=float, default=0.025,
                    help='validation set rate')
parser.add_argument('--RPMAX', type=int, default=10,
                    help='the number of test')
args = parser.parse_args()


space = {'lr': hyperopt.hp.choice('lr', [0.01, 0.05, 0.001, 0.005]),
         'prop_lr': hyperopt.hp.choice('prop_lr', [0.01, 0.05, 0.001, 0.005]),
         'weight_decay': hyperopt.hp.choice('weight_decay', [0.0005, 0.0001, 0.001, 0.005, 0.0]),
         'prop_wd': hyperopt.hp.choice('prop_wd', [0.0005,  0.0001, 0.001,  0.005, 0.0]),
         'hidden': hyperopt.hp.choice('hidden', [16, 32, 64, 128,256]),
         'dprate': hyperopt.hp.quniform('dprate', 0, 0.9, 0.1),
         'alpha': hyperopt.hp.quniform('alpha', 0, 0.9, 0.1),
         'rank': hyperopt.hp.choice('rank', [1, 3, 5, 7, 10]),
         'K': hyperopt.hp.choice('K', [1, 3, 5, 7, 10]),
         'dataset': args.dataset,
         'train_rate': args.train_rate,
         'val_rate': args.val_rate,
         'RPMAX': args.RPMAX}
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=2000)
print('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))
print('>>>>>>>>>> tst=%5.2f%% @ %s' % (-min_tst * 100, min_tst_c))