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
    cmd = 'python large_train_products.py --net NFGNN --batch_size 256000'
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
        output = subprocess.check_output(cmd, shell=True)
        val, tst, checkpt_file = output.decode().split(',')
        val = float(val)
        tst = float(tst)
        index = checkpt_file.strip()
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val*100, tst*100, cmd))
    print('>>>>>>>>>> min tst now=%5.2f%% @ %s' % (-min_tst*100, min_tst_c))
    tst_score = -tst/100
    score = -val/100
    if score < min_y:
        min_y, min_c = score, cmd
        f= open("./result/logger-{}-val.txt".format(args.dataname),"a+")
        f.write('>>>>>>>>>> min val now=%5.2f%% @ %s\n' % (-min_y*100, min_c))
        f.close()

        min_tst, min_tst_c = tst_score, cmd
        f= open("./result/logger-{}-tst.txt".format(args.dataname),"a+")
        f.write('>>>>>>>>>> min tst now=%5.2f%% @ %s\n' % (-min_tst*100, min_tst_c))
        f.close()
    else:
        os.remove(index)
    return {'loss': score, 'status': hyperopt.STATUS_OK}


parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cora',
                    help='dataset name')
parser.add_argument('--train_rate', type=float, default=0.025,
                    help='training set rate')
parser.add_argument('--val_rate', type=float, default=0.025,
                    help='validation set rate')
parser.add_argument('--RPMAX', type=int, default=10,
                    help='the number of test')
parser.add_argument('--rank', type=int, default=5,
                    help='approx rank')
parser.add_argument('--K', type=int, default=5,
                    help='approx rank')
args = parser.parse_args()


space = {'lr': hyperopt.hp.choice('lr', [0.01, 0.05, 0.001, 0.005, 0.002]),
         'pro_lr': hyperopt.hp.choice('pro_lr', [0.01, 0.05, 0.001, 0.005, 0.002]),
         
         'weight_decay': hyperopt.hp.choice('weight_decay', [0.0005, 0.00025, 0.0001, 0.001, 0.0025, 0.005, 0.0]),
         'pro_wd': hyperopt.hp.choice('pro_wd', [0.0005, 0.00025, 0.0001, 0.001, 0.0025, 0.005, 0.0]),
         
         'hidden': hyperopt.hp.choice('hidden', [64, 128, 256, 512, 1024, 2048]),
         'dprate': hyperopt.hp.quniform('dprate', 0, 0.9, 0.1),
         'dropout': hyperopt.hp.quniform('dropout', 0, 0.9, 0.1),
         'alpha': hyperopt.hp.quniform('alpha', 0, 0.9, 0.1),
         'rank': hyperopt.hp.choice('rank', [1, 3, 5, 10]),
         'dataname': args.dataname,
         'K': hyperopt.hp.choice('K', [3, 4, 5])
#          'train_rate': args.train_rate,
#          'val_rate': args.val_rate,
#          'RPMAX': args.RPMAX
        }
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
print('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))
print('>>>>>>>>>> tst=%5.2f%% @ %s' % (-min_tst * 100, min_tst_c))
