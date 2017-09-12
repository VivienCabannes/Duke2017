#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import pickle
import time
import os
import numpy as np
from global_path import SAVE_DIR

"""
--------------------------- Saving Variables Values ---------------------------
""" 
def save_variables(save_file, list_of_variables):
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    save_path = os.path.join(SAVE_DIR, save_file+'.pkl')
    output = open(save_path, 'wb')
    for variable in list_of_variables:
        pickle.dump(variable, output)
    output.close()
    
def load_variables(save_file):
    save_path = os.path.join(SAVE_DIR, save_file+'.pkl')
    pkl_file = open(save_path, 'rb')
    list_of_variables = []
    while True:
        try:
            variable = pickle.load(pkl_file)
            list_of_variables.append(variable)
        except EOFError:
            break
    pkl_file.close()
    return list_of_variables

def save_fig(fig, save_file, dpi = 90):
    save_path = os.path.join(SAVE_DIR, save_file + '.png')
    fig.savefig(save_path, dpi=dpi)

"""
------------------------- Keep track of Execution Time ------------------------
""" 
class tic_toc:
    def __init__(self, message):
        self.message = '{0:40}: '.format(message)
        print(self.message + "...\n", end='', flush=True)
        self.t0 = time.time()
    def stop(self):
        t1 = time.time()
        ending = 'Done in %.2f secondes' %(t1-self.t0)
        print(self.message + ending + '\n', end='', flush=True)
        
def pause(sec):
    top = time.time()
    while True:
        if (time.time()-top) > sec:
            break
        
"""
------------------------------- Linear Algebra --------------------------------
"""
def norm(x, p=2):
    return np.sum(np.abs(x)**p)**(1./p)

def normalization(filt, p=1):
    return filt / norm(filt, p=p)

def distance_l2 (set1, set2):
    nb1 = set1.shape[0]
    nb2 = set2.shape[0]
    nrm1 = np.matlib.repmat(np.reshape(np.sum(set1*set1,axis=1),(nb1,1)),1,nb2)
    nrm2 = np.matlib.repmat(np.sum(set2*set2, axis=1), nb1, 1 )
    return ( nrm1 + nrm2 - 2 * set1.dot(np.transpose(set2)) )

def sign(x):
    return 2*(x>0) - 1
  
def PCA(X, nb_dims = 3):
    U, s, V = np.linalg.svd(X, full_matrices=False)
    V = np.transpose(V)[:,:nb_dims]
    res = np.matmul(U[:,:nb_dims], np.diag(s[:nb_dims]))
#    res = np.matmul(X, V)
    return res, V

