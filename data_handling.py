#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import os
import scipy.io as sio
import numpy as np
import pandas as pd
from global_path import DATA_DIR, STIMULI

"""
-------------------------------- Read file data -------------------------------
"""
def read_mat_file(stimulus, patient_id, data_dir = DATA_DIR):
    save_file  = 'Stimuli'+str(stimulus)+'__'+patient_id+'.mat'
    file_path = os.path.join(data_dir, save_file)
    vid_data = sio.loadmat(file_path)['vid_data']
    values = vid_data.item()
    field = vid_data.dtype.names
    dict_list = [(field[i], values[i]) for i in range(len(field))]
    dict_list.append(('id', patient_id))
    return dict(dict_list)

def read_MCHAT(data_dir = DATA_DIR):
    file_path = os.path.join(data_dir, 'questions.xlsx')
    df = pd.read_excel(file_path, index_col=None)
    ids = df['Individual'].values
    X = (df.values[:,2:22] == 'fail') + 0
    y = df['Asd'].values
    neg, pos, mid, unk = y=='no', y=='yes', y=='suspected', y=='unknown'
    y[pos], y[neg], y[mid], y[unk] = 1, -1, 0, np.nan
    mchat_scoring = df.values[:,22:]
    return ids, X, y, mchat_scoring

"""
------------------------- Split training and testing --------------------------
"""
def split(data_dir = DATA_DIR, percentage = .3):
    # Read files
    ids, X, y, mchat_scoring = read_MCHAT(data_dir)
    pos, neg = ids[y==1], ids[y==-1]
    # Number to remove
    remove_pos = int(percentage * pos.shape[0])
    remove_neg = int(percentage * neg.shape[0])
    # Split ids
    test_pos = np.random.permutation(pos)[:remove_pos]
    test_neg = np.random.permutation(neg)[:remove_neg]
    # Write it in a file, this is not really the best way to do it
    test, test_y = list(), list()
    train, train_y = list(), list()
    for i in range(len(ids)):
        patient_id = ids[i]
        if patient_id in test_pos or patient_id in test_neg:
            test.append(patient_id)
            test_y.append(y[i])
        else:   
            train.append(patient_id)
            train_y.append(y[i])
    test_tmp = np.transpose(np.vstack((test, test_y)))
    testFrame = pd.DataFrame(test_tmp, columns=['Id','Output'])
    train_tmp = np.transpose(np.vstack((train, train_y)))
    trainFrame = pd.DataFrame(train_tmp, columns=['Id','Output'])
    testFrame.to_csv(os.path.join(data_dir, 'test_ids.csv'), index=None)
    trainFrame.to_csv(os.path.join(data_dir, 'train_ids.csv'), index=None)

"""
--------------------------------- Access data ---------------------------------
"""
def get_ids(data_dir = DATA_DIR):
    ids, _, labels, _ = read_MCHAT(data_dir=data_dir)
    return ids, labels

def keep_label(ids, labels, keep="good", verbose = False):
    if keep == "pos":
        ind = labels==1
    elif keep == "neg":
        ind = labels==-1
    elif keep == "good":
        ind = np.logical_or(labels==-1, labels==1)
    if verbose:
        return ids[ind], labels[ind].astype(np.int), ind
    else:
        return ids[ind], labels[ind].astype(np.int)
    
def get_data(stimulus, patient_id, data_dir = DATA_DIR):
    dictionary = read_mat_file(stimulus, patient_id, data_dir = data_dir)
    detected = dictionary.get('face_detection').flatten() == 1
    timestamps = dictionary.get('timestamps').flatten()
    timestamps = timestamps[detected]
    landmarks = dictionary.get('original_landmarks')
    landmarks = landmarks[:, detected]
    # Correct potential overflow
    timestamps = timestamps.astype(np.float32)
    landmarks = landmarks.astype(np.float32)
    return landmarks, timestamps

"""
------------------------------- Access all data -------------------------------
""" 
def compute_all(features_extractor, ids):
    nb_ids, nb_stimulis = len(ids), len(STIMULI)
    values = np.zeros((nb_ids, nb_stimulis)).astype(np.dtype(object))
    ind = np.zeros(nb_ids)==0
    for i in range(nb_ids):
        patient_id = ids[i]
        for j in range(nb_stimulis):
            stimulus = STIMULI[j]
            try:
                landmarks, timestamps = get_data(stimulus, patient_id)
                features = features_extractor(landmarks, timestamps, stimulus)
                values[i,j] = features
            except FileNotFoundError:
                ind[i] = False
                print("Patient %s stimuli %d not found" %(patient_id,stimulus))
                continue    
    return values[ind,:], ind
    
def concatenate_all(values): 
    all_val = np.hstack(values[:,0])
    for i in range(1,4):
        tmp = np.hstack(values[:,i])
        all_val = np.hstack((all_val, tmp))
    all_val[np.isnan(all_val)] = 0
    all_val[all_val==np.inf] = 0
    all_val[all_val==-np.inf] = 0
    return all_val

def homogenized_description(extractor, keep='good',
                            homogenize=False, keep_norm='neg'):
    all_ids, all_labels = get_ids()
    ids, labels = keep_label(all_ids, all_labels, keep=keep)
    values, ind = compute_all(extractor, ids)
    ids, labels = ids[ind], labels[ind]
    
    if homogenize:
        _, _, ind = keep_label(ids, labels, keep=keep_norm, verbose=True)
        val = values[ind, :]
        all_values = concatenate_all(val)
        mean = np.expand_dims(np.mean(all_values, axis=1), axis=1)
        std = np.expand_dims(np.sqrt(np.var(all_values, axis=1)), axis=1)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                tmp = values[i,j] 
                tmp -= mean
                tmp /= std
                tmp[np.isnan(tmp)] = 0
                values[i,j] = tmp
    return values, labels, ids