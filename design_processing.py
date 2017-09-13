#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from global_path import STIMULI, FPS, LENGTHS
from data_handling import homogenized_description
from helper import load_variables, save_variables, norm, PCA
from face_parameterization import face_parameterization_extractor
from face_parameterization import FACE_PARAMETERIZATION
from signal_processing import preprocess_extractor 
from signal_processing import first_layer_extractor, FIRST_LAYER 
from signal_processing import second_layer_extractor, SECOND_LAYER
from signal_processing import assignation_extractor, ASSIGNATION
from signal_processing import histogram_extractor, HISTOGRAM
from behaviors_modeling import motor_behaviors_extractor, MOTOR_BEHAVIORS
from behaviors_modeling import event_extractor, EVENT
from behaviors_modeling import video_response_extractor, VIDEO_RESPONSE
from behaviors_modeling import pattern_response_extractor, PATTERN_RESPONSE
from behaviors_modeling import patterns_response_extractor, PATTERNS_RESPONSE
from models import MatchingPursuit, FeatureElimination, fast_estimation

"""
-------------------------------- Access Layers --------------------------------
"""
def get_layer(name):
    try:
        data, labels, ids, explanation = load_variables(name)
    except FileNotFoundError:
        if name == 'face_parameterization':
            extractor = face_parameterization_extractor
            explanation = FACE_PARAMETERIZATION
        elif name == 'preprocess':
            extractor = preprocess_extractor
            explanation = FACE_PARAMETERIZATION
        elif name == 'event':
            extractor = event_extractor
            explanation = EVENT
        elif name == 'video_response':
            extractor = video_response_extractor
            explanation = VIDEO_RESPONSE
        elif name == 'pattern_response':
            extractor = pattern_response_extractor
            explanation = PATTERN_RESPONSE            
        elif name == 'patterns_response':
            extractor = patterns_response_extractor
            explanation = PATTERNS_RESPONSE            
        elif name == 'first_layer':
            extractor = first_layer_extractor
            explanation = FIRST_LAYER
        elif name == 'second_layer':
            extractor = second_layer_extractor
            explanation = SECOND_LAYER
        elif name == 'assignation':
            extractor = assignation_extractor
            explanation = ASSIGNATION
        elif name == 'histogram':
            extractor = histogram_extractor
            explanation = HISTOGRAM
        elif name == "motor_behaviors":
            extractor = motor_behaviors_extractor
            explanation = MOTOR_BEHAVIORS
        else:
            print(name, 'not implemented yet')
            raise ValueError
        data, labels, ids = homogenized_description(extractor)
        save_variables(name, [data, labels, ids, explanation])
    return data, labels, ids, explanation

"""
---------------------------- Cast as Design Matrix ----------------------------
"""
def expand_explanation(explanations):
    explanation = []
    if len(explanations)==4 and type(explanations[0])==list:
        for i in range(len(STIMULI)):
            stimulus = STIMULI[i]
            tmp = 'Stimulus %s: ' %stimulus
            for exp in explanations[i]:
                explanation.append(tmp + exp)
    else:
        for i in range(len(STIMULI)):
            stimulus = STIMULI[i]
            increment = 1. / FPS
            length = int((LENGTHS[i] + increment) * FPS) + 1
            for exp in explanations:
                for pos in range(length):
                    tmp = 'Stimulus %s, position %.1f: ' %(stimulus, pos/FPS)
                    explanation.append(tmp + exp)
    return explanation

def cast_design(st_data, explanation = None):
    if np.ndim(st_data[0,0]) == 2:
        for i in range(st_data.shape[0]):
            for j in range(st_data.shape[1]):
                st_data[i,j] = st_data[i,j].flatten()

    design = np.vstack(st_data[:,0])
    for i in range(1,4):
        tmp = np.vstack(st_data[:,i])
        design = np.hstack((design, tmp))
    design[design==np.inf] = 0
    design[design==-np.inf] = 0
    design[np.isnan(design)] = 0
    if not explanation is None:
        exp = expand_explanation(explanation)
        return design, exp
    return design
    
"""
-------------------------------- Access Design --------------------------------
"""
def get_design(name):
    save_name = 'design_' + name    
    try:
        design, labels, ids, explanation = load_variables(save_name)
    except FileNotFoundError:
        st_data, labels, ids, explanation = get_layer(name)
        design, explanation = cast_design(st_data, explanation)
        save_variables(save_name, [design, labels, ids, explanation])
    return design, labels, ids, np.array(explanation)

"""
------------------------------ Features Reduction -----------------------------
"""        
def ind_sort_design(design, labels):
    ind = np.argsort(design, axis=0)
    order = labels[ind] == 1
    extremity = int(design.shape[0]/10)
    pos_begin = np.mean(order[:extremity], axis = 0)
    pos_end = np.mean(order[-extremity:], axis = 0)
    tmp = pos_begin > pos_end
    score = pos_end
    score[tmp] = pos_begin[tmp]
    ind = np.argsort(score)[::-1]
    return ind

def fast_ind_sort_selection(design, labels, thres = .21):
    ind = np.argsort(design, axis=0)
    order = labels[ind] == 1
    extremity = int(design.shape[0]/10)
    pos_begin = np.mean(order[:extremity], axis = 0)
    pos_end = np.mean(order[-extremity:], axis = 0)
    tmp = pos_begin > pos_end
    score = pos_end
    score[tmp] = pos_begin[tmp]
    score = np.abs(score - np.median(score))
    ind = np.argsort(score)[::-1]
    cut = len(ind) - np.sum(score < thres) 
    ind = ind[:cut]
    return ind

def ind_linear_independent(sorted_design, cor_thres = .5, reduct = 1):
    X = np.copy(sorted_design)
    X -= np.mean(X, axis=0)
    X /= np.sqrt(np.sum(X**2, axis=0))
    family, ind, i = [X[:,0]], [0], 1
    dim, nb_features = X.shape[0], X.shape[1]
    ind_extractor = np.zeros(nb_features) == 1
    ind_extractor[0] = True
    while i < nb_features and len(family) < dim-reduct:
        if i % 5000 == 0:
            print(i, len(family))
        cur = X[:,i]
        add = True
        """ Avoid to much correlation with prior features """
        for feature_ind in ind:
            cor = np.abs(np.sum(cur*X[:,feature_ind]))
            if cor > cor_thres:
                add = False
                break
        if not add:
            i += 1
        else:
            """ Gram_Schimdt """
            for f in family:
                scp = np.sum(cur*f)
                cur = cur - scp * f
            
            """ Check Linear Dependency """
            tmp = norm(cur)
            if tmp < 10**(-4):
                i += 1
            else:
                cur = cur / tmp
                ind.append(i)
                family.append(cur)
                ind_extractor[i] = True
                i += 1
    return ind_extractor

def ind_selection(design, labels, cor_thres = .5, reduct = 1):
    ind_1 = ind_sort_design(design, labels)
    ind_2 = ind_linear_independent(design[:,ind_1], cor_thres = cor_thres,
                                   reduct = reduct)
    ind = ind_1[ind_2]
    return ind

def fast_ind_selection(design, labels, N = 1000):
    ind = ind_sort_design(design, labels)
    return ind[:N]

def preprocess_design(design, labels, explanation = None, cor_thres = .5,
                      reduct = 1):
    ind = ind_selection(design, labels, cor_thres = cor_thres, reduct = reduct)
    if not explanation is None:
        return design[:,ind], np.array(explanation)[ind]
    return design[:,ind]

def get_preprocess_design(name):
    save_name = 'preprocess_design_' + name
    try:
        design, labels, ids, explanation = load_variables(save_name)
    except FileNotFoundError:
        design, labels, ids, explanation = get_design(name)
        design, explanation = preprocess_design(design, labels, explanation)
        save_variables(save_name, [design, labels, ids, explanation])
    return design, labels, ids, np.array(explanation)

def small_design(design, labels, explanation = None):
    ind = fast_ind_sort_selection(design, labels)
    if not explanation is None:
        return design[:,ind], np.array(explanation)[ind]
    return design[:,ind]

def get_small_design(name):
    save_name = 'small_design_' + name
    try:
        design, labels, ids, explanation = load_variables(save_name)
    except FileNotFoundError:
        design, labels, ids, explanation = get_design(name)
        design, explanation = small_design(design, labels, explanation)
        save_variables(save_name, [design, labels, ids, explanation])
    return design, labels, ids, np.array(explanation)

"""
------------------------------ Likelihood Design ------------------------------
"""
def proba_design(design):
    res = np.zeros(design.shape)
    for i in range(design.shape[1]):
        tmp = design[:, i]
        res[:,i] = fast_estimation(tmp, tmp, smoothing = False)
    return res

def get_proba_design(name):
    save_name = 'proba_design_' + name
    try:
        design, labels, ids, explanation = load_variables(save_name)
    except FileNotFoundError:
        design, labels, ids, explanation = get_design(name)
        design = normalize_design(design)
        design = proba_design(design)
        save_variables(save_name, [design, labels, ids, explanation])
    return design, labels, ids, np.array(explanation)

"""
-------------------------------- Concatenation --------------------------------
"""        
def concatenate_design(designs, explanations):
    design = None
    explanation = []
    for i in range(len(designs)):
        if design is None:
            design = np.copy(designs[i])
        else:
            design = np.hstack((design, designs[i]))
        explanation += list(explanations[i])
    return design, np.array(explanation)

def get_standard_design(preprocess = False, proba = False, small = False):
    save_name = 'st_design'
    if proba:
        if preprocess:
            save_name = 'st_design_proba'
        elif small:
            save_name = 'small_st_design_proba'
        else:
            save_name = 'st_design_all_proba'
        try:
            design, labels, ids, explanation = load_variables(save_name)
        except FileNotFoundError:
            design, labels, ids, explanation = get_standard_design(
                    preprocess = preprocess, proba = False, small = small)
            design = proba_design(design)
            save_variables(save_name, [design, labels, ids, explanation])
    else:
        if preprocess:
            save_name = 'st_design'
            try:
                design, labels, ids, explanation = load_variables(save_name)
            except FileNotFoundError:
                design, labels, ids, explanation = get_standard_design(
                        preprocess = False, proba = False, small = False)
                design,explanation=preprocess_design(design,labels,explanation)
                save_variables(save_name, [design, labels, ids, explanation])            
        elif small:
            save_name = 'small_st_design'
            try:
                design, labels, ids, explanation = load_variables(save_name)
            except FileNotFoundError:
                design, labels, ids, explanation = get_standard_design(
                        preprocess = False, proba = False, small = False)
                design, explanation = small_design(design, labels, explanation)
                save_variables(save_name, [design, labels, ids, explanation])
        else:
            save_name = 'st_design_all'
            try:
                design, labels, ids, explanation = load_variables(save_name)
            except FileNotFoundError:
                names = ['second_layer', 'motor_behaviors']
                designs, explanations = [], []
                for name in names:
                    design, labels, ids, explanation = get_design(name)
                    designs.append(design)
                    explanations.append(explanation)
                design, explanation = concatenate_design(designs, explanations)
                save_variables(save_name, [design, labels, ids, explanation])
    return design, labels, ids, explanation

"""
----------------------------------- Helper ------------------------------------
"""
def weight_labels(labels):
    weighted_labels = np.zeros(labels.shape) 
    weighted_labels[labels==-1] = -2*np.mean(labels==1)
    weighted_labels[labels==1] = 2*np.mean(labels==-1)
    return weighted_labels

def add_offset(design):
    return np.hstack((design, np.ones((design.shape[0],1)))) 

def normalize_design(design):
    X = np.copy(design)
    X[np.isnan(design)] = 0
    X -= np.mean(X, axis=0)
    X /= np.sqrt(np.var(X, axis=0))
    ind = np.sum(np.isnan(X), axis=0) == 0
    X = X[:,ind]
    return X

def show_design(design, labels, explanation = None, met = 'random'):
    label = ['x', 'y', 'z']
    if met == 'Mp':
        classifier = MatchingPursuit(M = 3)
        classifier.fit(design, labels, balanced = True)
        if not explanation is None:
            label = np.array(explanation)[classifier.I]
        descriptors = design[:,classifier.I]
    elif met == 'Pca':
        descriptors, _ = PCA(design)
        label = ['component 1','component 2','component 3']
    elif met == 'Rfe':
        classifier = FeatureElimination(nb = 3)
        classifier.fit(design, labels)
        if not explanation is None:
            label = np.array(explanation)[classifier.I]
        descriptors = design[:,classifier.I]
    else:
        met = 'random'
        ind = np.random.permutation(design.shape[1])[:3]
        descriptors = design[:,ind]
        if not explanation is None:
            label = np.array(explanation)[ind]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = descriptors[:,0], descriptors[:,1], descriptors[:,2]
    ind = labels==1
    l1 = ax.scatter(x[ind], y[ind], z[ind], c='b', marker='o')
    ind = labels==-1
    l2 = ax.scatter(x[ind], y[ind], z[ind], c='r', marker='^')
    
    ax.set_xlabel(label[0], fontsize=10)
    ax.set_ylabel(label[1], fontsize=10)
    ax.set_zlabel(label[2], fontsize=10)
    ax.set_xticks([])    
    ax.set_yticks([])    
    ax.set_zticks([])    
    ax.set_title('Simple Features ('+met+')')
    ax.legend([l1,l2], ['autistic', 'normal'])
    return fig, ax

"""
------------------------ Compute all data with Savings ------------------------
"""
def compute_all_data():
    names = ['face_parameterization', 'preprocess', 'event', 'video_response',
             'pattern_response', 'patterns_response', 'first_layer',
             'second_layer', 'histogram', 'motor_behaviors']
    for name in names:
        print(name)
        get_preprocess_design(name)
        get_proba_design(name)
    print('standard_design')
    for proba in [False, True]:
        for preprocess in [False, True]:
            get_standard_design(preprocess=preprocess, proba=proba)

        
if __name__=='__main__':
    compute_all_data()
