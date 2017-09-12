#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import numpy as np
from face_parameterization import FACE_PARAMETERIZATION, FACE_ASYMMETRY
from face_parameterization import face_parameterization_extractor
from face_parameterization import face_asymmetry_extraction
from signal_processing import scaling, conv_filter, convolution
from signal_processing import preprocess_signal
from global_path import SMOOTH_SCALE, MOTION_SCALE, FPS, STIMULI, LENGTHS
from global_path import RESPONSES, NB_CLUSTERS, ALL_RESPONSES
from helper import load_variables, save_variables
from models import k_means, k_mean_assign
from data_handling import homogenized_description
from signal_processing import preprocess_extractor 

"""
--------------------------------- Correlation ---------------------------------
"""
def correlation(signal, verbose = False, explanation = None):
    if verbose:
        exp = []
        for e1 in explanation:
            for e2 in explanation:
                exp.append("correlation " + e1 + " and " + e2)
        return exp
    if len(signal.shape)==1:
        signal = np.expand_dims(signal, axis=0)
        
    signal[np.isnan(signal)] = 0
    tmp = signal - np.expand_dims(np.mean(signal, axis=1), axis=1)
    cov = np.matmul(tmp, tmp.transpose()) / tmp.shape[1]

    std = np.expand_dims(np.diag(cov)**.5, axis=1)
    normalization = np.matmul(std, std.transpose())
    cor = (cov / normalization)
    for i in range(len(cor)):
        cor[i,i] = 1
    return cor.flatten()

def cross_correlation(signal, dtime=FPS, verbose = False, explanation = None):
    if verbose:
        exp = []
        for e in explanation:
            exp.append("cross-correlation " + e)
        return exp
    
    if len(signal.shape)==1:
        signal = np.expand_dims(signal, axis=0)

    signal1 = signal[:,dtime:]
    signal2 = signal[:,:-dtime]
    
    tmp = signal - np.expand_dims(np.mean(signal, axis=1), axis=1)
    tmp1 = signal1 - np.expand_dims(np.mean(signal1, axis=1), axis=1)
    tmp2 = signal2 - np.expand_dims(np.mean(signal2, axis=1), axis=1)
    
    cov12 = np.mean(tmp1 * tmp2, axis = 1)
    cov = np.mean(tmp * tmp, axis = 1)
    cor = cov12 / cov
    return cor

"""
-------------------------- Motor Behaviors Features ---------------------------
"""
def micromovements(signal, verbose = False):
    if verbose:
        explanation = []
        for f in FACE_PARAMETERIZATION:
            explanation.append("micro-movements " + f)
        return explanation
    res = None
    for i in range(len(signal)):
        feature = np.copy(signal[i])
        feature = feature - np.mean(feature)
        x = scaling(SMOOTH_SCALE)
        filt = conv_filter(x, name='Gaussian')
        smooth_feature = convolution(feature, filt)
        tmp = np.mean(np.abs(feature - smooth_feature))
        if res is None:
            res = np.copy(tmp)
        else:
            res = np.hstack((res, tmp))
    return res

def asymmetric_movements(asym_signal, verbose = False):
    if verbose:
       return FACE_ASYMMETRY
    return np.mean(np.abs(asym_signal), axis=1)

def get_motion(signal):
    res = None
    for i in range(len(signal)):
        feature = np.copy(signal[i])
        feature = feature - np.mean(feature)
        x = scaling(MOTION_SCALE)
        filt = conv_filter(x, name='Gaussian')
        tmp = convolution(feature, filt)
        if res is None:
            res = np.copy(tmp)
        else:
            res = np.vstack((res, tmp))
    return res

def non_smooth_movements(signal, verbose = False):
    if verbose:
        explanation = []
        for f in FACE_PARAMETERIZATION:
            explanation.append("non-smooth-movements " + f)
        return explanation
    motion = get_motion(signal)
    return cross_correlation(motion)

def cumulative_motion(signal, verbose = False):
    if verbose:
        explanation = []
        for f in FACE_PARAMETERIZATION:
            explanation.append("cumulative motion " + f)
        return explanation
    motion = get_motion(signal)
    return np.mean(np.abs(motion), axis=1)

def motion_variability(signal, verbose = False):
    if verbose:
        explanation = []
        for f in FACE_PARAMETERIZATION:
            explanation.append("motion variability " + f)
        return explanation
    motion = get_motion(signal)
    return np.var(motion, axis=1)

def motion_organization(signal, verbose = False):
    if verbose:
        exp = []
        for e1 in FACE_PARAMETERIZATION:
            for e2 in FACE_PARAMETERIZATION:
                exp.append("correlation " + e1 + " and " + e2)
        return exp
    motion = get_motion(signal)
    return correlation(motion)

def blink_rate(eye_signal, verbose = False):
    if verbose:
        explanation = ["blink rate"]
        return explanation
    tmp = np.zeros(len(eye_signal)) == 1
    tmp[eye_signal < .3] = True
    x = scaling(MOTION_SCALE)
    filt = conv_filter(x, name='Gaussian')
    tmp = convolution(tmp, filt)
    tmp = (tmp > 0.01).astype(np.int)
    return np.sum(tmp[1:] - tmp[:-1] > .5)

"""
-------------------------- Motor Behaviors Extractor --------------------------
"""
def motor_behaviors_extractor(landmarks, timestamps, stimulus):
    features = face_parameterization_extractor(landmarks, timestamps, stimulus)
    signal = preprocess_signal(features)
    eye_signal = signal[8]        
    features = face_asymmetry_extraction(landmarks, timestamps, stimulus)
    asym_signal = preprocess_signal(features)

    res = micromovements(signal)
    tmp = asymmetric_movements(asym_signal)
    res = np.hstack((res, tmp))
    tmp = non_smooth_movements(signal)
    res = np.hstack((res, tmp))
    tmp = cumulative_motion(signal)
    res = np.hstack((res, tmp))
    tmp = motion_variability(signal)
    res = np.hstack((res, tmp))
    tmp = motion_organization(signal) 
    res = np.hstack((res, tmp))
    tmp = blink_rate(eye_signal)
    res = np.hstack((res, tmp))

    return res

def motor_behaviors_explanation():
    exp = micromovements(None, verbose = True)
    exp += asymmetric_movements(None, verbose = True)
    exp += non_smooth_movements(None, verbose = True)
    exp += cumulative_motion(None, verbose = True)
    exp += motion_variability(None, verbose = True)
    exp += motion_organization(None, verbose = True)
    exp += blink_rate(None, verbose = True)
    return exp

MOTOR_BEHAVIORS = [ motor_behaviors_explanation() for i in STIMULI ] 

"""
------------------------------- Event Response --------------------------------
"""
def stack_all(st_data, event):
    stimulus, begin, end = event[0], int(event[2]*FPS), int(event[3]*FPS)
    for i in range(len(STIMULI)):
        if stimulus == STIMULI[i]:
            break
    all_val = None
    for ind in range(len(st_data)):
        tmp = st_data[ind,i][:, begin:end]
        """ eventually normalise per subject """
        if all_val is None:
            all_val = np.copy(tmp)
        else:
            all_val = np.hstack((all_val, tmp))
    mean = np.mean(all_val, axis=1)
    mean[np.isnan(mean)] = 0
    std = np.var(all_val, axis=1)**.5    
    std[np.isnan(std)] = 1
    all_val -= np.expand_dims(mean, axis=1)
    all_val /= np.expand_dims(std, axis=1)
    all_val[np.isnan(all_val)] = 0
    return all_val, mean, std

def perform_clustering(st_data, event):
    values, mean, std = stack_all(st_data, event)
    design = values.transpose()
    center = k_means(design, k = NB_CLUSTERS)
    return center, mean, std

def get_center(event):
    name = event[1]
    try:
        center, mean, std = load_variables(name)
    except FileNotFoundError:
        try:
            st_data, _, _, _ = load_variables('preprocess')
        except FileNotFoundError:
            extractor = preprocess_extractor
            explanation = FACE_PARAMETERIZATION
            st_data, labels, ids = homogenized_description(extractor)
            save_variables('preprocess', [st_data, labels, ids, explanation])
        center, mean, std = perform_clustering(st_data, event)
        save_variables(name, [center, mean, std])
    return center, mean, std

def event_extractor(landmarks, timestamps, stimulus):
    signal = preprocess_extractor(landmarks, timestamps, stimulus)
    res = []
    for i in range(len(RESPONSES)):
        event = RESPONSES[i]
        if event[0] == stimulus:
            begin, end = int(event[2]*FPS), int(event[3]*FPS)
            center, mean, std = get_center(event)
            tmp = signal[:, begin:end].transpose()
            tmp -= mean
            tmp /= std
            I = k_mean_assign(tmp, center)
            k = center.shape[0]
            for j in range(k):
                res.append(np.mean(I==j))
    return np.array(res)

def event_explanation(stimulus):
    exp = []
    for i in range(len(RESPONSES)):
        event = RESPONSES[i]
        if event[0] == stimulus:
            center, mean, std = get_center(event)
            k = center.shape[0]
            for j in range(k):
                exp.append(event[1] + " cluster " + str(j))
    return exp

EVENT = []
for stimulus in STIMULI:
    tmp = event_explanation(stimulus)
    EVENT.append(tmp)

"""
------------------------------- Video Response --------------------------------
"""
def video_response_extractor(landmarks, timestamps, stimulus):
    signal = preprocess_extractor(landmarks, timestamps, stimulus)
    res = []
    for i in range(len(ALL_RESPONSES)):
        event = ALL_RESPONSES[i]
        if event[0] == stimulus:
            begin, end = int(event[2]*FPS), int(event[3]*FPS)
            center, mean, std = get_center(event)
            tmp = signal[:, begin:end].transpose()
            tmp -= mean
            tmp /= std
            I = k_mean_assign(tmp, center)
            k = center.shape[0]
            for j in range(k):
                res.append(np.mean(I==j))
    return np.array(res)

def video_response_explanation(stimulus):
    exp = []
    for i in range(len(ALL_RESPONSES)):
        event = ALL_RESPONSES[i]
        if event[0] == stimulus:
            center, mean, std = get_center(event)
            k = center.shape[0]
            for j in range(k):
                exp.append(event[1] + " cluster " + str(j))
    return exp

VIDEO_RESPONSE = []
for stimulus in STIMULI:
    tmp = video_response_explanation(stimulus)
    VIDEO_RESPONSE.append(tmp)

"""
-------------------------- From Clusters to Patterns --------------------------
"""
def pattern_convolution(signal, filt):
    tmp = signal * np.expand_dims(filt, axis=1)
    return np.mean(tmp, axis=0)

def pattern_response_extractor(landmarks, timestamps, stimulus):
    signal = preprocess_extractor(landmarks, timestamps, stimulus)
    for i in range(4):
        if stimulus == STIMULI[i]:
            break
    event = ALL_RESPONSES[i]
    center, mean, std = get_center(event)
    signal -= np.expand_dims(mean, axis=1)
    signal /= np.expand_dims(std, axis=1)
    res = None
    for cur_center in center:
        tmp = pattern_convolution(signal, cur_center)
        if res is None:
            res = np.copy(tmp)
        else:
            res = np.vstack((res, tmp))
    return res

def pattern_response_explanation(stimulus):
    for i in range(len(STIMULI)):
        if stimulus == STIMULI[i]:
            break
    increment = 1. / FPS
    length = int((LENGTHS[i] + increment) * FPS) + 1

    event = ALL_RESPONSES[i]
    center, mean, std = get_center(event)

    explanation = []
    for i in range(NB_CLUSTERS):
        for pos in range(length):
            exp = 'cluster %d, frame %d' %(i, pos)
            explanation.append(exp)
            
    return explanation
        
PATTERN_RESPONSE = []
for stimulus in STIMULI:
    tmp = pattern_response_explanation(stimulus)
    PATTERN_RESPONSE.append(tmp)

"""
------------------------------- Mixing Patterns -------------------------------
"""
def patterns_response_extractor(landmarks, timestamps, stimulus):
    signal = preprocess_extractor(landmarks, timestamps, stimulus)
    res = None
    for event in RESPONSES:
        center, mean, std = get_center(event)
        signal_tmp = signal - np.expand_dims(mean, axis=1)
        signal_tmp /= np.expand_dims(std, axis=1)
        for cur_center in center:
            tmp = pattern_convolution(signal_tmp, cur_center)
            if res is None:
                res = np.copy(tmp)
            else:
                res = np.vstack((res, tmp))
    return res

def patterns_response_explanation(stimulus):
    for i in range(len(STIMULI)):
        if stimulus == STIMULI[i]:
            break
    increment = 1. / FPS
    length = int((LENGTHS[i] + increment) * FPS) + 1

    explanation = []
    for event in RESPONSES:
        center, mean, std = get_center(event)
        for i in range(NB_CLUSTERS):
            for pos in range(length):
                exp = event[1] + (', cluster %d, frame %d' %(i, pos))
                explanation.append(exp)
            
    return explanation

PATTERNS_RESPONSE = []
for stimulus in STIMULI:
    tmp = patterns_response_explanation(stimulus)
    PATTERNS_RESPONSE.append(tmp)
