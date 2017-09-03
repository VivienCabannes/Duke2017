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
from global_path import SMOOTH_SCALE, MOTION_SCALE, FPS, STIMULI

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
#from face_parameterization import face_parameterization_extractor
#from face_parameterization import face_asymmetry_extraction
#from data_handling import get_data, get_ids
#from signal_processing import preprocess_signal
#import matplotlib.pyplot as plt
#from global_path import STIMULI
#
#patient_id = 'A52V8149'
#stimulus = 1
#
#ids, labels = get_ids()
#
#x = scaling(MOTION_SCALE)
#filt = conv_filter(x, name='Gaussian')
#
#i=0
#patient_id = ids[i]
#label = labels[i]
#stimulus = STIMULI[2]
#landmarks, timestamps = get_data(stimulus, patient_id)
#signal = face_parameterization_extractor(landmarks, timestamps, stimulus)
