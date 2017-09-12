#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import numpy as np
from scipy.signal import fftconvolve
from global_path import FPS, NB_BINS, STIMULI, LENGTHS, ONE_HOT, POOLING
from global_path import FIRST_SCALES, SECOND_SCALES
from helper import normalization, load_variables, save_variables
from data_handling import concatenate_all, homogenized_description
from face_parameterization import FACE_PARAMETERIZATION
from face_parameterization import face_parameterization_extractor

"""
-------------------------------- Preprocessing --------------------------------
"""    
def preprocess_signal(features):
    H = np.invert(np.isnan(features))
    H = np.mean(H, axis=0) == 1
    signal = np.copy(features)
    
    signal[:, H==0] = 0
    tmp = np.argmax(H)
    if tmp > 0:
        signal[:, :tmp] = np.matlib.repmat(signal[:, tmp:tmp+1], 1, tmp)
    tmp = np.argmax(H[::-1])
    if tmp > 0:
        signal[:,-tmp:] = np.matlib.repmat(signal[:,-tmp-1:-tmp], 1, tmp) 
    last, need = 0, False
    for current in range(signal.shape[1]):
        if H[current]:
            if need:
                size = current-last
                a, b = signal[:,last], signal[:,current]
                for i in range(len(signal)):
                    signal[i,last:current] = np.linspace(a[i], b[i], num=size)
            need = False
            last = current
        else:
            need = True
    return signal

def preprocess_extractor(landmarks, timestamps, stimulus):
    features = face_parameterization_extractor(landmarks, timestamps, stimulus)
    signal = preprocess_signal(features)
    return signal

"""
----------------------------- Convolutional Layer -----------------------------
"""    
def convolution(signal, filt):
    return fftconvolve(signal, filt, mode = 'same')

def conv_filter(x, name = 'Gaussian', complete = True, p = 1):
    if complete:
        b = 10
    else:
        b = 1.25
    if name == 'Gaussian':
        filt = np.exp( - (x*b)**2 / 2 )
    elif name == 'derivative':
        filt = (x*b) * np.exp( - (x*b)**2 / 2 )
    elif name == 'averaging':
        b = 1.25
        filt = np.exp( - (x*b)**2 / 2 )
    else:
        print(name, 'Filter not implemented')
        raise ValueError
    return normalization(filt, p = p)

def scaling(scale, fps = FPS):
    nb_points = scale*fps
    return np.linspace(-1, 1, num = 2*(nb_points//2)+1)

"""
-------------------------------- Non Linearity --------------------------------
"""
def local_averaging(feature, scale, name = 'mean'):
    x = scaling(scale)
    filt = conv_filter(x, name='averaging')
    if name == 'mean':
        res = convolution(feature, filt)
    elif name == 'var':
        mean = local_averaging(feature, scale, name = 'mean')
        res = convolution(feature**2, filt) - mean**2
    elif name == 'cumul':
        res = convolution(np.abs(feature), filt)
    else:
        print(name, 'Filter not implemented')
        raise ValueError
    return res

def subsampling(features, scale = 1):
    rate = int(scale*FPS)
    if np.ndim(features)==1:
        return features[::rate]
    return features[:,::rate]

def max_pooling(x, axis = 1):
    return np.max(x, axis = axis)

def mean_pooling(x, axis = 1):
    return np.mean(x, axis = axis)

def var_pooling(x, axis = 1):
    return np.var(x, axis = axis)

def pooling_func(x, scale, func = max_pooling, windows = None):
    if windows is None:
        windows = []
        if np.ndim(x)==1:
            length = len(x)
        else:
            length = x.shape[1]
        increment = max(int(scale*FPS), 1)
        for i in range(max(int(2*length / increment)-1, 1)):
            begin = int(i*increment/2)
            end = min(begin + increment, length)
            windows.append([begin, end])
    if np.ndim(x)==1:
        res = []
        for window in windows:
            begin, end = window[0], window[1]
            res.append(func(x[begin:end], axis = 0))
        return np.array(res)
    else:
        res = None
        for window in windows:
            begin, end = window[0], window[1]
            tmp = func(x[:,begin:end], axis = 1)
            if res is None:
                res = np.copy(tmp)
            else:
                res = np.vstack(res, tmp)
        res = res.transpose()
        return res

"""
-------------------------------- Architecture ---------------------------------
"""
ATOMS = [['Gaussian', FIRST_SCALES], ['derivative', FIRST_SCALES]]
NON_LINEARITY = [['mean', SECOND_SCALES], ['var', SECOND_SCALES], 
                 ['cumul', SECOND_SCALES]]
#NON_LINEARITY = [['cumul', SECOND_SCALES]]

"""
--------------------------------- First Layer ---------------------------------
"""
def first_layer_transform(signal):
    res = None  
    for i in range(len(signal)):
        feature = signal[i]
        for atom in ATOMS:
            name, scales = atom[0], atom[1]
            for scale in scales:
                x = scaling(scale)
                filt = conv_filter(x, name=name)
                tmp = convolution(feature, filt)
                if res is None:
                    res = np.copy(tmp)
                else:
                    res = np.vstack((res, tmp))
    return res

def first_layer_extractor(landmarks, timestamps, stimulus):
    signal = preprocess_extractor(landmarks, timestamps, stimulus)
    res = first_layer_transform(signal)
    return res

def first_layer_explanation():
    explanation = []
    for feature in FACE_PARAMETERIZATION:
        for atom in ATOMS:
            name, scales = atom[0], atom[1]
            for scale in scales:
                exp = feature + ' ' + name + (' scale %.1f' %scale) 
                explanation.append(exp)
    return explanation
    
FIRST_LAYER = first_layer_explanation()

"""
-------------------------------- Second Layer ---------------------------------
"""
def second_layer_transform(signal, pooling = POOLING):
    res = None  
    for i in range(len(signal)):
        feature = signal[i]
        for atom in NON_LINEARITY:
            name, scales = atom[0], atom[1]
            for scale in scales:
                tmp = local_averaging(feature, scale, name=name)
                if pooling:
                    tmp = pooling_func(tmp, scale)
                    if res is None:
                        res = np.copy(tmp)
                    else:
                        res = np.hstack((res, tmp))
                else:
                    if res is None:
                        res = np.copy(tmp)
                    else:
                        res = np.vstack((res, tmp))
    return res

def second_layer_extractor(landmarks, timestamps, stimulus):
    signal = first_layer_extractor(landmarks, timestamps, stimulus)
    res = second_layer_transform(signal)
    return res

def second_layer_explanation(stimulus = 1, pooling = POOLING):
    if pooling:
        for i in range(len(STIMULI)):
            if stimulus == STIMULI[i]:
                break
        increment = 1. / FPS
        length = int((LENGTHS[i] + increment) * FPS) + 1
        
    explanation = []
    for feature in FIRST_LAYER:
        for atom in NON_LINEARITY:
            name, scales = atom[0], atom[1]
            for scale in scales:
                exp = feature+', layer 2: '+name+(' scale %.1f' %scale)
                if pooling:
                    windows = []
                    increment = max(int(scale*FPS), 1)
                    for i in range(max(int(2*length / increment)-1, 1)):
                        begin = int(i*increment/2)
                        end = begin + increment
                        windows.append([begin, end])
                    for window in windows:
                        begin, end = window[0]/FPS, window[1]/FPS
                        tmp = ', pooling: begin %.1f, end %.1f' %(begin, end)
                        explanation.append(exp+tmp)
                else:
                    explanation.append(exp)
    return explanation

SECOND_LAYER = []
for stimulus in STIMULI:
    SECOND_LAYER.append(second_layer_explanation(stimulus = stimulus))

"""
----------------------------- Histogram variation -----------------------------
"""
def bins_assignation(flat_features, bins, center=True):
    features = np.copy(flat_features)
    nb_bins = len(bins)
    assignation = nb_bins * np.ones(features.shape)
    for i in range(nb_bins)[::-1]:
        thres = bins[i]
        ind = features <= thres
        assignation[ind] = i
    if center:
        assignation = assignation - len(bins)/ 2.
    return assignation

def bins_one_hot_assignation(flat_features, bins):
    features = np.copy(flat_features)
    nb_bins = len(bins)+1
    assignation = np.zeros((nb_bins, len(features)))
    for i in range(nb_bins-1)[::-1]:
        tmp = np.zeros((nb_bins,1))
        tmp[i] = 1
        thres = bins[i]
        ind = features <= thres
        assignation[:, ind] = tmp
    ind = np.sum(assignation, axis=0) == 0
    tmp = np.zeros((nb_bins,1))
    tmp[nb_bins-1] = 1
    assignation[:, ind] = tmp
    return assignation

def empirical_bins(values, nb_bins = NB_BINS, tmp = None):
    """
    Computation of bins to scatter sample in histogram according
    to the distribution that generated the values "values"
    The implementation consists in inversing the repartition function
    tmp allow to only access some p_value one is looking for
    """
    """
    Tabulate values seen to speed up computation
    """
    empirical_bins.values = np.sort(values)
    empirical_bins.nb_values = len(values)
    empirical_bins.x = np.array([])
    empirical_bins.y = np.array([])
    empirical_bins.nb_tab = 0

    def phi(x):
        """
        phi is the repartition function of the distribution to scatter
        """
        tmp = np.searchsorted(empirical_bins.values, x)
        tmp /= empirical_bins.nb_values
        
        # Tabulate the value seen
        i = np.searchsorted(empirical_bins.x, x)
        if i == empirical_bins.nb_tab:
            empirical_bins.x = np.append(empirical_bins.x, x)
            empirical_bins.y = np.append(empirical_bins.y, tmp)
        else:
            empirical_bins.x = np.insert(empirical_bins.x, i, x)
            empirical_bins.y = np.insert(empirical_bins.y, i, tmp)
        empirical_bins.nb_tab += 1
        return tmp
    phi(0)
    def inv_phi(y, precision = .01):
        """ find bound """
        i = np.searchsorted(empirical_bins.y, y)
        if i==empirical_bins.nb_tab:
            """ no tabulated upper bound """
            i -= 1
            lower = empirical_bins.x[i]
            upper = max(1,lower)
            while y > phi(upper):
                lower = upper
                upper *= 2
        elif i==0:
            """ no tabulated lower bound """
            upper = empirical_bins.x[i]
            lower = min(-1, upper)
            while y < phi(lower):
                upper = lower
                lower *= 2
        else:
            lower = empirical_bins.x[i-1]
            upper = empirical_bins.x[i]

        """ dichotomic search """
        while upper - lower > precision:
            x = (upper+lower)/2
            if y > phi(x):
                lower = x
            else:
                upper = x
        x = (upper+lower)/2
        return x
    if type(tmp) == type(None):
        incr = 1/nb_bins
        tmp = np.arange(0+incr, 1-incr/2, incr)
    bins = np.zeros(tmp.shape[0])
    for i in range(len(bins)):
        bins[i] = inv_phi(tmp[i])
    return bins

def get_histogram_bins(nb_bins = NB_BINS):
    name = 'all_bins'
    try:
        bins = get_histogram_bins.BINS
    except AttributeError:
        try:
            print('Loading histogram bins.')
            bins = load_variables(name)
        except FileNotFoundError:
            print('Computing histogram bins.')
            ext = first_layer_extractor
            st_desc, _, _ = homogenized_description(ext, homogenize=False)
            all_val = concatenate_all(st_desc)
            all_val[np.isnan(all_val)] = 0
            
            length = all_val.shape[0]
            if type(nb_bins) == int:
                nb_bins = nb_bins * np.ones(length)
            
            bins = [[] for k in range(length)]
            for k in range(length):
                val = all_val[k,:]
                bins[k] = empirical_bins(val, nb_bins = nb_bins[k])
            save_variables(name, bins)
    get_histogram_bins.BINS = bins
    return bins

def assignation_extractor(landmarks, timestamps, stimulus):
    signal = first_layer_extractor(landmarks, timestamps, stimulus)
    length = signal.shape[0]
    bins = get_histogram_bins()
    res = None
    for k in range(length):
        if ONE_HOT:
            tmp = bins_one_hot_assignation(signal[k], bins[k])
        else:
            tmp = bins_assignation(signal[k], bins[k])
        if res is None:
            res = np.copy(tmp)
        else:
            res = np.vstack((res,tmp))
    return res

def assignation_explanation(nb_bins = NB_BINS, one_hot = ONE_HOT):
    length = len(FIRST_LAYER)
    if type(nb_bins) == int:
        nb_bins = nb_bins * np.ones(length, dtype = np.int)
    explanation = []
    for k in range(length):
        nb_bin = nb_bins[k]
        if one_hot:
            tmp = (nb_bin-1)/2.
            for i in range(nb_bin):
                i = (i - tmp)/tmp
                explanation.append(FIRST_LAYER[k] + (' intensity %.1f' %i))
        else:
            explanation.append(FIRST_LAYER[k] + (' %d intensities' %nb_bin))
    return explanation
    
ASSIGNATION = assignation_explanation(one_hot = ONE_HOT)

def localize_histogram(assignation, pooling = POOLING, one_hot = ONE_HOT):
    length = assignation.shape[0]
    res = None
    for k in range(length):
        for scale in SECOND_SCALES:
            tmp = local_averaging(assignation[k], scale)
            if pooling:
                tmp = pooling_func(tmp, scale)
                if res is None:
                    res = np.copy(tmp)
                else:
                    res = np.hstack((res,tmp))
            else:
                if res is None:
                    res = np.copy(tmp)
                else:
                    res = np.vstack((res,tmp))
    return res

def histogram_extractor(landmarks, timestamps, stimulus):
    signal = assignation_extractor(landmarks,timestamps,stimulus)
    res = localize_histogram(signal)
    return res

def histogram_explanation(stimulus = 1, pooling = POOLING, one_hot = ONE_HOT):
    if pooling:
        for i in range(len(STIMULI)):
            if stimulus == STIMULI[i]:
                break
        increment = 1. / FPS
        length = int((LENGTHS[i] + increment) * FPS) + 1

    explanation = []
    for feature in ASSIGNATION:
        for scale in SECOND_SCALES:
            exp = feature + (' histo: scale %d' %scale)
            if pooling:
                windows = []
                increment = max(int(scale*FPS), 1)
                for i in range(max(int(2*length / increment)-1, 1)):
                    begin = int(i*increment/2)
                    end = begin + increment
                    windows.append([begin, end])
                for window in windows:
                    begin, end = window[0]/FPS, window[1]/FPS
                    tmp = ', pooling: begin %.1f, end %.1f' %(begin, end)
                    explanation.append(exp+tmp)
            else:
                explanation.append(exp)
    return explanation

HISTOGRAM = []
for stimulus in STIMULI:
    tmp = histogram_explanation(stimulus = stimulus)
    HISTOGRAM.append(tmp)
