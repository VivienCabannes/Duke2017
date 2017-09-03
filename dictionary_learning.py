#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import numpy as np
from scipy.signal import fftconvolve
from signal_processing import preprocess_signal

import matplotlib.pyplot as plt
from global_path import FPS, LENGTHS, SCALES, STIMULI
from helper import norm, normalization, distance_l2
from data_handling import get_data, get_ids
from face_parameterization import features_extraction

"""
---------------------------- Refined Convolutions -----------------------------
"""
def symetrize(signal, size, begin, end):
    length = end - begin
    extend_signal = np.zeros(size)
    extend_signal[begin:end] = signal
    
    if begin < length:
        tmp = np.copy(signal[:begin])
        tmp = -tmp + 2*tmp[0]
        extend_signal[:begin] = tmp[::-1]
    else:
        tmp = np.copy(signal)
        tmp = -tmp + 2*tmp[0]
        extend_signal[begin-length:begin] = tmp[::-1]
    if size-end < length:
        tmp = np.copy(signal[end-size:])
        tmp = - tmp + 2*tmp[-1]
        extend_signal[end:] = tmp[::-1]
    else:
        tmp = np.copy(signal)
        tmp = -tmp + 2*tmp[-1]
        extend_signal[end:end+length] = tmp[::-1]
    return extend_signal
    
def convolution_sym(signal, filt):
    n, length = len(filt), len(signal)
    begin = n
    end = begin + length
    size = end + n
    extend_signal = symetrize(signal, size, begin, end)
    res = fftconvolve(extend_signal, filt, mode = 'same')
    return res[begin:end]

def convolution_weigth(signal, filt, reg = 1, thres = .01):
    x, H = preprocess_signal(signal, affine = False)
    if reg==1:
        tmp = filt
    elif reg==0:
        tmp = np.ones(filt.shape)
    weight = fftconvolve(H, tmp, mode = 'same')
    weight[weight < thres] = 1
    res = fftconvolve(x, filt, mode = 'same')
    return res / weight

def convolution(signal, filt, mode = 'same'):
    if mode == 'sym':
        return convolution_sym(signal, filt)
    elif mode == 'weight':
        return convolution_weigth(signal, filt, 1)
    elif mode == 'weight receptive':
        return convolution_weigth(signal, filt, 0)
    else:
        return fftconvolve(signal, filt, mode = mode)
    
"""
----------------------------------- Helpers -----------------------------------
""" 
def threshold(x, thres, soft = True):
    if soft:
        tmp = np.maximum(abs(x), 1e-7*np.ones(np.shape(x)))
        ind = np.maximum(1-thres/tmp, np.zeros(np.shape(x)))
    else:
        ind = np.abs(x) > thres
    return x*ind

def thresholding(coef, thres, soft = True):
    res = []
    for i in range(len(coef)-1):
        detail = coef[i]
        res.append(threshold(detail, thres, soft))
    coarse = coef[-1]
    res.append(coarse)
    return res

def local_maxima(signal, thres = .5):
    """
    Compute local maxima in each channel x[i,:] falling above threshold
    """
    x = np.abs(signal)
    above_right = np.zeros(x.shape).astype(np.bool)
    tmp = x[:, :-1] > x[:, 1:]
    above_right[:,:-1] = np.copy(tmp)
    above_right[:,-1] = True
    above_left = np.zeros(x.shape).astype(np.bool)
    above_left[:,1:] = np.copy(np.invert(tmp))
    above_left[:,0] = True
    maxima = np.logical_and(above_left, above_right)
    maxi = np.max(x, axis=1)
    above_thres = x >= thres * np.expand_dims(maxi, axis=1)
    local_maxi = np.logical_and(maxima, above_thres)
    return local_maxi

def sc_sp_local_maxima(scale_space, thres = .5, full=True):
    """
    Compute local maxima in 2D falling above threshold
    """
    x = np.abs(scale_space)
    
    space_filter = np.array([[1,-1]])
    tmp = convolution(x, space_filter, mode = 'same') > 0 
    above_left = np.copy(tmp)
    above_left[:,0] = True
    above_right = np.zeros(x.shape).astype(np.bool)
    above_right[:,:-1] = np.copy(np.invert(tmp[:,1:]))
    above_right[:,-1] = True
    maxima_space = np.logical_and(above_left, above_right)

    scale_filter = np.array([[1],[-1]])
    tmp = convolution(x, scale_filter, mode = 'same') > 0 
    above_up = np.copy(tmp)
    above_up[0,:] = True
    above_down = np.zeros(x.shape).astype(np.bool)
    above_down[:-1,:] = np.copy(np.invert(tmp[1:,:]))
    above_down[-1,:] = True
    maxima_scale = np.logical_and(above_down, above_up)

    maxima = np.logical_and(maxima_scale, maxima_space)
    
    if full:
        diag1_filter = np.array([[1,0],[0,-1]])
        tmp = convolution(x, diag1_filter, mode = 'same') > 0 
        diag11 = np.copy(tmp)
        diag11[0,:] = True
        diag11[:,0] = True
        diag12 = np.zeros(x.shape).astype(np.bool)
        diag12[:-1,:-1] = np.copy(np.invert(tmp[1:,1:]))
        diag12[-1,:] = True
        diag12[:,-1] = True
        diag1 = np.logical_and(diag11, diag12)

        diag2_filter = np.array([[0,-1],[1,0]])
        tmp = convolution(x, diag2_filter, mode = 'same')[1:,1:] > 0
        diag21 = np.zeros(x.shape).astype(np.bool)
        diag21[:-1,1:] = tmp
        diag21[-1,:] = True
        diag21[:,0] = True
        diag22 = np.zeros(x.shape).astype(np.bool)
        diag22[1:,:-1] = np.copy(np.invert(tmp))
        diag22[0,:] = True
        diag22[:,-1] = True
        diag2 = np.logical_and(diag21, diag22)
        
        diag = np.logical_and(diag1, diag2)
        maxima = np.logical_and(maxima, diag)

    maxi = np.max(x)
    above_thres = x >= thres * maxi
    local_maxi = np.logical_and(maxima, above_thres)
    
    return local_maxi

def force_separation(scale_space, scales, thres = .5):
    x = np.abs(scale_space)
    maxima = sc_sp_local_maxima(scale_space, thres = thres) 
    ind = np.arange(len(x.flatten()))[maxima.flatten()]
    values = x.flatten()
    length = x.shape[1]
    last_pos, last_val, res = -100, 0, [0]
    for j in ind:
        scale, pos, val = scales[j//length], j % length, values[j]
        if np.abs(pos - last_pos) < (FPS * scale):
            if last_val < val:
                res[-1] = j
                last_val = val
            continue
        last_val, last_pos = val, pos
        res.append(j)            
    return np.array(res)

"""
---------------------------------- Wavelets -----------------------------------
"""
def standard_wavelet():
    h = np.array([0, (1 + np.sqrt(3)) / (4*np.sqrt(2)), 
                  (3 + np.sqrt(3)) / (4*np.sqrt(2)),
                  (3 - np.sqrt(3)) / (4*np.sqrt(2)), 
                  (1 - np.sqrt(3)) / (4*np.sqrt(2))])
    u = np.power(-np.ones(len(h)-1),range(1,len(h)))
    g = np.concatenate(([0], h[-1:0:-1] * u))
    return h, g

"""
----------------------------- Wavelets Transform ------------------------------
"""
def ind_wt_signal(length, scale = 0):
    """
    Raise signal length to a power of 2
    scale = 0:
        raise to next power
    scale = 2:
        raise to the two further for a complete symetrization of the signal
    """
    size = 2**(np.ceil(np.log2(length)).astype(np.int)+scale)
    begin = int((size-length)/2)
    end = length + begin
    return begin, end, size

def wt_preprocess(signal):
    length = len(signal)
    begin, end, size = ind_wt_signal(length)
    return symetrize(signal, size, begin, end)
    
def wavelet_transform(signal, h=None, g=None, mode='same', blinded=True):
    if h is None:
        h, g = standard_wavelet()
    coarse = wt_preprocess(signal)
    coef = []
    while len(coarse) > 1:
        blind = np.isnan(coarse)
        if not mode in ['weight', 'weight receptive']:
            coarse[blind] = 0

        detail = convolution(coarse, g, mode)
        coarse = convolution(coarse, h, mode)

        coef.append(np.copy(detail[::2]))
        if len(coarse)==2:
            coef.append(np.copy(coarse[::2]))
            return coef

        if blinded:
            if mode == 'valid':
                r = int(len(h)/2)
                detail[blind[r:-r]] = np.nan
                coarse[blind[r:-r]] = np.nan
    
            detail[blind] = np.nan
            coarse[blind] = np.nan
            
        detail = detail[::2]
        coarse = coarse[::2]
    return coef

def inverse_wavelet_transform(coef, length=None, h=None, g=None):
    if h is None:
        h, g = standard_wavelet()
    h_rev, g_rev = h[::-1], g[::-1]
    def upsampling(x):
        y = np.zeros(2 * len(x))
        y[::2] = x
        return y

    i = 1    
    coarse = coef[-i]
    while i < len(coef):
        i += 1
        detail = coef[-i]
        coarse = convolution(upsampling(coarse), h_rev, mode = 'same')
        detail = convolution(upsampling(detail), g_rev, mode = 'same')
        coarse = np.array(coarse) + np.array(detail)
        
    if not length is None:
        begin, end, _ = ind_wt_signal(length)
        coarse = coarse[begin:end]
    return coarse
       
def time_frequency_representation(coef, timestamps = None, verbose = False):
    if timestamps is None:
        timestamps = np.arange(len(coef[0])) / 15.
    length = len(timestamps)
    def upsample(detail):
        current_length = len(detail)
        blocks = np.linspace(0, length, num = current_length+1).astype(np.int)
        res = np.zeros(length)
        end = 0
        for i in range(current_length):
            begin = end
            end = blocks[i+1]
            res[begin:end] = detail[i]
        res[np.abs(res)<0]=0
        return res
    l = len(coef)
    scale_space = np.zeros((length, l))
    for i in range(l):    
        scale_space[:,i] = upsample(coef[i])

    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('time')
        ax.set_ylabel('frequency')
        x, y = np.meshgrid(timestamps, -np.arange(l), indexing = 'ij')
        surf = ax.pcolor(x, y, scale_space, cmap='gist_heat', linewidth=0, 
                         antialiased=False, vmax=.5, vmin = -.5)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        return scale_space, fig
    return scale_space

"""
--------------------------- Wavelets Interpolation ----------------------------
"""
def alternative_projection(signal, lambd = .1, alpha = 0, soft = True, 
                           tol = 10**(-2), num_it = 25, 
                           mode = 'same', affine = True):
    length = len(signal)
    guess, H = preprocess_signal(signal, affine = affine)
    sig = np.copy(guess)
    if mode == 'weigth':
        guess = signal

    def projection(guess):
        return H*sig + (1-H)*guess 
    
    def gradient(guess):
        return alpha * guess + (1-alpha)*projection(guess)

    def wavelet_threshold(guess, thres):
        coef = wavelet_transform(guess, mode = mode)
        coef = thresholding(coef, thres, soft)
        return inverse_wavelet_transform(coef, length = length)

    sig_norm, i = norm(sig), 1
    while norm(sig - guess) / sig_norm > tol and i < num_it:
        i += 1
        guess = wavelet_threshold(gradient(guess), lambd)
    return guess

"""
----------------------------- Hand-Made Patterns ------------------------------
"""
def Gaussian(x, complete = True, p = 1):
    if complete:
        b = 10
    else:
        b = 1.25
    filt = np.exp( -  (x*b)**2 / 2 )
    return normalization(filt, p = p)

def derivative_Gaussian(x, complete = True, p = 1):
    if complete:
        b = 10
    else:
        b = 1.25
    filt = (x*b) * np.exp( -  (x*b)**2 / 2 )
    return normalization(filt, p = p)

def Gaussian_bis(x):
    return Gaussian(x, complete = False)

def TurnUp(x):
    l = 10
    filt = (1 + np.exp(- l*x ))**(-1)
    return normalization(filt)

def TurnDown(x):
    l = 10
    filt = (1 + np.exp( l*x ))**(-1)
    return normalization(filt)

def StandingStill(x):
    filt = np.ones(x.shape)
    return normalization(filt)

"""
-------------------------------- Dictionaries ---------------------------------
"""
class Atom:
    def __init__(self, pattern =  Gaussian, 
                 scales = np.array([10, 70, 300])):
        self.pattern = pattern
        self.scales = scales

atom1 = Atom(scales = SCALES)
atom2 = Atom(pattern = Gaussian_bis, scales = np.logspace(0,3,num=7))
    
class Dictionary:
    def __init__(self, atoms = [Atom()]):
        self.atoms = atoms

sc_sp_dict = Dictionary([atom1])
ln_dict = Dictionary([atom2])

"""
---------------------------- Dictionary Operations ----------------------------
"""
def scaling(scale, fps = FPS):
    nb_points = scale*fps
    return np.linspace(-1, 1, num = 2*(nb_points//2)+1)
    
class DictionaryTransform:
    def __init__(self, stimulus = 1, dictionary = sc_sp_dict):
        self.dictionary = dictionary
        for i in range(len(STIMULI)):
            if stimulus == STIMULI[i]:
                break
        length = LENGTHS[i]
        increment = 1. / FPS
        self.frames = np.int((length + increment) * FPS) + 1
        self.get_hashing()
            
    def transform(self, signal, mode = 'weight'):
        res = np.array([])                 
        for atom in self.dictionary.atoms:
            for scale in atom.scales:
                filt = atom.pattern(scaling(scale))
                tmp = convolution(signal, filt, mode)
                res = np.append(res, tmp)
        return res
    
    def inverse(self, res):
        signal = np.zeros(self.frames)
        end = 0
        for atom in self.dictionary.atoms:
            for scale in atom.scales:
                filt = atom.pattern(scaling(scale))[::-1]
                begin = end
                end = begin + self.frames
                tmp = np.copy(res[begin:end])
                tmp = convolution(tmp, filt) 
                signal = signal + tmp
        return signal
            
    def get_hashing(self):
        nb_atoms = len(self.dictionary.atoms)
        atoms_ind = [0]
        s_max = 0
        for i in range(nb_atoms):
            nb_scales = len(self.dictionary.atoms[i].scales)
            if nb_scales > s_max:
                s_max = nb_scales
            atoms_ind.append(nb_scales * self.frames + atoms_ind[-1])
        self.atoms_ind = np.array(atoms_ind)
        self.scales_ind = np.array([(i+1) * self.frames for i in range(s_max)])
        
    def get_atom_info(self, i):
        atom_ind = np.sum(self.atoms_ind[1:] <= i)
        i -= self.atoms_ind[atom_ind]
        scale_ind = np.sum(self.scales_ind <= i)
        pos = i - scale_ind * self.frames
        filt =  self.dictionary.atoms[atom_ind]
        scale = filt.scales[scale_ind]
        return filt, scale, pos
        
    def build_atom(self, filt, scale, pos):
        length = self.frames
        atom = np.zeros(length)
        x = filt.pattern(scaling(scale))
        size = len(x)
        begin = pos - np.floor(size / 2.).astype(np.int)
        end = begin+size
        if begin < 0:
            x = x[-begin:]
            begin = 0
        if end > length:
            x = x[:length-end]
            end = length
        atom[begin:end] = x
        return atom

    def get_atom(self, i):
        filt, scale, pos = self.get_atom_info(i)
        atom = self.build_atom(filt, scale, pos)
        return atom
    
    def get_pattern(self, signal, scale, pos):
        length = self.frames
        size = len(scaling(scale))
        begin = pos - np.floor(size / 2.).astype(np.int)
        end = begin+size
        tmp_begin = 0
        tmp_end = 0
        if begin < 0:
            tmp_begin = np.abs(begin)
            begin = 0
        if end > length:
            tmp_end = end - length
            end = length            
        pattern = np.copy(signal[begin:end])
        for i in range(tmp_begin):
            pattern = np.insert(pattern, 0, np.nan)
        for i in range(tmp_end):
            pattern = np.append(pattern, np.nan)
        return pattern
    
    def inverse_sparse(self, coef):
        ind = np.arange(len(coef))[np.invert(coef==0)]
        signal = np.zeros(self.frames)
        for i in ind:
            filt, scale, pos = self.get_atom_info(i)
            atom = self.build_atom(filt, scale, pos)
            signal = signal + coef[i]*atom
        return signal
        
    def show_reconstruction(self, coef, ax):
        ind = np.arange(len(coef))[np.invert(coef==0)]
        for i in ind:
            filt, scale, pos = self.get_atom_info(i)
            atom = self.build_atom(filt, scale, pos)
            g = coef[i]*atom
            g[np.abs(g)<10**(-2)] = np.nan
            ax.plot(np.arange(self.frames)/FPS, g)

    def scale_space(self, res, verbose = False, scales = None, 
                    vmax = None, vmin = None):
        tmp = len(res)
        assert(tmp % self.frames == 0)
        nb_scales = tmp // self.frames
        sc_sp = res.reshape((nb_scales, self.frames))
        if verbose:
            assert(nb_scales == len(scales))
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel(r'time (s)')
            ax.set_ylabel(r'frequency ($\sigma^{-1}$)', fontsize = 18)
            timestamps = np.arange(self.frames) / FPS
            x, y = np.meshgrid(timestamps, 100/scales, indexing = 'ij')
            surf = ax.pcolor(x, y, sc_sp.transpose(), 
                             cmap='gist_heat', linewidth=0, 
                             antialiased=False, vmax=vmax, vmin = vmin)
            ax.semilogy()
            ax.set_title(r'Scale Space')
            fig.colorbar(surf, shrink=0.5, aspect=5)
            return sc_sp, fig            
        return sc_sp
        
    def scale_spaces(self, res, verbose = False, vmax = None, vmin = None):
        tmp = len(res)
        nb_atoms = np.sum(self.atoms_ind < tmp)
        assert(nb_atoms == len(self.dictionary.atoms))
        sc_sps, figs, self.scales = [], [], []
        for i in range(nb_atoms):
            scales = self.dictionary.atoms[i].scales
            begin, end = self.atoms_ind[i], self.atoms_ind[i+1]
            tmp = res[begin:end]
            self.scales.append(scales)
            tmp = self.scale_space(tmp, verbose = verbose, scales = scales, 
                                   vmax = vmax, vmin = vmin)
            if verbose:
                sc_sp, fig = tmp[0], tmp[1]
                sc_sps.append(sc_sp)
                figs.append(fig)
            else:
                sc_sps.append(tmp)
        if verbose:            
            return sc_sps, figs           
        return sc_sps
    
    def build_scale_scape(self, signal, mode = 'weight', verbose = False):
        res = self.transform(signal, mode = mode)
        return self.scale_spaces(res, verbose = verbose)
        
    def alternative_projection(self, signal, mode = 'weight', verbose = False,
                               lambd = .2, soft = False, scaled = True,
                               tol = 10**(-5), num_it = 15):
        x, H = preprocess_signal(signal, affine = True)
        coef = self.transform(signal)
        reconstruct = self.inverse(coef) 

        sig_norm, iteration = norm(x), 1
        while norm(x - H*reconstruct) / sig_norm > tol and iteration < num_it:
            iteration += 1
            coef = coef + 2*lambd*self.transform(H*(x-reconstruct), mode=mode)
            if scaled:
                coef = np.reshape(coef, (len(coef)//self.frames, self.frames))
                for i in range(len(coef)):
                    coef[i,:] = thresholding(coef[i,:], lambd, soft=soft)
                coef = coef.flatten()
            else:
                coef = thresholding(coef, lambd, soft)
            reconstruct = self.inverse(coef) 

        if verbose:
            return coef, reconstruct
        return coef
    
    def pursuit(self, signal, method = 'sc_sp', mode = 'weight', 
                nb_it = 2, thres = 0.2, tol = 10**(-3), verbose = False):
#                method = 'force', nb_it = 1, thres = 0, dictionary = ln_dict
        """
        methods:
            'maximum': add atom one by one
            'maxima': add local maxima at each scale with iteration
                put thres = 1 to add only one atom per scale
            'sc_sp': add scale space maxima
            'force': force maxima separation
        """
        x, H = preprocess_signal(signal, affine = True)  
        x0 = np.copy(x)
        
        all_ind = np.array([], dtype = np.int)
        iteration = 0
        while (np.mean(x**2) > tol and iteration < nb_it) or iteration == 0:
            iteration = iteration+1
            
            if mode in ['weight', 'weight receptive']:
                x[H==0] = np.nan
            coef = self.transform(x, mode = mode)
            if method == 'maxima':
                res = coef.reshape((len(coef)//self.frames, self.frames))
                maxima = local_maxima(res, thres = thres).flatten()
                ind = np.arange(len(coef))[maxima]
            elif method == 'sc_sp':
                sc_sps = self.scale_spaces(coef)
                maxima = np.array([], dtype = np.bool)
                for i in range(len(sc_sps)):
                    sc_sp, scale = sc_sps[i], self.scales[i]
                    res = sc_sp_local_maxima(sc_sp, thres = thres)
                    maxima = np.append(maxima, res.flatten())
                ind = np.arange(len(coef))[maxima]
            elif method == 'force':
                sc_sps = self.scale_spaces(coef)
                maxima = np.array([], dtype = np.bool)
                for i in range(len(sc_sps)):
                    sc_sp, scale = sc_sps[i], self.scales[i]
                    res = force_separation(sc_sp, scale, thres = thres)
                    maxima = np.append(maxima, res.flatten())
                ind = np.arange(len(coef))[maxima]
            else:
                assert(method == 'maximum')
                ind = [np.argmax(coef)]
                
            # Build dictionary
            all_ind = np.append(all_ind, ind) 
            all_ind = list(set(all_ind))
            Psi = None
            for i in all_ind:
                atom = self.get_atom(i)
                if Psi is None:
                    Psi = np.copy(atom)
                else:
                    Psi = np.vstack((Psi, atom))
            if len(Psi.shape) < 2:
                Psi = np.expand_dims(Psi, axis=0)
                    
            a = np.matmul(np.linalg.pinv(Psi.transpose()), x0)
            f = np.matmul(Psi.transpose(), a)
            x = x0 - H*f
            
        coef = np.zeros(coef.shape)
        coef[all_ind] = a
        if verbose:
            return coef, f
        return coef
    
"""
----------------------------- Dictionary Learning -----------------------------
"""
def rescale(x, scale = 1):
    points = scaling(scale)
    res = np.zeros(len(points))
    current = np.linspace(-1, 1, num = len(x))
    # Iterative scheme. Could be faster with matrix operations
    res[0] = x[0]
    res[-1] = x[-1]
    for i in range(1,len(points)-1):
        point = points[i]
        ind = np.sum(current < point)
        if np.isnan(x[ind-1]):
            res[i] = x[ind]
        elif np.isnan(x[ind]):
            res[i] = x[ind-1]
        else:
            tmp0, tmp1 = point - current[ind-1], current[ind] - point
            res[i] = (tmp1 * x[ind-1] + tmp0 * x[ind]) / (tmp0 + tmp1)
    return res
    
all_ids, all_labels = get_ids()
class DictionaryLearning:
    def __init__(self, ids = all_ids, features = 0, scale = 1, 
                 distance = distance_l2):
        DT = []
        for i in range(len(STIMULI)):
            DT.append(DictionaryTransform(stimulus = STIMULI[i], 
                                          dictionary = ln_dict))
        self.DT = DT
        self.ids = ids
        self.features = features
        self.scale = scale
        self.distance = distance
        
    def get_dict(self, stimulus):
        for i in range(len(STIMULI)):
            if stimulus == STIMULI[i]:
                return self.DT[i]
        print('Error in Dictionary Learning, get_dict')
        return None
        
    def get_all_pattern(self, thres = 0.1, mode = 'same', verbose = False):
        patterns = None
        infos = None
        for stimulus in STIMULI:
            D = self.get_dict(stimulus)
            for i in range(len(self.ids)):
                patient_id = self.ids[i]
                try:
                    landmarks, timestamps = get_data(stimulus, patient_id)
                except FileNotFoundError:
                    print('Patient %s, stimulus %d not found' %(patient_id, 
                                                                stimulus))
                    continue
                
                features = features_extraction(landmarks, timestamps, stimulus)
                signal = np.copy(features[0])
                x, H = preprocess_signal(signal)
                
                if verbose:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)                    
                    ax.plot(signal)
                    ax.set_title(patient_id + ', stimulus ' + str(stimulus))

                if mode in ['weight', 'weight receptive']:
                    x[H==0] = np.nan
                coef = D.transform(x, mode = mode)
                sc_sps = D.scale_spaces(coef)
                maxima = np.array([], dtype = np.bool)
                for k in range(len(sc_sps)):
                    sc_sp = sc_sps[k]
                    res = sc_sp_local_maxima(sc_sp, thres = thres)
                    maxima = np.append(maxima, res.flatten())
                ind = np.arange(len(coef))[maxima]
                for j in ind:
                    _, scale, pos = D.get_atom_info(j)
                    pattern = D.get_pattern(signal, scale, pos) / coef[j]
                    pattern = rescale(pattern, scale = self.scale)
                    info = np.array([i, stimulus, scale, pos, coef[j]])
                    if patterns is None and infos is None:
                        patterns = np.copy(pattern)
                        infos = np.copy(info)
                    else:
                        patterns = np.vstack((patterns, pattern))
                        infos = np.vstack((infos, info))
        return patterns, infos
    
    def build_atom(self, pattern, stimulus, scale, pos):
        length = self.get_dict(stimulus).frames
        atom = np.zeros(length)
        x = rescale(pattern, scale)
        size = len(x)
        begin = (pos - np.floor(size / 2.)).astype(np.int)
        end = begin+size
        if begin < 0:
            x = x[-begin:]
            begin = 0
        if end > length:
            x = x[:length-end]
            end = length
        atom[begin:end] = x
        return atom
    
    def reconstruction_without_clustering(self, patterns, infos):
        st = []
        for k in range(len(STIMULI)):
            st.append(np.zeros((len(self.ids), self.DT[k].frames)))
        for j in range(len(infos)):
            pattern = patterns[j]
            i, stimulus, scale, pos, amp = infos[j]
            i = int(i)
            atom = self.build_atom(pattern, stimulus, scale, pos)
            for k in range(len(STIMULI)):
                if stimulus == STIMULI[k]:
                    st[k][i,:] += amp*atom
        return st
    
    def clustering(self, patterns, k=5, max_it=20, nb_try=3, verbose=False):
        """
        k-means algorithm
        """
        nb_pt = len(patterns)
        x = []
        for pattern in patterns:
            tmp, _ = preprocess_signal(pattern)
            x.append(tmp)
        x = np.array(x)
        
        def one_try(nb_pt=nb_pt, x=x, patterns=patterns, k=k, max_it=max_it):
            # Initialization with K random center
            center_ind = np.random.permutation(np.arange(nb_pt))[:k]
            center = patterns[center_ind, :]
            center_old = np.zeros(center.shape);
            
            # Until convergence criterion is not reached
            it = 0
            while (not np.sum(center_old == center) == 1) and it < max_it: 
                it += 1
                if it%10==0:
                    print(it)
                center_old = np.copy(center)
                
                # Compute distances to centers
                dist = self.distance(x, center)
                M = np.min(dist, axis=1)
                y = np.argmin(dist, axis=1)
                
                for i in range(k):
                    ind = y == i
                    if np.sum(ind)==0:
                        ind = int(np.random.rand() * nb_pt)
                    center[i,:] = np.mean(x[ind,:],axis=0)
            dist = self.distance(x, center)
            M =  np.min(dist, axis=1)
            distortion = np.sum(M)                
            return distortion, center, y
        
        best = np.inf
        for i in range(nb_try):
            distortion, center, y = one_try()
            if verbose:
                print(distortion)
                plt.figure()
                for pattern in center:
                    plt.plot(pattern)
            if distortion <= best:
                best = np.copy(distortion)
                self.words = np.copy(center)
                clusters = np.copy(y)
        return clusters        

    def reconstruction(self, clusters, infos):
        st = []
        for k in range(len(STIMULI)):
            st.append(np.zeros((len(self.ids), self.DT[k].frames)))
        for j in range(len(infos)):
            pattern = self.words[clusters[j]]
            i, stimulus, scale, pos, amp = infos[j]
            i = int(i)
            atom = self.build_atom(pattern, stimulus, scale, pos)
            for k in range(len(STIMULI)):
                if stimulus == STIMULI[k]:
                    st[k][i,:] += amp*atom
        return st
