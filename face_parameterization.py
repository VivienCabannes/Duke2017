#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------------------------- Libraries ---------------------------------- 
"""
from global_path import FPS, LENGTHS, STIMULI
import numpy as np

"""
------------------------------ Helper Functions -------------------------------
"""    
def angles(landmarks, pt1, pt2, pt3, pt4):
    x1 = landmarks[pt2, :] - landmarks[pt1, :]
    x2 = landmarks[pt4, :] - landmarks[pt3, :]
    y1 = landmarks[pt2+49, :] - landmarks[pt1+49, :]
    y2 = landmarks[pt4+49, :] - landmarks[pt3+49, :]
    d1 = (x1**2 + y1**2)**.5
    d2 = (x2**2 + y2**2)**.5
    ux, uy = x1/d1, y1/d1
    vx, vy = x2/d2, y2/d2
    ps = ux*vx + uy*vy
    ax, ay = ux - ps * vx, uy - ps * vy
    angles = vx*ay - vy*ax
    # Correct instability
    angles[np.isnan(angles)] = 0
    angles[angles >  1] = 1
    angles[ -angles >  1] = -1
    return angles  

def distances(landmarks, pt1, pt2):
    x = (landmarks[pt1,:] - landmarks[pt2,:])**2
    y = (landmarks[pt1+49,:] - landmarks[pt2+49,:])**2
    return (x+y)**.5

def distances_proj(landmarks, pt1, pt2, pt3, pt4):
    ux = landmarks[pt3,:] - landmarks[pt4,:]
    uy = landmarks[pt3+49,:] - landmarks[pt4+49,:]
    x = landmarks[pt1,:] - landmarks[pt2,:]
    y = landmarks[pt1+49,:] - landmarks[pt2+49,:]
    return (ux*x + uy*y) / ((ux**2+uy**2)**.5)

def normalization_extension(feature, log = True):
    feature /= np.median(feature)
    if log:
        feature = np.log2(feature)
        # Correct instability
        feature[np.isnan(feature)] = 0
        feature[feature<-2]=-2
    else:
        feature[np.isnan(feature)] = 0
    return feature

def derive_depth(landmarks):
    a, b = landmarks[28,:] - landmarks[16,:], landmarks[65,:] - landmarks[77,:]
    c, d = landmarks[19,:] - landmarks[16,:], landmarks[65,:] - landmarks[68,:]
    n_area = np.abs(a*d - b*c);
    depth = n_area**.5
    x = np.mean(landmarks[:49, :], axis=0)
    y = np.mean(landmarks[49:, :], axis=0)
    return depth, x, y

def normalize(landmarks):
    """
    Eventually add normalization regarding yaw, pitch and roll
    If we have a 3D model of the face
    """
    if len(landmarks.shape)==1:
        landmarks = np.expand_dims(landmarks, axis=1)
    depth, x, y = derive_depth(landmarks)
    landmarks[:49, :] -= x
    landmarks[49:, :] -= y
    landmarks /= depth
    return landmarks, depth, x, y  
    
"""
------------------------- Parameterization Extraction -------------------------
"""
def extract_parametrization(landmarks, normalized = True, depth_norm = False):
    if len(landmarks.shape)==1:
        landmarks = np.expand_dims(landmarks, axis=1)

    # Normalized landmarks
    landmarks, depth, x, y = normalize(landmarks)
        
    # Head pose features
    nose_down = np.array([np.mean(landmarks[14:19,:], axis=0),
                          np.mean(-landmarks[63:68,:], axis=0)])
    nose_mid = np.array([landmarks[13,:], -landmarks[62,:]])
    mouth = np.array([np.mean(landmarks[31:,:], axis=0), 
                      np.mean(-landmarks[80:,:], axis=0)])
    nose_up = np.array([landmarks[10,:], -landmarks[59,:]])
    eye_left = np.array([landmarks[19,:], -landmarks[68,:]])
    eye_right = np.array([landmarks[28,:], -landmarks[77,:]])
    
    vector0 = mouth - nose_up
    vector1 = eye_left - nose_up
    vector2 = eye_right - nose_up
    ps0 = np.sum(vector0 * vector1, axis=0) / (np.sum(vector0**2, axis=0))
    ps1 = np.sum(vector0 * vector1, axis=0) / (np.sum(vector0**2, axis=0))
    y_1 = np.sum((vector1 - ps0 * vector0)**2, axis=0)**.5
    y_2 = np.sum((vector2 - ps1 * vector0)**2, axis=0)**.5
    yaw = (y_1 - y_2)/(y_1 + y_2)

    y_l = landmarks[68,:] - landmarks[77,:]
    x_l = landmarks[28,:] - landmarks[19,:]
    roll = y_l/x_l

    vector1 = 2*(nose_mid - nose_down)
    vector2 = nose_up - nose_mid
    x_1 = np.sum(vector1**2, axis=0)**.5
    x_2 = np.sum(vector2**2, axis=0)**.5
    pitch = (x_1-x_2)/(x_1+x_2)

    if depth_norm:
        x -= np.mean(x)
        x /= np.mean(depth)
        y -= np.mean(y)
        y /= np.mean(depth)
        depth = normalization_extension(depth)
    
    # Upper face features
    eyebrows_raise = distances(landmarks, 4, 22) + distances(landmarks, 5, 25)
    eyebrows_push_aside = distances_proj(landmarks, 4, 5, 19, 28)    
    if normalized:
        eyebrows_raise = normalization_extension(eyebrows_raise)
        eyebrows_push_aside = normalization_extension(eyebrows_push_aside)
#    eyes_opening = angles(landmarks,19,23,19,21)+angles(landmarks,28,26,28,30)
    eyes_opening = distances(landmarks,21,23) + distances(landmarks,26,30)
    if normalized:
        eyes_opening = normalization_extension(eyes_opening, log = False)
        
    # Lower face features
    mouth_width = distances(landmarks, 31, 37)
    mouth_opening = distances(landmarks, 44, 47)
    lips_protrusion = distances(landmarks,34,44) + distances(landmarks,47,40)
    mouth_smile = angles(landmarks, 35, 37, 31, 33)
    if normalized:
        mouth_width = normalization_extension(mouth_width)
        lips_protrusion = normalization_extension(lips_protrusion)

    features = (yaw, roll, pitch, x, y, depth, 
                eyebrows_raise, eyebrows_push_aside, eyes_opening, 
                mouth_width, mouth_opening, lips_protrusion, mouth_smile)
    features = np.vstack(features)
    return features

"""
----------------------------- Asymmetric Quantity -----------------------------
"""
def asymmetric_quantity(landmarks, normalized = True, depth_norm = False):
    if len(landmarks.shape)==1:
        landmarks = np.expand_dims(landmarks, axis=1)

    # Normalized landmarks
    landmarks, _, _, _ = normalize(landmarks)
    
    eyebrows_inner = distances(landmarks, 4, 22) - distances(landmarks, 5, 25)
    eyebrows_outer = distances(landmarks, 0, 19) - distances(landmarks, 9, 28)
    eyes_opening = distances(landmarks,21,23) - distances(landmarks,26,30)
    if normalized:
        eyebrows_inner = normalization_extension(eyebrows_inner)
        eyebrows_outer = normalization_extension(eyebrows_outer)
        eyes_opening = normalization_extension(eyes_opening, log = False)

    mouth_smile = angles(landmarks,31,33,19,28) - angles(landmarks,37,35,28,19)

    features = (eyebrows_inner, eyebrows_outer, eyes_opening, mouth_smile)
    features = np.vstack(features)
    return features
    
"""
----------------------------- Features Extraction -----------------------------
"""
def expand(vector, timestamps, stimulus, fps = FPS):
    for i in range(len(STIMULI)):
        if stimulus == STIMULI[i]:
            break
    length = LENGTHS[i]
    if len(vector)==0:
        return vector
    increment = 1. / fps
    nb_frame = np.int((length + increment) * fps) + 1
    esp = increment / 2.
    result = np.nan * np.ones((vector.shape[0], nb_frame))
    for i in range(nb_frame):
        begin, end = i * increment - esp, i * increment + esp
        ind = np.logical_and(begin < timestamps, timestamps < end)
        tmp = np.sum(ind)
        if tmp > 1 :
            i = np.argmax(ind)
            ind = timestamps == -1
            ind[i] = True
        if tmp > 0:
            result[:, i] = vector[:, ind].flatten()
    return result

def face_parameterization_extractor(landmarks, timestamps, stimulus):
    param = extract_parametrization(landmarks)
    features = expand(param, timestamps, stimulus)
    return features

FACE_PARAMETERIZATION = ["yaw", "roll", "pitch", "x", "y", "depth",
            "eyebrows raise", "eyebrows push aside", "eyes opening",
            "mouth width", "mouth opening", "lips protrusion", "mouth smile"] 

def face_asymmetry_extraction(landmarks, timestamps, stimulus):
    param = asymmetric_quantity(landmarks)
    features = expand(param, timestamps, stimulus)
    return features

FACE_ASYMMETRY = ["asymmetry inner eyebrows", "asymmetry outer eyebrows", 
                  "asymmetry eyes opening", "asymmetry mouth smile"]
