#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------- Global Variables ------------------------------
"""
import numpy as np

DATA_DIR = '/Users/duke/Desktop/Code/Internship/data'
SAVE_DIR = '/Users/duke/Desktop/Code/Internship/savings'

FPS = 30
LENGTHS = [30.6, 66.8, 69, 30.6]
STIMULI = [1,2,4,6]

FIRST_SCALES = np.logspace(0,3,num=7)
SECOND_SCALES = np.logspace(0,3,num=5)
NB_BINS = 5
ONE_HOT = True
POOLING = True

SMOOTH_SCALE = 10
MOTION_SCALE = 5
SMOOTHING = 50
RESPONSES = [ [2, [[4,7], [12,15], [15,17], [25,27], [35,38], [44,47],
                   [51, 54], [58, 61]] ],
              [4, [[7,15], [17,21], [30,38],  [43,48], [55,65]] ] ]
