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
RESPONSES = [[2, "bunny_ear1", 4,7], [2, "bunny_ear2", 12,15], 
             [2, "bunny_name",  15,17], [2, "bunny_fall", 25,27], 
             [2, "bunny_pick_up", 35,38], [2, "bunny_ear3", 44,47],
             [2, "bunny_ear4", 51, 54], [2, "bunny_ear5", 58, 61], 
             [4, "tower_construction1", 7,15], [4, "tower_fall1", 17,21], 
             [4, "tower_name", 20, 23], [4, "tower_construction2", 30,38],  
             [4, "tower_fall2", 43,48], [4, "tower_construction3", 55, 65] ]
ALL_RESPONSES=[[STIMULI[i], "total" + str(i), 0, LENGTHS[i]] for i in range(4)]
NB_CLUSTERS = 10
