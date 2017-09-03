#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:43:34 2017

@author: duke
"""
import numpy as np
from design_processing import get_proba_design, ind_selection
from models import cross_evaluation
from models import DetectionModel
    
    
name = 'second_layer'
#prob_design, labels, ids, explanation = get_proba_design(name)
design, labels, ids, explanation = get_preprocess_design(name)
design = normalize_design(design)
prob_design = proba_design(design)

met = DetectionModel(phi_func= lambda x: x**(-2),
                     psi_func = lambda x: (x < .1).astype(np.int))

res = cross_evaluation(met, prob_design, labels, nb_folds = 5, verbose = True)
print(res, np.mean(res))