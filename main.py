 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import numpy as np
from models import DetectionClassifier, dual_score, cross_evaluation
from models import Classifier, dual_score_detection, cross_evaluation_detection
from design_processing import get_standard_design, normalize_design
from design_processing import ind_selection, fast_ind_selection
from helper import save_variables, load_variables

from models import SparseClassifier, SPARSE_CLASSIFIER_HYPER
from models import FeatureElimination, FEATURE_ELIMINATION_HYPER
from models import Svm, SVM_HYPER
from models import NearestNeighbors, NEAREST_NEIGHBORS_HYPER
from models import GenerativeModel, GENERATIVE_MODEL_HYPER
from models import DetectionModel, DETECTION_MODEL_HYPER

"""
------------------------------- Standard Design -------------------------------
"""
design, labels, ids, explanation = get_standard_design(small = True)
design = normalize_design(design)
prob_design, _, _, _ = get_standard_design(small = True, proba = True)

"""
------------------------------ Find Best Methods ------------------------------
"""

to_try = [[SparseClassifier(), SPARSE_CLASSIFIER_HYPER],
          [FeatureElimination(), FEATURE_ELIMINATION_HYPER],
          [Svm(), SVM_HYPER],
          [NearestNeighbors(), NEAREST_NEIGHBORS_HYPER],
          [GenerativeModel(), GENERATIVE_MODEL_HYPER]]

NB = 50
good = []
good_cl = []
best = 0
best_cl = 0
for trial in to_try:
    met, HYPER = trial[0], trial[1]
    for para in HYPER:
        print('here')
        met.set_parameter(para)
        cl = Classifier(met, ind_sel_func = ind_selection)
        res, abst = dual_score(met, design, labels, nb_try = NB)
        res2 = []
        for i in range(10):
            tmp = cross_evaluation(met, design, labels)
            res2.append(tmp)
        res2 = np.mean(res2)
        good.append([res, abst, res2, met, para])
        
        tmp = res*(1-abst) + res2
        if tmp > best:
            best = tmp            
            print('met', res, abst, res2, met, para)
            
        res, abst = dual_score(cl, design, labels, nb_try = NB)
        res2 = []
        for i in range(10):
            tmp = cross_evaluation(cl, design, labels)
            res2.append(tmp)
        res2 = np.mean(res2)
        good_cl.append([res, abst, res2, met, para])

        tmp = res*(1-abst) + res2
        if tmp > best_cl:
            best_cl = tmp            
            print('cl', res, abst, res2, met, para)

met = DetectionModel()
for para in DETECTION_MODEL_HYPER:
    print('here')
    met.set_parameter(para)
    res, abst = dual_score(met, prob_design, labels, nb_try = NB)
    res2 = []
    for i in range(10):
        tmp = cross_evaluation(met, design, labels)
        res2.append(tmp)
    res2 = np.mean(res2)
    good.append([res, abst, res2, 'Detection'])

    tmp = res*(1-abst) + res2
    if tmp > best:
        best = tmp            
        print('met', res, abst, res2, 'Detection')

    res, abst = dual_score(cl, prob_design, labels, nb_try = NB)
    res2 = []
    for i in range(10):
        tmp = cross_evaluation(cl, design, labels)
        res2.append(tmp)
    res2 = np.mean(res2)
    good_cl.append([res, abst, res2, 'Detection'])

    tmp = res*(1-abst) + res2
    if tmp > best_cl:
        best_cl = tmp            
        print('cl', res, abst, res2, 'Detection')

save_variables('final_results', [good, good_cl])


#good, good_cl = load_variables('final_results')
#
#best = 0
#for res in good:
#    tmp = (res[0]*(1-res[1]) + res[2])
#    if tmp > best:
#        best = tmp
#        met = res[3]
#        print(res)
#    
#print('\n\n\n\n')
#best = 0
#for res in good_cl:
#    tmp = (res[0]*(1-res[1]) + res[2])
#    if tmp > best:
#        best = tmp
#        met = res[3]
#        print(res)