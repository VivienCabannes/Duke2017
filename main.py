#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import numpy as np
from design_processing import get_design, normalize_design, ind_selection
from design_processing import get_preprocess_design
from models import Svm, poly_kernel, MatchingPursuit, NearestNeighbors
from models import SparseClassifier, classifier, DecisionTree, show_tree, split
from models import cross_validation, cross_evaluation, scoring_LOSO
from models import plot_ROC, plot_AP, AUC, linear_Svm
from data_handling import read_MCHAT
import matplotlib.pyplot as plt
#from models import Node, Leaf, Tree

"""
------------------------------- Experimentation -------------------------------
"""
name = 'motor_behaviors'
design, labels, ids, explanation = get_preprocess_design(name)
design = normalize_design(design)

from models import classifier, FeatureElimination, Svm, MatchingPursuit

met = classifier(MatchingPursuit(), ind_sel_func = ind_selection)
res = cross_evaluation(met, design, labels, verbose = True)
print(res, np.mean(res))

#ids_mchat, X, y, mchat_scoring = read_MCHAT()
#mchat = []
#for id_ in ids:
#    for i in range(len(ids_mchat)):
#        if id_==ids_mchat[i]:
#            mchat.append(mchat_scoring[i][2])
#            break
#mchat = np.array(mchat)
#
#
#Thres = [1, .1, .01, .001]
#Thres = [.1]
#NB = [0, 5, 10, 15]       
#NB = [5]
#met1 = Svm(kernel = poly_kernel, sigma = [1,0])
#met2 = MatchingPursuit()
#met3 = NearestNeighbors()
#met4 = SparseClassifier()
#met5 = classifier(method = met1, ind_sel_func = ind_selection)
#met6 = classifier(method = met2, ind_sel_func = ind_selection)
#met7 = classifier(method = met3, ind_sel_func = ind_selection)
#met8 = classifier(method = met4, ind_sel_func = ind_selection)
#Method = [met1]
#
#L = []
#for t in Thres:
#    for n in NB:
#        for m in Method:
#            L.append([split, t, n, m])
#
#cl = DecisionTree(thres = .1, nb_points = 8, lin_method = met1)
#
#res = cross_evaluation(cl, design, labels, evaluation=AUC)
#cl.tree.explain(explanation)
#show_tree(cl.tree, explain = True)
#
#pred, true = scoring_LOSO(cl, design[:50], labels[:50], verbose = True)
#
#plot_ROC(pred, true)
#plot_AP(pred, true)
#print(AUC(pred, true))
#plt.figure()
#a = np.argsort(pred)
#b = mchat[a]
#plt.scatter(np.arange(101), b)
#plt.ylabel('mchat score')
#plt.xlabel('prediction order')