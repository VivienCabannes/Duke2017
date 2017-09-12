#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:16:56 2017

@author: duke
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from models import MetricLearningClassifier
from helper import save_fig
from global_path import SAVE_DIR

np.random.seed(0)

N = 100

mu = np.array([[3,0], [0,0]])
Sigma0 = np.array([[.25,.3], [.3, 2]])
Sigma1 = np.array([[.1, 0], [0,2]])

x = np.random.randn(2, N)

distrib0 = np.matmul(Sigma0, (x - np.expand_dims(mu[:,0], axis=1)))
distrib1 = np.matmul(Sigma1, (x - np.expand_dims(mu[:,1], axis=1)))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(distrib0[0], distrib0[1], color = 'b', s = 10)
ax.scatter(distrib1[0], distrib1[1], color = 'y', s = 10)
ax.scatter(0,0,color = 'r', s = 100)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
save_fig(fig, 'silly_example')

design = np.hstack((distrib1, distrib0)).transpose()
labels =  np.hstack((np.ones(N), -np.ones(N)))

cl = MetricLearningClassifier(gamma=10**(-3), alpha=10**(-7), max_it=100)
cl.fit(design, labels, verbose = True)
S = cl.S_tot
        
cur_S = S[0]
x0 = np.matmul(cur_S, distrib0)
x1 = np.matmul(cur_S, distrib1)


fig = plt.figure()
ax = fig.add_subplot(111)
class0 = ax.scatter(x0[0], x0[1], color = 'b', s = 10)
class1 = ax.scatter(x1[0], x1[1], color = 'y', s = 10)
point0 = ax.scatter(0,0,color = 'r', s = 100)
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
        
def update(i):
    cur_S = S[i]
    x0 = np.matmul(cur_S, distrib0)
    x1 = np.matmul(cur_S, distrib1)

    class0.set_offsets(x0.transpose())
    class1.set_offsets(x1.transpose())
    ax.set_xlim(min(np.min(x0[0]), np.min(x1[0]))-.5, 
                max(np.max(x0[0]), np.max(x1[0]))+.5)
    ax.set_ylim(min(np.min(x0[1]), np.min(x1[1]))-1, 
                max(np.max(x0[1]), np.max(x1[1]))+1)
    return class0, class1, ax



anim = animation.FuncAnimation(fig, update, frames=100)
# Save video
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Vivien'),
                extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
save_file = 'silly_example.mp4'
anim.save(os.path.join(SAVE_DIR, save_file), writer=writer)



    