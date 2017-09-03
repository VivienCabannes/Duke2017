#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:53:22 2017

@author: duke
"""

"""
------------------------------ Show all Features ------------------------------
"""
def band_visualization(stimulus, patient_id, scale_space=True, mode='same',
                       begin = 0, end = None, ax=None, vmax=None, vmin=None):
    landmarks, timestamps = get_data(stimulus, patient_id)
    features = extractor(landmarks, timestamps, stimulus, scale_space, mode)
    def f():
        return report_normalization(scale_space)
    features = homogenize_features(features, f)
    features = features[:, begin:end]
    ctl = False
    if ax is None:
        ctl = True
        fig = plt.figure()
        ax = fig.add_subplot(111)
    surf = ax.pcolor(features, cmap='Blues', linewidth=0, 
                     antialiased=False, vmax=vmax, vmin=vmin)
    if scale_space:
        tmp = len(features) / 13
        ax.set_yticks(tmp * (np.arange(13) + 0.5), minor=False)
    else:
        ax.set_yticks(np.arange(13) + 0.5, minor=False)
    ax.set_yticklabels(FEATURES)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if ctl:
        return fig, ax, surf
    else:
        return surf

if __name__=='__main__' and False:
    from helper import save_fig
    all_ids, all_labels = get_ids(data_set = 'all')
    begin, end = 20*30, 23*30
#    begin, end = 0, None
    for stimulus in [4]:
#    for stimulus in [1, 2, 4, 6]:
        for i in range(40,50):
#        for i in range(len(all_ids)):
            patient_id = all_ids[i]
            label = all_labels[i]
            try:
                fig, ax, _ = band_visualization(stimulus, patient_id, 
                                                scale_space = True,
                                                begin = begin, end = end,
                                                vmin=-.6, vmax=.6)
                if label == 1:
                    tmp = 'Autistic'
                elif label == -1:
                    tmp = 'Normal'
                else:
                    tmp = 'Unknown'
                ax.set_title('%s, %d, %s' %(patient_id, stimulus, tmp))
                plt.tight_layout()
                
                tmp = patient_id + '_' + str(stimulus) + '_' + tmp
                save_fig(fig, save_file = tmp)
                
                plt.close()
            except FileNotFoundError:
                tmp='Patient %s, stimulus %d not found'%(patient_id, stimulus)
                print(tmp)
                continue


if __name__=='__main__' and False:
    from mpl_toolkits.mplot3d import Axes3D
    
    def get_points(stimulus, patient_id, n=0):
        landmarks, timestamps = get_data(stimulus, patient_id)
        features = features_extraction(landmarks, timestamps, stimulus)
        features, H = preprocess_signal(features)
        features = homogenize_features(features)
        if n>0:
            features = increments(features, n=n)
        mean = np.mean(features, axis=1)
        var = np.var(features, axis=1)
        cumul = np.mean(np.abs(features), axis=1)
        return mean, var, cumul

    def get_color(label, stimulus):
        if label == 1:
            color = 'b'
        elif label == -1:
            color = 'r'
        else:
            color = 'g'
        if stimulus == 1:
            marker = 'o'
        elif stimulus == 2:
            marker = '^'
        elif stimulus == 4:
            marker = 's'
        else:
            marker = '*'
        return marker, color
    
    figs = []
    for i in range(13):
        fig = plt.figure()
        figs.append(fig.add_subplot(111, projection='3d'))
        figs[i].set_title(FEATURES[i])
    for stimulus in STIMULI:
        for i in range(len(all_ids)):
            patient_id, label = all_ids[i], all_labels[i]
            try:
                x, y, z = get_points(stimulus, patient_id)
            except FileNotFoundError:
                continue
            marker, color = get_color(label, stimulus)
            for k in range(13):
                figs[k].scatter(x[k], y[k], z[k], c=color, marker=marker)

