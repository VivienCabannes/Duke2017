#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
from global_path import SAVE_DIR
from invert_parameterization import report_average_face
    
face, z_ref = report_average_face()
x_ref, y_ref = face[:49], -face[49:]
depth = True

fig = plt.figure()
if depth:
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=90)
else:
    ax = fig.add_subplot(111)
ax.set_title('Face Parameterization', fontsize = 25)
plt.tight_layout()


def create_face(ax, x, y, z, depth = depth):
    # Create the different object
    tmp_x, tmp_y, tmp_z = x[0:5], y[0:5], z[0:5]
    if depth:
        left_eyebrow, = ax.plot(tmp_x, tmp_z, tmp_y, color='brown')
    else:
        left_eyebrow, = ax.plot(tmp_x, tmp_y, color='brown')
        
    tmp_x, tmp_y, tmp_z = x[5:10], y[5:10], z[5:10]
    if depth:
        right_eyebrow, = ax.plot(tmp_x, tmp_z, tmp_y, color='brown')
    else:
        right_eyebrow, = ax.plot(tmp_x, tmp_y, color='brown')
    tmp_x = np.append(x[10:19], x[13])
    tmp_y = np.append(y[10:19], y[13])
    tmp_z = np.append(z[10:19], z[13])
    if depth:
        nose, = ax.plot(tmp_x, tmp_z, tmp_y, color='black')
    else:
        nose, = ax.plot(tmp_x, tmp_y, color='black')
    tmp_x = np.append(x[19:25], x[19])
    tmp_y = np.append(y[19:25], y[19])
    tmp_z = np.append(z[19:25], z[19])
    if depth:
        left_eye, = ax.plot(tmp_x, tmp_z, tmp_y, color='blue')
    else:
        left_eye, = ax.plot(tmp_x, tmp_y, color='blue')
    tmp_x = np.append(x[25:31], x[25])
    tmp_y = np.append(y[25:31], y[25])
    tmp_z = np.append(z[25:31], z[25])
    if depth:
        right_eye, = ax.plot(tmp_x, tmp_z, tmp_y, color='blue')
    else:
        right_eye, = ax.plot(tmp_x, tmp_y, color='blue')
    tmp_x = np.append(x[31:43], x[31])
    tmp_y = np.append(y[31:43], y[31])
    tmp_z = np.append(z[31:43], z[31])
    if depth:
        mouth, = ax.plot(tmp_x, tmp_z, tmp_y, color='red')
    else:
        mouth, = ax.plot(tmp_x, tmp_y, color='red')
    tmp_x = np.append(x[43:46], x[37])
    tmp_x = np.insert(tmp_x, 0, x[31])
    tmp_y = np.append(y[43:46], y[37])
    tmp_y = np.insert(tmp_y, 0, y[31])
    tmp_z = np.append(z[43:46], z[37])
    tmp_z = np.insert(tmp_z, 0, z[31])
    if depth:
        lip_sup, = ax.plot(tmp_x, tmp_z, tmp_y, color='red')
    else:
        lip_sup, = ax.plot(tmp_x, tmp_y, color='red')
    tmp_x = np.append(x[46:49], x[31])
    tmp_x = np.insert(tmp_x, 0, x[37])
    tmp_y = np.append(y[46:49], y[31])
    tmp_y = np.insert(tmp_y, 0, y[37])
    tmp_z = np.append(z[46:49], z[31])
    tmp_z = np.insert(tmp_z, 0, z[37])
    if depth:
        lip_inf, = ax.plot(tmp_x, tmp_z, tmp_y, color='red')
    else:
        lip_inf, = ax.plot(tmp_x, tmp_y, color='red')
        
    xmean = np.mean(x)
    ymean = np.mean(y)
    xmin = min(-1.2+xmean, np.min(x)-.1)
    xmax = max(1.2+xmean, np.max(x)+.1)
    ymin = min(-1.2+ymean, np.min(y)-.1)
    ymax = max(1.2+ymean, np.max(y)+.1)
    axis_limits = [[xmin, xmax], [ymin, ymax], [0, 2.4]]
    
    return (left_eyebrow, right_eyebrow, nose, left_eye, right_eye, mouth,
            lip_sup, lip_inf), axis_limits
    
face, axis_limits = create_face(ax, x_ref, y_ref, z_ref)

def set_ax_limits(ax, axis_limits, depth = depth):
    if depth:
        ax.set_xlim(axis_limits[0])
        ax.set_zlim(axis_limits[1])
        ax.set_ylim(axis_limits[2])
        ax.set_xticks([])
        ax.set_zticks([])
        ax.set_yticks([])
    else:
        ax.set_xlim(axis_limits[0])
        ax.set_ylim(axis_limits[1])
        ax.set_xticks([])
        ax.set_yticks([])
    
set_ax_limits(ax, axis_limits)

def renew_face(face, x, y, z, depth = depth):
    # Extract object
    face[0].set_xdata(x[0:5])
    face[1].set_xdata(x[5:10])
    face[2].set_xdata(np.append(x[10:19], x[13]))
    face[3].set_xdata(np.append(x[19:25], x[19]))
    face[4].set_xdata(np.append(x[25:31], x[25]))
    face[5].set_xdata(np.append(x[31:43], x[31]))
    tmp_x = np.append(x[43:46], x[37])
    face[6].set_xdata(np.insert(tmp_x, 0, x[31]))
    tmp_x = np.append(x[46:49], x[31])
    face[7].set_xdata(np.insert(tmp_x, 0, x[37]))
    
    if depth:
        face[0].set_ydata(z[0:5])
        face[1].set_ydata(z[5:10])
        face[2].set_ydata(np.append(z[10:19], z[13]))
        face[3].set_ydata(np.append(z[19:25], z[19]))
        face[4].set_ydata(np.append(z[25:31], z[25]))
        face[5].set_ydata(np.append(z[31:43], z[31]))
        tmp_z = np.append(z[43:46], z[37])
        face[6].set_ydata(np.insert(tmp_z, 0, z[31]))
        tmp_z = np.append(z[46:49], z[31])
        face[7].set_ydata(np.insert(tmp_z, 0, z[37]))
        
        face[0].set_3d_properties(y[0:5])
        face[1].set_3d_properties(y[5:10])
        face[2].set_3d_properties(np.append(y[10:19], y[13]))
        face[3].set_3d_properties(np.append(y[19:25], y[19]))
        face[4].set_3d_properties(np.append(y[25:31], y[25]))
        face[5].set_3d_properties(np.append(y[31:43], y[31]))
        tmp_y = np.append(y[43:46], y[37])
        face[6].set_3d_properties(np.insert(tmp_y, 0, y[31]))
        tmp_y = np.append(y[46:49], y[31])
        face[7].set_3d_properties(np.insert(tmp_y, 0, y[37]))
    else:
        face[0].set_ydata(y[0:5])
        face[1].set_ydata(y[5:10])
        face[2].set_ydata(np.append(y[10:19], y[13]))
        face[3].set_ydata(np.append(y[19:25], y[19]))
        face[4].set_ydata(np.append(y[25:31], y[25]))
        face[5].set_ydata(np.append(y[31:43], y[31]))
        tmp_y = np.append(y[43:46], y[37])
        face[6].set_ydata(np.insert(tmp_y, 0, y[31]))
        tmp_y = np.append(y[46:49], y[31])
        face[7].set_ydata(np.insert(tmp_y, 0, y[37]))
        
    xmean = np.mean(x)
    ymean = np.mean(y)
    xmin = min(-1.2+xmean, np.min(x)-.1)
    xmax = max(1.2+xmean, np.max(x)+.1)
    ymin = min(-1.2+ymean, np.min(y)-.1)
    ymax = max(1.2+ymean, np.max(y)+.1)
    axis_limits = [[xmin, xmax], [ymin, ymax], [0, 2.4]]

    return face, axis_limits
    
def rot_mat(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,-sin],[sin,cos]])

def apply_rot(theta, s, t):
    tmp = np.vstack((s, t))
    mat = rot_mat(theta)
    tmp = np.matmul(mat, tmp)
    return tmp[0], tmp[1]

def get_rot_frame(i):
    if i < 10:
        tmp = i/10.
    elif i < 30:
        tmp = 1-(i-10)/10.
    else:
         tmp = (i-30)/10. - 1 
    return tmp * np.pi / 4.

def update_void(i):
    ax.set_title('Face Parameterization', fontsize = 25)
    
def update_yaw(i):
    global face
    theta = get_rot_frame(i)
    tmp = apply_rot(theta, x_ref, z_ref)
    face, axis_limits = renew_face(face, tmp[0], y_ref, tmp[1])
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('8. Yaw', fontsize = 25)
    return face
    
def update_roll(i):
    global face
    theta = get_rot_frame(i)
    tmp = apply_rot(theta, x_ref, y_ref)
    face, axis_limits = renew_face(face, tmp[0], tmp[1], z_ref)
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('9. Roll', fontsize = 25)
    return face

def update_pitch(i):
    global face
    theta = get_rot_frame(i)
    tmp = apply_rot(theta, y_ref, z_ref)
    face, axis_limits = renew_face(face, x_ref, tmp[0], tmp[1])
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('10. Pitch', fontsize = 25)
    return face

def update_xyz(i):
    if i < 30:
        tmp, tmp_z = i/30. + 1.2, 2.4 + i/30.
        axis_limits = [[-tmp, tmp], [-tmp, tmp], [0, tmp_z]]
        set_ax_limits(ax, axis_limits, depth = depth)
        ax.set_title('11. Depth', fontsize = 25)
    elif i < 60:
        tmp = 1.2*(1-abs(i-45)/15)
        axis_limits = [[-2.2+tmp, 2.2+tmp], [-2.2, 2.2], [0, 3.4]]
        set_ax_limits(ax, axis_limits, depth = depth)
        ax.set_title('12. x', fontsize = 25)
    elif i < 90:
        tmp = 1.2*(1-abs(i-75)/15)
        axis_limits = [[-2.2, 2.2], [-2.2+tmp, 2.2+tmp], [0, 3.4]]
        set_ax_limits(ax, axis_limits, depth = depth)
        ax.set_title('13. y', fontsize = 25)
    else:
        tmp, tmp_z = 2.2 - (i-90)/10., 3.4 - (i-90)/10.
        axis_limits = [[-tmp, tmp], [-tmp, tmp], [0, tmp_z]]
        set_ax_limits(ax, axis_limits, depth = depth)
        ax.set_title('Face Parameterization', fontsize = 25)

def update_eyebrows_raise(i):
    global face
    if i <30:
        tmp = 2*i/30.
    elif i < 60:
        tmp = 2 - 3*(i-30)/30.
    else:
        tmp = (i-60)/30. - 1    
    y = np.copy(y_ref)
    ext = .7*np.array([.1,.15,.2,.2,.25])
    y[0:5] += tmp*ext
    y[5:10] += tmp*ext[::-1]
    face, axis_limits = renew_face(face, x_ref, y, z_ref)
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('1. Raise Eyebrows', fontsize = 25)
    return face

def update_eyebrows_push_aside(i):
    global face
    if i <30:
        tmp = -i/30.
    elif i < 60:
        tmp = -1 + 1.5*(i-30)/30.
    else:
        tmp = -.5*(i-60)/30. + .5    
    x = np.copy(x_ref)
    ext = .7*np.array([.1,.15,.2,.2,.25])
    x[0:5] -= tmp*ext
    x[5:10] += tmp*ext[::-1]
    face, axis_limits = renew_face(face, x, y_ref, z_ref)
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('2. Push Eyebrows Aside', fontsize = 25)
    return face

def update_eyes_opening(i):
    global face
    if i <30:
        tmp = i/30. + 1
    elif i < 60:
        tmp = 2 - 2*(i-30)/30.
    else:
        tmp = (i-60)/30.
    y = np.copy(y_ref)
    ext1 = (y_ref[21] - y_ref[23])/2
    mid1 = (y_ref[21] + y_ref[23])/2
    ext2 = (y_ref[20] - y_ref[24])/2
    mid2 = (y_ref[20] + y_ref[24])/2
    y[21] = mid1 + tmp*ext1
    y[23] = mid1 - tmp*ext1
    y[20] = mid2 + tmp*ext2
    y[24] = mid2 - tmp*ext2 
    ext1 = (y_ref[26] - y_ref[30])/2
    mid1 = (y_ref[26] + y_ref[30])/2
    ext2 = (y_ref[27] - y_ref[29])/2
    mid2 = (y_ref[27] + y_ref[29])/2
    y[26] = mid1 + tmp*ext1
    y[30] = mid1 - tmp*ext1
    y[27] = mid2 + tmp*ext2
    y[29] = mid2 - tmp*ext2 
    face, axis_limits = renew_face(face, x_ref, y, z_ref)
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('3. Eyes Opening', fontsize = 25)
    return face

def update_mouth_width(i):
    global face
    if i <30:
        tmp = .25*i/30. + 1
    elif i < 60:
        tmp = 1.25 - .5*(i-30)/30.
    else:
        tmp = .25*(i-60)/30. + .75  
    x = np.copy(x_ref)
    x[31:] *= tmp
    face, axis_limits = renew_face(face, x, y_ref, z_ref)
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('4. Mouth Width', fontsize = 25)
    return face

def update_mouth_opening(i):
    global face
    if i <30:
        tmp = i/30.
    elif i < 60:
        tmp = 1 - 2*(i-30)/30.
    else:
        tmp = (i-60)/30. - 1 
    y = np.copy(y_ref)

    middle = (y[43:46]+y[46:])/2
    ext = (y[43:46]-y[46:])/2
    y[43:46] = middle + (10**tmp)*ext 
    y[46:] = middle - (10**tmp)*ext  

    lip_up = y_ref[33:36] - y_ref[43:46]
    y[33:36] = lip_up + y[43:46]    
    lip_down = y_ref[46:] - y_ref[39:42] 
    y[39:42] = y[46:] - lip_down 
    
    ratio = (y_ref[32] - y_ref[31])/ (y_ref[33] - y_ref[31])
    y[32] = ratio*(y[33] - y[31]) + y[31]
    ratio = (y_ref[36] - y_ref[37])/ (y_ref[35] - y_ref[37])
    y[36] = ratio*(y[35] - y[37]) + y[37]
    ratio = (y_ref[42] - y_ref[31])/ (y_ref[41] - y_ref[31])
    y[42] = ratio*(y[41] - y[31]) + y[31]
    ratio = (y_ref[38] - y_ref[37])/ (y_ref[39] - y_ref[37])
    y[38] = ratio*(y[39] - y[37]) + y[37]
    
    if tmp > 0:
        ext = 1 - .7*tmp
        dep = y[16] - y[34] - ext * (y_ref[16] - y_ref[34])
        y[31:] += dep

    face, axis_limits = renew_face(face, x_ref, y, z_ref)
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('5. Mouth Opening', fontsize = 25)
    return face

def update_lips_protrusion(i):
    global face
    if i <30:
        tmp = 1+.5*i/30
    elif i < 60:
        tmp = 1.5 - 1.25 * (i-30)/30
    else:
        tmp = .75*(i-60)/30 + .25
    y = np.copy(y_ref)
    tmp_y = np.mean(y[31:])
    y[31:] = tmp*(y[31:] - tmp_y) + tmp_y
    face, axis_limits = renew_face(face, x_ref, y, z_ref)
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('6. Lips Protrusion', fontsize = 25)
    return face

def update_mouth_smile(i):
    global face
    if i <30:
        tmp = i/30.
    elif i < 60:
        tmp = 1 - 2*(i-30)/30.
    else:
        tmp = (i-60)/30. - 1 
    y = np.copy(y_ref)
    y[31] += .25*tmp
    y[37] += .25*tmp
    def update_with_ratio(i0, i1, i2):
        ratio = (y_ref[i0] - y_ref[i1])/ (y_ref[i2] - y_ref[i1])
        y[i0] = ratio*(y[i2] - y[i1]) + y[i1]
    update_with_ratio(32, 31, 34)
    update_with_ratio(33, 31, 34)
    update_with_ratio(35, 37, 34)
    update_with_ratio(36, 37, 34)
    update_with_ratio(38, 37, 40)
    update_with_ratio(39, 37, 40)
    update_with_ratio(41, 31, 40)
    update_with_ratio(42, 31, 40)
    face, axis_limits = renew_face(face, x_ref, y, z_ref)
    set_ax_limits(ax, axis_limits, depth = depth)
    ax.set_title('7. Smiling', fontsize = 25)
    return face

def update(i):
    """ Combine all regarding time frame """
    if i < 30:
        update_void(i)
    elif i < 120:
        update_eyebrows_raise(i-30)
    elif i < 210:
        update_eyebrows_push_aside(i-120)
    elif i < 300:
        update_eyes_opening(i-210)
    elif i < 390:
        update_mouth_width(i-300)
    elif i < 480:
        update_mouth_opening(i-390)
    elif i < 570:
        update_lips_protrusion(i-480)
    elif i < 660:
        update_mouth_smile(i-570)
    elif i < 700:
        update_yaw(i-660)
    elif i < 740:
        update_roll(i-700)
    elif i < 780:
        update_pitch(i-740)
    elif i < 880:
        update_xyz(i-780)

if True:
    anim = animation.FuncAnimation(fig, update, frames=880)
    # Save video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Vivien'),
                    extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])
    save_file = 'face_para.mp4'
    anim.save(os.path.join(SAVE_DIR, save_file), writer=writer)
            