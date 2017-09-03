#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---------------------------------- Libraries ---------------------------------- 
"""
import numpy as np
import math 

"""
------------------------------ Helper Functions -------------------------------
"""    
def angles(landmarks, pt1, middle, pt2):
    x1 = landmarks[pt1, :] - landmarks[middle, :]
    x2 = landmarks[pt2, :] - landmarks[middle, :]
    y1 = landmarks[pt1+49, :] - landmarks[middle+49, :]
    y2 = landmarks[pt2+49, :] - landmarks[middle+49, :]
    angles = (x1*x2 + y1*y2) / (((x1**2 + y1**2)**.5)*((x2**2 + y2**2)**.5))
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

"""
------------------------- Invert the parameterization -------------------------
"""
def convert_revert(i):
    if i < 10:
        j = 9-i
    elif 9 < i < 14:
        j = i
    elif 13 < i < 19:
        j = 32 - i
    elif 18 < i < 23 or 24 < i < 29:
        j = 47 - i
    elif 22 < i < 25 or 28 < i < 31:
        j = 53 - i
    elif 30 < i < 38:
        j = 68 - i
    elif 37 < i < 43:
        j = 80 - i
    elif 42 < i < 46:
        j = 88 - i
    else:
        j = 94 - i
    return j

def symetrize(landmarks):
    tmp = np.copy(landmarks)
    tmp[:49, :] *= -1
    landmarks_revert = np.zeros(landmarks.shape)
    for i in range(49):
        j = convert_revert(i)
        landmarks_revert[i,:] = tmp[j,:]
        landmarks_revert[i+49,:] = tmp[j+49,:]
    return landmarks_revert
    
def report_average_face(verbose = False):
    face = np.zeros(98)
    face[0] = -0.952549397945
    face[1] = -0.826879560947
    face[2] = -0.664468050003
    face[3] = -0.492884621024
    face[4] = -0.31603872776
    face[5] = 0.31603872776
    face[6] = 0.492884621024
    face[7] = 0.664468050003
    face[8] = 0.826879560947
    face[9] = 0.952549397945
    face[10] = 0.0
    face[11] = 0.0
    face[12] = 0.0
    face[13] = 0.0
    face[14] = -0.212071061134
    face[15] = -0.108363457024
    face[16] = 0.0
    face[17] = 0.108363457024
    face[18] = 0.212071061134
    face[19] = -0.745798766613
    face[20] = -0.605419933796
    face[21] = -0.443957984447
    face[22] = -0.312523543835
    face[23] = -0.457851871848
    face[24] = -0.614739000797
    face[25] = 0.312523543835
    face[26] = 0.443957984447
    face[27] = 0.605419933796
    face[28] = 0.745798766613
    face[29] = 0.614739000797
    face[30] = 0.457851871848
    face[31] = -0.400816306472
    face[32] = -0.286880642176
    face[33] = -0.153368100524
    face[34] = 0.0
    face[35] = 0.153368100524
    face[36] = 0.286880642176
    face[37] = 0.400816306472
    face[38] = 0.291254475713
    face[39] = 0.162526160479
    face[40] = 0.0
    face[41] = -0.162526160479
    face[42] = -0.291254475713
    face[43] = -0.18249604851
    face[44] = 0.0
    face[45] = 0.18249604851
    face[46] = 0.180189713836
    face[47] = 0.0
    face[48] = -0.180189713836
    face[49] = -0.662028372288
    face[50] = -0.765388160944
    face[51] = -0.804332405329
    face[52] = -0.796701550484
    face[53] = -0.761949062347
    face[54] = -0.761949062347
    face[55] = -0.796701550484
    face[56] = -0.804332405329
    face[57] = -0.765388160944
    face[58] = -0.662028372288
    face[59] = -0.405257403851
    face[60] = -0.255039170384
    face[61] = -0.103482171893
    face[62] = 0.0448309034109
    face[63] = 0.261723458767
    face[64] = 0.274327129126
    face[65] = 0.280901789665
    face[66] = 0.274327129126
    face[67] = 0.261723458767
    face[68] = -0.387672483921
    face[69] = -0.475479468703
    face[70] = -0.469970226288
    face[71] = -0.353144690394
    face[72] = -0.313445031643
    face[73] = -0.31803587079
    face[74] = -0.353144690394
    face[75] = -0.469970226288
    face[76] = -0.475479468703
    face[77] = -0.387672483921
    face[78] = -0.31803587079
    face[79] = -0.313445031643
    face[80] = 0.651566624641
    face[81] = 0.549032092094
    face[82] = 0.469267964363
    face[83] = 0.478075906634
    face[84] = 0.469267964363
    face[85] = 0.549032092094
    face[86] = 0.651566624641
    face[87] = 0.749098598957
    face[88] = 0.805266559124
    face[89] = 0.824949622154
    face[90] = 0.805266559124
    face[91] = 0.749098598957
    face[92] = 0.58447933197
    face[93] = 0.582112550735
    face[94] = 0.58447933197
    face[95] = 0.656750559807
    face[96] = 0.668041408062
    face[97] = 0.656750559807
    
    z = 1.5*np.ones(49)
    z[0] -= .35
    z[1] -= .2
    z[2] -= .1
    z[3] -= .05
    z[6] -= .05
    z[7] -= .1
    z[8] -= .2
    z[9] -= .35
    z[10] += .1 
    z[11] += .15
    z[12] += .2
    z[13] += .3
    z[14] -= .02 
    z[15] -= .01 
    z[17] -= .01 
    z[18] -= .02 
    z[19] -= .35
    z[20] -= .3
    z[21] -= .2
    z[22] -= .25
    z[23] -= .2
    z[24] -= .3
    z[25] -= .25
    z[26] -= .2
    z[27] -= .3
    z[28] -= .35
    z[29] -= .3
    z[30] -= .2 
    z[31] -= .2 
    z[32] -= .1 
    z[34] += .1
    z[36] -= .1 
    z[37] -= .2 
    z[38] -= .1 
    z[40] += .05 
    z[42] -= .1 
    z[43] -= .1 
    z[45] -= .1
    z[46] -= .1
    z[48] -= .1     
    return face, z

def extract_face_info():
    face, _ = report_average_face()
    face = np.expand_dims(face, axis=1)
    eb_i_rest = distances(face, 4, 22)
    eb_o_rest = distances(face, 0, 19)
    eb_d_rest = distances_proj(face, 4, 5, 19, 28)
    m_w_rest = distances(face, 31, 37)
    l_u_rest = distances(face, 34, 44)
    l_d_rest = distances(face, 47, 40)
    theta_l_u = math.acos(angles(face, 21, 19, 22))
    theta_l_d = math.acos(angles(face, 23, 19, 22))
    theta_r_u = math.acos(angles(face, 26, 28, 25))
    theta_r_d = math.acos(angles(face, 30, 28, 25))
    m_n = distances(face, 34, 16)
    features = (eb_i_rest, eb_o_rest, eb_d_rest, m_w_rest, l_u_rest, l_d_rest,
                theta_l_u, theta_l_d, theta_r_u, theta_r_d, m_n)
    features = np.vstack(features)
    return features    
    
def my_matmul(A, x):
    """
    Trick to use parallele matrix multiplication
    A[i, j, frame]
    x[pt, i, frame]
    y[j, pt, frame]
    """
    x = x.swapaxes(0,2)
    A = A.swapaxes(0,2)
    y = np.matmul(A, x).swapaxes(0,2) 
    return y

def adjusting_others(x_pt, y_pt, roll):
    cos_roll = np.cos(np.arctan(roll))
    sin_roll = np.sin(np.arctan(roll))
    roll_rot_mat = np.array([[cos_roll,-sin_roll],[sin_roll,cos_roll]])
    x_pt = np.expand_dims(x_pt, axis=1)
    y_pt = np.expand_dims(y_pt, axis=1)
    tmp = my_matmul(roll_rot_mat, np.concatenate((x_pt, y_pt), axis=1))
    x_pt = tmp[:,0,:]
    y_pt = tmp[:,1,:]

    #The best is probably to do it from the average face
    # Outer part of the upper lips
    w = .5; x_pt[32,:] = (w*x_pt[31,:] + (1-w)*x_pt[33])
    w = .3; y_pt[32,:] = (w*y_pt[31,:] + (1-w)*y_pt[33])
    w = .5; x_pt[36,:] = (w*x_pt[37,:] + (1-w)*x_pt[35])
    w = .3; y_pt[36,:] = (w*y_pt[37,:] + (1-w)*y_pt[35])
    
    # Outer part of the lower lips
    w = .6; x_pt[41,:] = (w*x_pt[40,:] + (1-w)*x_pt[31,:])
    w = .8; y_pt[41,:] = (w*y_pt[40,:] + (1-w)*y_pt[31,:])
    w = .3; x_pt[42,:] = (w*x_pt[40,:] + (1-w)*x_pt[31,:])
    w = .5; y_pt[42,:] = (w*y_pt[40,:] + (1-w)*y_pt[31,:])
    w = .6; x_pt[39,:] = (w*x_pt[40,:] + (1-w)*x_pt[37,:])
    w = .8; y_pt[39,:] = (w*y_pt[40,:] + (1-w)*y_pt[37,:])
    w = .3; x_pt[38,:] = (w*x_pt[40,:] + (1-w)*x_pt[37,:])
    w = .5; y_pt[38,:] = (w*y_pt[40,:] + (1-w)*y_pt[37,:])
    
    # Inner part of the upper lip
    w = .5; x_pt[43,:] = (w*x_pt[31,:] + (1-w)*x_pt[44,:])
    w = .3; y_pt[43,:] = (w*y_pt[31,:] + (1-w)*y_pt[44,:])
    w = .5; x_pt[45,:] = (w*x_pt[37,:] + (1-w)*x_pt[44,:])
    w = .3; y_pt[45,:] = (w*y_pt[37,:] + (1-w)*y_pt[44,:])
    
    # Outer part of the upper lip
    w = .5; x_pt[46,:] = (w*x_pt[37,:] + (1-w)*x_pt[47,:])
    w = .3; y_pt[46,:] = (w*y_pt[37,:] + (1-w)*y_pt[47,:])
    w = .5; x_pt[48,:] = (w*x_pt[31,:] + (1-w)*x_pt[47,:])
    w = .3; y_pt[48,:] = (w*y_pt[31,:] + (1-w)*y_pt[47,:])
    
    # Adjusting the rest of the eyebrows
    x_pt[1,:], y_pt[1,:]=(x_pt[0,:]+x_pt[2,:])/2., (y_pt[0,:]+2*y_pt[2,:])/3. 
    x_pt[3,:], y_pt[3,:]=(x_pt[2,:]+x_pt[4,:])/2., (y_pt[4,:]+2*y_pt[2,:])/3. 
    x_pt[6,:], y_pt[6,:]=(x_pt[5,:]+x_pt[7,:])/2., (y_pt[5,:]+2*y_pt[7,:])/3. 
    x_pt[8,:], y_pt[8,:]=(x_pt[9,:]+x_pt[7,:])/2., (y_pt[9,:]+2*y_pt[7,:])/3. 

    roll_rot_mat = np.array([[cos_roll,sin_roll],[-sin_roll,cos_roll]])
    x_pt = np.expand_dims(x_pt, axis=1)
    y_pt = np.expand_dims(y_pt, axis=1)
    tmp = my_matmul(roll_rot_mat, np.concatenate((x_pt, y_pt), axis=1))
    x_pt = tmp[:,0,:]
    y_pt = tmp[:,1,:]
    
    return x_pt, y_pt

def reconstruction(features):
    """ Extract Features """
    eb_l_i, eb_l_o, eb_l_a = features[0], features[1], features[2] 
    eb_d, eb_r_i, eb_r_o = features[3], features[4], features[5]
    eb_r_a, e_l_a, e_r_a = features[6], features[7], features[8]
    yaw, roll, pitch = features[9], features[10], features[11]
    x, y, depth = features[12], features[13], features[14]
    m_w, m_h = features[16], features[17]
    l_u, l_d = features[18], features[19]
    m_l_a, m_r_a, m_n = features[20], features[21], features[22]
    
    """ Create average face to deform """
    face, z_pt = report_average_face()
    x_pt = face[:49]
    y_pt = -face[49:]
    nb_frame = features.shape[1]
    x_pt = np.tile(x_pt, (nb_frame,1,1)).swapaxes(0,2)
    y_pt = np.tile(y_pt, (nb_frame,1,1)).swapaxes(0,2)
    z_pt = np.tile(z_pt, (nb_frame,1,1)).swapaxes(0,2)
    
    """ Head pose: rotate according to roll, pitch and yaw """
    # Yaw
    tmp = 2*yaw
    tmp[tmp>1] = 1
    tmp[tmp<-1] = -1
    cos_yaw = np.cos(np.arcsin(tmp))
    sin_yaw = tmp
    yaw_rot_mat = np.array([[cos_yaw,-sin_yaw],[sin_yaw,cos_yaw]])
    tmp = my_matmul(yaw_rot_mat, np.concatenate((x_pt, z_pt), axis=1))
    x_pt = np.expand_dims(tmp[:,0,:], axis=1)
    z_pt = np.expand_dims(tmp[:,1,:], axis=1)
    
    # Pitch
    tmp = .75*pitch
    tmp[tmp>1] = 1
    tmp[tmp<-1] = -1
    cos_pitch = np.cos(np.arcsin(tmp))
    sin_pitch = tmp
    pitch_rot_mat = np.array([[cos_pitch,sin_pitch],[-sin_pitch,cos_pitch]])
    tmp = my_matmul(pitch_rot_mat, np.concatenate((z_pt, y_pt), axis=1))
    y_pt = np.expand_dims(tmp[:,1,:], axis=1)
    
    # Roll
    cos_roll = np.cos(np.arctan(roll))
    sin_roll = np.sin(np.arctan(roll))
    roll_rot_mat = np.array([[cos_roll,sin_roll],[-sin_roll,cos_roll]])
    tmp = my_matmul(roll_rot_mat, np.concatenate((x_pt, y_pt), axis=1))
    x_pt = tmp[:,0,:]
    y_pt = tmp[:,1,:]
        
    """ Distance to put on the face """
    features = extract_face_info()
    eb_i_rest, eb_o_rest, eb_d_rest = features[0], features[1], features[2]
    m_w_rest, l_u_rest, l_d_rest = features[3], features[4], features[5]
    theta_l_u, theta_l_d = features[6], features[7]
    theta_r_u, theta_r_d, m_n_rest = features[8], features[9], features[10]
    eb_l_i = (2 ** eb_l_i) * eb_i_rest
    eb_l_o = (2 ** eb_l_o) * eb_o_rest
    eb_d = (2 ** eb_d) * eb_d_rest
    eb_r_i = (2 ** eb_r_i) * eb_i_rest
    eb_r_o = (2 ** eb_r_o) * eb_o_rest
    m_w = (2 ** m_w) * m_w_rest
    l_u = (2 ** l_u) * l_u_rest
    l_d = (2 ** l_d) * l_d_rest
    m_n = (2 ** m_n) * m_n_rest
    
    """ Muscular action one by one """
    # Eyebrows
    # Distance between them
    x_tmp, y_tmp = x_pt[5,:] - x_pt[4,:], y_pt[5,:] - y_pt[4,:]
    current_d = (x_tmp**2+y_tmp**2)**.5
    x_elong = (eb_d / current_d - 1) * x_tmp
    y_elong = (eb_d / current_d - 1) * y_tmp
    x_pt[0:5,:] -= x_elong/2.
    y_pt[0:5,:] -= y_elong/2.
    x_pt[5:10,:] += x_elong/2.
    y_pt[5:10,:] += y_elong/2.
    
    # Raising inner part
    u, v = -y_tmp, x_tmp
    i, j = x_pt[4,:], y_pt[4,:]
    r, s = x_pt[22,:], y_pt[22,:]
    a = u**2 + v**2
    b = (u*(i-r) + v*(j-s))
    c = (i-r)**2 + (j-s)**2 - eb_l_i**2
    discr = (b**2 - a*c)**.5
    elong = (-b + discr) / a
    elong[np.isnan(elong)] = 0
    x_pt[1:5,:] += elong*u
    y_pt[1:5,:] += elong*v
    i, j = x_pt[5,:], y_pt[5,:]
    r, s = x_pt[25,:], y_pt[25,:]
    a = u**2 + v**2
    b = (u*(i-r) + v*(j-s))
    c = (i-r)**2 + (j-s)**2 - eb_r_i**2
    discr = (b**2 - a*c)**.5
    elong = (-b + discr) / a
    elong[np.isnan(elong)] = 0
    x_pt[5:9,:] += elong*u
    y_pt[5:9,:] += elong*v
    
    # Raising outer part
    x_tmp, y_tmp = x_pt[0,:] - x_pt[19,:], y_pt[0,:] - y_pt[19,:]
    current_d = (x_tmp**2+y_tmp**2)**.5
    x_pt[0,:] = eb_l_o * x_tmp / current_d + x_pt[19,:]
    y_pt[0,:] = eb_l_o * y_tmp / current_d + y_pt[19,:]
    x_tmp, y_tmp = x_pt[9,:] - x_pt[28,:], y_pt[9,:] - y_pt[28,:]
    current_d = (x_tmp**2+y_tmp**2)**.5
    x_pt[9,:] = eb_r_o * x_tmp / current_d + x_pt[28,:]
    y_pt[9,:] = eb_r_o * y_tmp / current_d + y_pt[28,:]
    
    # Raising the middle part
    x_1, x_2, y_1, y_2 = x_pt[0,:], x_pt[4,:], y_pt[0,:], y_pt[4,:]
    u, v = y_1 - y_2, x_2 - x_1
    d = (u**2+v**2)**.5 
    retract = (2*np.tan(np.arccos(eb_l_a)/2))
    x_pt[2,:] = (x_1+x_2)/2 + u / retract
    y_pt[2,:] = (y_1+y_2)/2 + v / retract
    x_1, x_2, y_1, y_2 = x_pt[5,:], x_pt[9,:], y_pt[5,:], y_pt[9,:]
    u, v = y_1 - y_2, x_2 - x_1
    d = (u**2+v**2)**.5 
    retract = (2*np.tan(np.arccos(eb_r_a)/2))
    x_pt[7,:] = (x_1+x_2)/2 + u / retract
    y_pt[7,:] = (y_1+y_2)/2 + v / retract
        
    # Eyes
    theta = np.arccos(e_l_a)
    theta_u = theta / (1 + theta_l_d / theta_l_u )
    theta_d = theta - theta_u
    elong_u = np.tan(theta_u) / np.tan(theta_l_u) 
    elong_d = np.tan(theta_d) / np.tan(theta_l_d) 
    
    u_x, u_y = x_pt[21,:] - x_pt[19,:], y_pt[21,:] - y_pt[19,:] 
    v_x, v_y = x_pt[22,:] - x_pt[19,:], y_pt[22,:] - y_pt[19,:]
    d = v_x**2 + v_y**2 
    ps = u_x*v_x + u_y*v_y
    v_x *= ps / d
    v_y *= ps / d
    v_x += x_pt[19,:]
    v_y += y_pt[19,:]
    x_pt[21,:] = v_x + elong_u * (x_pt[21,:]-v_x)
    y_pt[21,:] = v_y + elong_u * (y_pt[21,:]-v_y)
    
    u_x, u_y = x_pt[23,:] - x_pt[19,:], y_pt[23,:] - y_pt[19,:] 
    v_x, v_y = x_pt[22,:] - x_pt[19,:], y_pt[22,:] - y_pt[19,:]
    d = v_x**2 + v_y**2 
    ps = u_x*v_x + u_y*v_y
    v_x *= ps / d
    v_y *= ps / d
    v_x += x_pt[19,:]
    v_y += y_pt[19,:]
    x_pt[23,:] = v_x + elong_d * (x_pt[23,:]-v_x)
    y_pt[23,:] = v_y + elong_d * (y_pt[23,:]-v_y)
    
    u_x, u_y = x_pt[20,:] - x_pt[22,:], y_pt[20,:] - y_pt[22,:] 
    v_x, v_y = x_pt[19,:] - x_pt[22,:], y_pt[19,:] - y_pt[22,:]
    d = v_x**2 + v_y**2 
    ps = u_x*v_x + u_y*v_y
    v_x *= ps / d
    v_y *= ps / d
    v_x += x_pt[22,:]
    v_y += y_pt[22,:]
    x_pt[20,:] = v_x + elong_u * (x_pt[20,:]-v_x)
    y_pt[20,:] = v_y + elong_u * (y_pt[20,:]-v_y)
    
    u_x, u_y = x_pt[24,:] - x_pt[22,:], y_pt[24,:] - y_pt[22,:] 
    v_x, v_y = x_pt[19,:] - x_pt[22,:], y_pt[19,:] - y_pt[22,:]
    d = v_x**2 + v_y**2 
    ps = u_x*v_x + u_y*v_y
    v_x *= ps / d
    v_y *= ps / d
    v_x += x_pt[22,:]
    v_y += y_pt[22,:]
    x_pt[24,:] = v_x + elong_d * (x_pt[24,:]-v_x)
    y_pt[24,:] = v_y + elong_d * (y_pt[24,:]-v_y)
    
    theta = np.arccos(e_r_a)
    theta_u = theta / (1 + theta_r_d / theta_r_u )
    theta_d = theta - theta_u
    elong_u = np.tan(theta_u) / np.tan(theta_r_u) 
    elong_d = np.tan(theta_d) / np.tan(theta_r_d) 
    
    u_x, u_y = x_pt[26,:] - x_pt[28,:], y_pt[26,:] - y_pt[28,:] 
    v_x, v_y = x_pt[25,:] - x_pt[28,:], y_pt[25,:] - y_pt[28,:]
    d = v_x**2 + v_y**2 
    ps = u_x*v_x + u_y*v_y
    v_x *= ps / d
    v_y *= ps / d
    v_x += x_pt[28,:]
    v_y += y_pt[28,:]
    x_pt[26,:] = v_x + elong_u * (x_pt[26,:]-v_x)
    y_pt[26,:] = v_y + elong_u * (y_pt[26,:]-v_y)
    
    u_x, u_y = x_pt[30,:] - x_pt[28,:], y_pt[30,:] - y_pt[28,:] 
    v_x, v_y = x_pt[25,:] - x_pt[28,:], y_pt[25,:] - y_pt[28,:]
    d = v_x**2 + v_y**2 
    ps = u_x*v_x + u_y*v_y
    v_x *= ps / d
    v_y *= ps / d
    v_x += x_pt[28,:]
    v_y += y_pt[28,:]
    x_pt[30,:] = v_x + elong_d * (x_pt[30,:]-v_x)
    y_pt[30,:] = v_y + elong_d * (y_pt[30,:]-v_y)
    
    u_x, u_y = x_pt[27,:] - x_pt[25,:], y_pt[27,:] - y_pt[25,:] 
    v_x, v_y = x_pt[28,:] - x_pt[25,:], y_pt[28,:] - y_pt[25,:]
    d = v_x**2 + v_y**2 
    ps = u_x*v_x + u_y*v_y
    v_x *= ps / d
    v_y *= ps / d
    v_x += x_pt[25,:]
    v_y += y_pt[25,:]
    x_pt[27,:] = v_x + elong_u * (x_pt[27,:]-v_x)
    y_pt[27,:] = v_y + elong_u * (y_pt[27,:]-v_y)
    
    u_x, u_y = x_pt[29,:] - x_pt[25,:], y_pt[29,:] - y_pt[25,:] 
    v_x, v_y = x_pt[28,:] - x_pt[25,:], y_pt[28,:] - y_pt[25,:]
    d = v_x**2 + v_y**2 
    ps = u_x*v_x + u_y*v_y
    v_x *= ps / d
    v_y *= ps / d
    v_x += x_pt[25,:]
    v_y += y_pt[25,:]
    x_pt[29,:] = v_x + elong_d * (x_pt[29,:]-v_x)
    y_pt[29,:] = v_y + elong_d * (y_pt[29,:]-v_y)
    
    # Mouth
    # Extension
    x_1, y_1 = x_pt[37,:] - x_pt[31,:], y_pt[37,:] - y_pt[31,:]
    x_2, y_2 = (x_pt[37,:] + x_pt[31,:])/2, (y_pt[37,:] + y_pt[31,:])/2
    current_d = (x_1**2 + y_1**2) **.5
    elong = m_w / current_d
    x_pt[31,:] = x_2 - elong * x_1 / 2
    y_pt[31,:] = y_2 - elong * y_1 / 2
    x_pt[37,:] = x_2 + elong * x_1 / 2
    y_pt[37,:] = y_2 + elong * y_1 / 2
    
    x_1, y_1 = x_pt[35,:] - x_pt[33,:], y_pt[35,:] - y_pt[33,:]
    x_2, y_2 = (x_pt[35,:] + x_pt[33,:])/2, (y_pt[35,:] + y_pt[33,:])/2
    x_pt[33,:] = x_2 - elong * x_1 / 2
    y_pt[33,:] = y_2 - elong * y_1 / 2
    x_pt[35,:] = x_2 + elong * x_1 / 2
    y_pt[35,:] = y_2 + elong * y_1 / 2
        
    # Smiling
    theta_l = -np.arcsin(m_l_a) + np.arctan(roll)
    theta_r = -np.arcsin(m_r_a) + np.arctan(roll)
    ux, uy = np.cos(theta_l), np.sin(theta_l)
    vx, vy = np.cos(theta_r), np.sin(theta_r)
    b = np.array([x_pt[31,:] - x_pt[37,:], y_pt[31,:] - y_pt[37,:]])
    #A = np.array([[vx, -ux], [vy, -uy]]) / ()
    A = np.array([[-uy, -vy], [ux, vx]]) / (ux*vy-vx*uy)
    A = A.swapaxes(0,2)
    b = np.expand_dims(b, axis=2)
    b = b.swapaxes(0,1)
    sol = np.matmul(A, b)[:,1,0]
    sol[np.isnan(sol)] = 0
    x_tmp, y_tmp = x_pt[31,:] + sol * ux, y_pt[31,:] + sol * uy
    ux, uy = x_tmp - x_pt[37,:], y_tmp - y_pt[37,:]
    vx, vy = x_pt[31,:] - x_pt[37,:], y_pt[31,:] - y_pt[37,:]
    e = (vx**2+vy**2)**.5
    d = ((x_pt[35,:]-x_pt[33,:])**2+(y_pt[35,:]-y_pt[33,:])**2)**.5
    elong = 1-d/e
    x_pt[35,:], y_pt[35,:] = x_pt[37,:] + elong*ux, y_pt[37,:] + elong*uy
    elong = d/e
    x_pt[33,:], y_pt[33,:] = x_pt[35,:] + elong*vx, y_pt[35,:] + elong*vy
    x_pt[34,:] = (x_pt[33,:] + x_pt[35,:])/2
    y_pt[34,:] = (y_pt[33,:] + y_pt[35,:])/2 -.01
    
    # Mouth opening
    vx, vy = -vy/e, vx/e
    x_pt[44,:], y_pt[44,:] = x_pt[34,:]+vx*l_u, y_pt[34,:]+vy*l_u
    x_pt[47,:], y_pt[47,:] = x_pt[44,:]+vx*m_h, y_pt[44,:]+vy*m_h
    x_pt[40,:], y_pt[40,:] = x_pt[47,:]+vx*l_d, y_pt[47,:]+vy*l_d

    x_pt, y_pt = adjusting_others(x_pt, y_pt, roll)
    
    vector0 = np.array([x_pt[34,:]-x_pt[16,:], 
                        y_pt[34,:]-y_pt[16,:]])
    m_n_current = np.sum(vector0**2, axis=0)**.5
    x_pt[31:,:] += (np.sign(vector0[0,:])*m_n/m_n_current - 1) * vector0[0,:]
    y_pt[31:,:] += (-np.sign(vector0[1,:])*m_n/m_n_current - 1) * vector0[1,:]
    
    landmarks = np.zeros((98, nb_frame))
    landmarks[:49,:] = x_pt
    landmarks[49:,:] = -y_pt
    landmarks, _, _, _ = normalize(landmarks)
    landmarks *= depth
    landmarks[:49,:] += x
    landmarks[49:,:] += y
    
    if not np.sum(np.isnan(landmarks)) == 0:
        reconstruction.x_pt = x_pt
        reconstruction.y_pt = y_pt
        reconstruction.e_r_a = e_r_a
        print('Error a rattraper')
        
    return landmarks