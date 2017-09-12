#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------ Libraries & Global variables -------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import cvxpy
from cvxopt import matrix, spmatrix, solvers
solvers.options['show_progress'] = False
from helper import distance_l2, norm
from signal_processing import scaling, convolution, conv_filter
from signal_processing import empirical_bins
from global_path import SMOOTHING

"""
----------------------------------- Scores ------------------------------------
"""
def AP(soft_pred, y_true, precision_score=False, recall_score=False, thres=1):
    nbp = np.sum(y_true==1)
    ind = np.argsort(soft_pred)
    order = y_true[ind][::-1]
    if precision_score:
        best = 0
    elif recall_score:
        best = 0
    else:
        precs, recs = [], []     
    for i in range(len(order)):
        tmp = order[:i+1]
        tp = np.sum(tmp==1)
        rec = tp / nbp
        prec = tp / (i+1)
        if precision_score:
            if prec >= thres and rec >= best:
                best = rec
        elif recall_score:
            if rec >= thres and prec >= best:
                best = prec
        else:
            recs.append(rec)
            precs.append(prec)
    if precision_score or recall_score:
        return best
    return np.array(precs), np.array(recs)
    
def plot_AP(soft_pred, y_true, ax=None):
    x, y = AP(soft_pred, y_true)
    ctl = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ctl = True
    ax.plot(y, x, '*--')
    ax.set_title('Recall-Precision curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if ctl:
        return fig, ax

def ROC(soft_pred, y_true):
    nbp = np.sum(y_true==1)
    nbn = np.sum(y_true==-1)
    ind = np.argsort(soft_pred)
    order = y_true[ind][::-1]
    sens, anti = [], []
    for i in range(len(order)):
        tmp = order[:i+1]
        tp = np.sum(tmp==1)
        fp = np.sum(tmp==-1)
        sens.append(tp / nbp)
        anti.append(fp / nbn)
    return np.array(anti), np.array(sens)

def plot_ROC(soft_pred, y_true, ax=None):
    x, y = ROC(soft_pred, y_true)
    ctl = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ctl = True
    ax.plot(x, y, '*--')
    ax.set_title('ROC curve')
    ax.set_xlabel('1 - Speficity')
    ax.set_ylabel('Sensibility')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if ctl:
        return fig, ax

def integrale(x, y):
    ind = np.argsort(x)
    x, y = x[ind], y[ind]
    x_inc = x[1:] - x[:-1]
    y_mid = (y[1:] + y[:-1])/2.
    return np.sum(x_inc * y_mid)

def AUC(soft_pred, y_true):
    try:
        x, y = ROC(soft_pred, y_true)
    except ZeroDivisionError:
        return .5
    return integrale(x, y)

"""
------------------- Sparse classifier with convex relaxation ------------------
"""
def linear_Svm(design, labels, gamm = .1, balanced = True, affine = True, p=2):
    X, y = np.copy(design), np.copy(labels).astype(np.float)
    N = X.shape[1]
    w = cvxpy.Variable(N)
    if not affine:
        all_loss = cvxpy.pos(1 - cvxpy.mul_elemwise(y, X*w))
    else:
        b = cvxpy.Variable()
        all_loss = cvxpy.pos(1 - cvxpy.mul_elemwise(y, X*w - b))
    reg = cvxpy.norm(w, p)
    gamma = cvxpy.Parameter(sign="positive")

    if not balanced:
        M = X.shape[0]
        loss = cvxpy.sum_entries(all_loss)/M
    else:
        balanced = np.zeros(y.shape)
        balanced[y==1] = 1./np.sum(y==1)
        balanced[y==-1] = 1./np.sum(y==-1)
        loss = cvxpy.sum_entries(cvxpy.mul_elemwise(balanced, all_loss))
    prob = cvxpy.Problem(cvxpy.Minimize(loss+gamma*reg))
    gamma.value = gamm
    prob.solve()
    if affine:
        return np.array(w.value).flatten(), np.array(b.value).flatten()
    else:
        return np.array(w.value).flatten(), 0
    
def lasso(design, labels, gamm = .1, balanced = True, affine = True, p = 1):
    X, y = np.copy(design), np.copy(labels).astype(np.float)
    if balanced:
        y[y==1] *= np.mean(y==-1)
        y[y==-1] *= 1 - np.mean(y==-1)
    N = X.shape[1]
    # Construct the problem.
    gamma = cvxpy.Parameter(sign="positive")
    w = cvxpy.Variable(N)
    if not affine:
        error = cvxpy.sum_squares(X*w - y)
    else:
        b = cvxpy.Variable()
        error = cvxpy.sum_squares(X*w - b - y)
    obj = cvxpy.Minimize(error + gamma*cvxpy.norm(w, p))
    prob = cvxpy.Problem(obj)

    # Solve it
    gamma.value = gamm
    prob.solve()
    if affine:
        return np.array(w.value).flatten(), np.array(b.value).flatten()
    else:
        return np.array(w.value).flatten(), 0
    
def least_square(design, labels, gamm=0, balanced=True, affine=True, p=None):
    X, y = np.copy(design), np.copy(labels).astype(np.float)
    if balanced:
        y[y==1] *= np.mean(y==-1)
        y[y==-1] *= 1 - np.mean(y==-1)
    if affine:
        X = np.hstack((X, np.ones((X.shape[0],1))))
    if gamm==0:
        w = np.matmul(np.linalg.pinv(X), y)
    else:
        try:
            A = np.matmul(X.transpose(), X) + gamm * np.eye(X.shape[1])
            w = np.matmul(np.linalg.inv(A),np.matmul(X.transpose(),y))
        except np.linalg.linalg.LinAlgError:
            w = np.matmul(np.linalg.pinv(X), y)
    if affine:
        return w[:-1], w[-1]
    else:
        return w, 0

class SparseClassifier:
    def __init__(self, gamma_sel = .1, gamma_up = 0.01, sel_func=least_square,
                 update_func = linear_Svm):
        self.gamma = gamma_sel
        self.gamma2 = gamma_up
        self.update_func = update_func
        self.sel_func = sel_func

    def set_parameter(self, parameter):
        """
        gamma (selection), opt: gamma2 (update), udapte_func, selection_func
        """
        self.gamma = parameter[0]
        if len(parameter) > 1:
            self.gamma2 = parameter[1]
        if len(parameter) > 2:
            self.update_func = parameter[2]
        if len(parameter) > 3:
            self.sel_func = parameter[3]

    def fit(self, x_train, y_train, balanced = True, affine = True,
            enhancement = False, thres = 10**(-2)):
        p = 2
        if enhancement:
            p = 1
        self.w, self.b = self.sel_func(x_train, y_train, gamm = self.gamma, 
                                       balanced=balanced, affine=affine, p=p)
        self.I = np.zeros(x_train.shape[1])==0
        if enhancement:
            """ Enhance Sparsity """
            tmp = np.abs(self.w)
            self.I = tmp > thres*np.max(tmp)
            self.w, self.b = self.update_func(x_train[:, self.I], y_train, 
                                              gamm=self.gamma2, affine=affine,
                                              balanced=balanced, p=2)

    def predict(self, x_test):
        return np.matmul(x_test[:,self.I], self.w) - self.b

    def evaluate(self, x_test, y_test):
        soft_pred = self.predict(x_test)
        return AUC(soft_pred, y_test)
    
Gamma = np.logspace(-2,2,num=3)
SPARSE_CLASSIFIER_HYPER = [[g] for g in Gamma]

"""
------------------------- Orthogonal Matching Pursuit -------------------------
"""
class MatchingPursuit:
    def __init__(self, update_func = least_square, M = None, gamma = 0):
        self.update_func = update_func
        self.M = M
        self.gamma = gamma

    def set_parameter(self, parameter):
        """
        M (selection), gamma (update), opt: udapte_function
        """
        self.M = parameter[0]
        self.gamma = parameter[1]
        if len(parameter) > 2:
            self.update_func = parameter[2]
    
    def fit(self, x_train, y_train, balanced = True, verbose=False):
        X, y = np.copy(x_train), y_train.astype(np.float)
        if balanced:
            y[y==1] *= np.mean(y==-1)
            y[y==-1] *= 1 - np.mean(y==-1)

        N = X.shape[1]
        if self.M is None:
            self.M = int(N/10)
        M = self.M

        if verbose:
           W, gamma, E = [], [], []
           
        w = np.zeros(N)
        for k in range(M):
            tmp = y - np.matmul(X, w)
            c = np.matmul(X.transpose(), tmp) 
            i = np.argmax(np.abs(c))
            w[i] = w[i] + c[i]
            I = np.invert(w==0)
            design = X[:,I]
            w[I], _ = self.update_func(design, y, gamm = self.gamma, 
                                         affine=False, balanced=False)
            if verbose:
                E.append(norm(y-np.matmul(X, w)))
                gamma.append(np.abs(c[i]))
                W.append(np.copy(w))
                
            if k%5==0:
                self.w = w[I]
                self.I = I
                score = self.evaluate(x_train, y_train)
                if verbose:
                    print('Iteration %d, score %.2f' %(k, score))
                if score > .95:
                    break
            
        self.w = w[I]
        self.I = I
        if verbose:
            W = np.array(W)
            gamma = np.array(gamma)
            E = np.array(E)
            return W, gamma, E
        
    def predict(self, x_test):
        return np.matmul(x_test[:, self.I], self.w)
    
    def evaluate(self, x_test, y_test):
        soft_pred = self.predict(x_test)
        return AUC(soft_pred, y_test)

Gamma = np.logspace(-1,1,num=3)
MATCHING_PURSUIT_HYPER = [[None, g] for g in Gamma]

"""
------------------------ Recursive Features Elimination -----------------------
"""
class FeatureElimination:
    def __init__(self, met=SparseClassifier(gamma_sel=1, sel_func=linear_Svm), 
                 nb = 20, percentage_remove = .05):
        self.met = met
        self.nb = nb
        self.pr = percentage_remove
        
    def set_parameter(self, parameter):
        """
        nb (selection), opt: met (update)
        """
        self.nb = parameter[0]
        if len(parameter) > 1:
            self.met = parameter[1]

    def fit(self, design, labels):
        self.I = np.zeros(design.shape[1])==0
        tmp = np.arange(len(self.I))
        nb_features = np.sum(self.I)
        while nb_features > self.nb:
            nb_to_remove = min(nb_features-self.nb, int(self.pr*nb_features+1))
            X = design[:,self.I]
            self.met.fit(X, labels)            
            ind = np.argsort(np.abs(self.met.w))[:nb_to_remove]
            index = tmp[self.I][ind]
            self.I[index] = False 
            nb_features = np.sum(self.I)
        X = design[:, self.I]
        self.met.fit(X, labels)
    
    def predict(self, x_test):
        return self.met.predict(x_test[:, self.I])

    def evaluate(self, x_test, y_test):
        soft_pred = self.predict(x_test)
        return AUC(soft_pred, y_test)
    
Gamma = np.logspace(-2,2,num=3)
Met = [SparseClassifier(gamma_sel=g) for g in Gamma ]
Nb = [20,40]
FEATURE_ELIMINATION_HYPER = [[n, m] for m in Met for n in Nb]

"""
---------------------------- Support Vector Machine ---------------------------
"""
def poly_kernel(set_1, set_2, sigma = None):
    if sigma is None:
        d = 1
        alpha = 0
    elif type(sigma)==list:
        d = sigma[0]
        alpha = sigma[1]
    else:
        d = sigma
        alpha = 0
    return (alpha + np.matmul(set_1, set_2.transpose()))**d

def rbf_kernel(set_1, set_2, sigma = None):
    if sigma is None:
        sigma = np.var(set_2)
    kernel = np.exp( - distance_l2(set_1, set_2)/ sigma)
    return kernel

def chi_kernel(set1, set2, sigma = None):
    a = min(np.min(set1),np.min(set2))
    b =  max(np.max(set1),np.max(set2))
    if a < 0 or b > 1:
        set1 -= a
        set1 /= (b-a)
        set2 -= a
        set2 /= (b-a)        
    n1, n2 = len(set1), len(set2)
    kernel = np.zeros((n1, n2))
    for k in range(n1):
        x = set1[k]
        for l in range(n2):
            y = set2[l]
            kernel[k,l] = np.sum(np.divide(np.multiply(x, y), x+y))
    return kernel

class Svm:
    def __init__(self, gamma=.0001, kernel=rbf_kernel, sigma=.5, 
                 affine=False, weighted=True):
        self.gamma = gamma
        self.sigma = sigma
        self.kernel = kernel
        self.affine = affine
        self.weighted = weighted
        
    def parse(self, gamma, sigma):
        if gamma is None:
            gamma = self.gamma
        else:
            self.gamma = gamma
        if sigma is None:
            sigma = self.sigma
        else:
            self.sigma = sigma
        return gamma, sigma
            
    def set_parameter(self, parameter):
        """
        gamma (svm), sigma (kernel), opt: kernel, affine, weighted
        """
        self.gamma = parameter[0]
        self.sigma = parameter[1]
        if len(parameter) > 2:
            self.kernel = parameter[2]
        if len(parameter) > 3:
            self.affine = parameter[3]
        if len(parameter) > 4:
            self.weighted = parameter[4]  
            
    def fit(self, x_train, y_train, gamma=None, sigma=None):
        """
        Solves the SVM problem with 'cost-free' translation
        It uses cvxopt to solve it under its qp dual formulation
            maximize    -(1/2)*alpha'*K*alpha + 1'*alpha  
            subject to  0 <= diag(y)*alpha <= C = 1/(2*gamma*n)
                        1'*alpha = 0
    
        *y_train is supposed to be in {-1, 1}!*
        """
        gamma, sigma = self.parse(gamma, sigma)
        label = y_train.astype(np.double)
        self.x_train = x_train
        kernel = self.kernel(x_train, x_train, sigma)
    
        # Implicit parameters 
        N = len(x_train) 
        C = 1/float(2*gamma*N)

        if self.weighted:
            w_neg = np.mean(y_train==1)
            w_pos = 1 - w_neg
            w = np.zeros(y_train.shape)
            w[y_train==1] = C*w_pos
            w[y_train==-1] = C*w_neg
        else:
            w = C*np.ones(y_train.shape)

        # Cast everything into the standard QP formulation for cvx_opt
        P = matrix(kernel.astype(np.double))
        q = matrix(-label, tc = 'd')
        G = spmatrix([], [], [], size = (2*N, N))
        G[::2*N+1], G[N::2*N+1] = label, -label
        h = matrix(0.0, (2*N,1))
        h[:N] = w.astype(np.double)
        
        # Solving the equivalent qp, in case of a affine classification
        try:
            if self.affine:
                A = matrix(1.0,(1,N))
                b = matrix(0.0)        
                sol = solvers.qp(P,q,G,h,A,b)        
                self.b  = sol['y'][0]
            else:
                sol = solvers.qp(P,q,G,h)
            # Extract solution, alpha and the affine coefficient b
            alpha  = sol['x']
        except ValueError:
            print('Svm failure: check for kernel positivity?')
            self.b = 0
            alpha = np.random.randn(N)
        
        # Since alpha is supposed to be sparse: we forced it to be so
        self.support_vector =[k for k in range(N) if abs(alpha[k]) > 
                              max(1e-6 * max(abs(alpha)), 1e-6)]
        self.alpha = alpha[self.support_vector]
                    
    def predict(self, x_test):
        kernel_test = self.kernel(x_test, self.x_train, self.sigma)
        kernel_test = kernel_test[:, self.support_vector]
        soft_pred = np.matmul(kernel_test, self.alpha)
        if self.affine:
            soft_pred += self.b
        soft_pred = soft_pred.flatten()
        return soft_pred

    def evaluate(self, x_test, y_test):
        soft_pred = self.predict(x_test)
        return AUC(soft_pred, y_test)
        
Gam = np.logspace(-6,2,num = 5)
P_Sig = [None, 2, 4]
R_Sig = np.logspace(0,2,num=4)
tmp1 = [[g, s, rbf_kernel] for g in Gam for s in R_Sig]
tmp2 = [[g, s, poly_kernel] for g in Gam for s in P_Sig]
SVM_HYPER = tmp1 + tmp2

"""
------------------------------ Nearest Neighbors ------------------------------
"""
def kernel2distance(set1, set2, kernel_func, sigma = None):
    nb1, nb2 = len(set1),len(set2)
    res = np.zeros((nb1, nb2))
    scp = kernel_func(set1, set2, sigma)
    scp1 = kernel_func(set1, set1, sigma)
    scp2 = kernel_func(set2, set2, sigma)
    for i in range(nb1):
        for j in range(nb2):
            res[i,j] = scp1[i,i] + scp2[j,j] - 2*scp[i,j]
    return res

class NearestNeighbors:
    def __init__(self, distance_func = distance_l2, k = 1):
       self.distance_func = distance_func
       self.k = k
       
    def set_parameter(self, parameter):
        """
        k (nb neighbors), opt: distance_func
        """
        self.k = parameter[0]
        if len(parameter) > 1:
            self.distance_func = parameter[1]

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        dist = self.distance_func(self.x_train, x_test)
        ind_sort = np.argsort(dist, axis=0)[:self.k,:]
        soft_pred = np.mean(self.y_train[ind_sort], axis=0)
        return soft_pred
    
    def evaluate(self, x_test, y_test):
        soft_pred = self.predict(x_test)
        return AUC(soft_pred, y_test)

K = [1,2,4,8,20]    
NEAREST_NEIGHBORS_HYPER = [[k] for k in K]
    
"""
------------------------- Kernel Space Representation -------------------------
"""
class KernelRepresentation:
    def __init__(self, x_train, kernel = rbf_kernel, sigma = None):
        self.x_train = x_train
        self.kernel = kernel
        self.sigma = sigma
        self.K = self.kernel(self.x_train, self.x_train, sigma = self.sigma)
        self.gram_schmidt()
        self.design = self.reparameterize(self.x_train)
                
    def scp_x(self, ind_set1, ind_set2):
        """
        f_x = sum_i ind_set_i K_x_i
        """
        return np.matmul(np.matmul(ind_set1.transpose(), self.K), ind_set2)
                
    def gram_schmidt(self):
        N = self.x_train.shape[0]
        
        """ Initialization """
        family = []
        f_0 = (np.arange(N) == 0).astype(np.float)
        norm = self.scp_x(f_0, f_0)
        family.append(f_0 / (norm**.5))
        
        """ Recurrence """
        ctl = False
        length = 1
        for i in range(1,N):
            f_i = (np.arange(N) == i).astype(np.float)
            for j in range(length):
                f_j = family[j]
                scp = self.scp_x(f_i, f_j)
                f_i -= scp*f_j
            norm = self.scp_x(f_i, f_i)
            f_i = f_i / (norm**.5)
            
            """ Assert orthogonality """
            for j in range(length):
                f_j = family[j]
                ind = self.scp_x(f_i, f_j)
                if ind > .01 or np.isnan(ind):
                    break
            ind = self.scp_x(f_i, f_i)
            if ctl or ind < .99 or ind > 1.01 or np.isnan(ind) :
                continue
            length += 1
            family.append(f_i)
        self.family = np.array(family)
                
    def reparameterize(self, x):
        ind = self.kernel(self.x_train, x, self.sigma)
        rep = np.matmul(self.family, ind).transpose()
        return rep
        
    def scp(self, ind_set1, ind_set2):
        """
        K_x = sum_i ind_set_i f_i
        """
        return np.matmul(ind_set1, ind_set2.transpose())

"""
------------------------------- Metric Learning -------------------------------
"""
def scalar_product(set_1, set_2, S):
    return np.matmul(set_1, np.matmul(S, set_2.transpose()))

def scp2dist(scp, i, j):
    return scp[i,i]+scp[j,j]-2*scp[i,j]

def distance(set1, set2, S, scp_func = scalar_product):
    nb1, nb2 = len(set1),len(set2)
    res = np.zeros((nb1, nb2))
    scp = scp_func(set1, set2, S)
    scp1 = scp_func(set1, set1, S)
    scp2 = scp_func(set2, set2, S)
    for i in range(nb1):
        for j in range(nb2):
            res[i,j] = scp1[i,i] + scp2[j,j] - 2*scp[i,j]
    return res

class MetricLearning:
    def __init__(self, design, labels):
        self.design = design
        self.labels = labels
        self.clusters = self.compute_clusters()
        self.X = self.compute_aux_matrices()
        
    def compute_clusters(self):
        nb = len(self.labels)
        self.ind = np.array([[[[i,j,k] for i in range(nb)] 
                            for j in range(nb) ] for k in range(nb)])
        self.ind_pos = np.arange(nb)[self.labels==1]
        self.ind_neg = np.arange(nb)[self.labels==-1]
        clusters = [self.ind_pos, self.ind_neg]
        return clusters

    def compute_aux_matrices(self):
        nb = len(self.labels)
        X = [[ [] for i in range(nb)] for j in range(nb)]
        for i in range(nb):
            x_i = self.design[i]
            for j in range(nb):
                x_j = self.design[j]
                x = np.expand_dims(x_i - x_j, axis=1)
                X[i][j] = np.matmul(x, x.transpose())
        X = np.array(X)
        return X
    
    def compute_init_gradient(self, clusters):
        G = None
        for cluster in clusters:
            nb_cl = len(cluster)
            for i_ind in range(nb_cl):
                i = cluster[i_ind]
                for j_ind in range(i_ind+1, nb_cl):
                    j = cluster[j_ind]
                    if G is None:
                        G = self.X[i,j]
                    else:
                        G += self.X[i,j]
        return G
    
    def derive_A(self, S):
        scp = scalar_product(self.design, self.design, S)
        nb = len(self.labels)
        A, loss = np.zeros((nb, nb, nb)), 0
        for cluster in self.clusters:
            nb_cl = len(cluster)
            for i_ind in range(nb_cl):
                i = cluster[i_ind]
                for j_ind in range(i_ind+1, nb_cl):
                    j = cluster[j_ind]
                    if self.labels[i]==1:
                        for l in self.ind_neg:
                            tmp = 1 + scp2dist(scp,i,j)-scp2dist(scp,i,l)
                            if tmp > 0:
                                A[i,j,l] = 1
                                loss += tmp
                    if self.labels[i]==-1:
                        for l in self.ind_pos:
                            tmp = 1 + scp2dist(scp,i,j)-scp2dist(scp,i,l)
                            if tmp > 0:
                                A[i,j,l] = 1
                                loss += tmp
        loss *= self.gamma
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                loss += S[i,j]*self.G_init[i,j]
        return A, loss
    
    def update_gradient(self, A_diff):
        inds = self.ind[A_diff==1]
        G = None
        for ind in inds:
            l,j,i = ind[0], ind[1], ind[2]
            if G is None:
                G = self.X[i,j] - self.X[i,l]
            else:
                G += self.X[i,j] - self.X[i,l]
        inds = self.ind[A_diff==-1]
        for ind in inds:
            l,j,i = ind[0], ind[1], ind[2]
            if G is None:
                G = - (self.X[i,j] - self.X[i,l])
            else:
                G -= self.X[i,j] - self.X[i,l]
        return G

    def update_S(self, S, gradient,  alpha):
        tmp = S - alpha*gradient
        """ Projecting on the SDP cone """
        s, V = np.linalg.eigh(tmp)
        S = np.diag(s)
        S[S<0] = 0
        return np.dot(V, np.dot(S, V.transpose()))
    
    def fit(self, clusters=None, gamma=.1, alpha=10**(-3), max_it=10**3,
            verbose = False):
        nb, d = len(self.labels), self.design.shape[1]
        if clusters is None:
            clusters = self.clusters
        else:
            self.clusters = clusters
        self.gamma = gamma
        self.G_init = self.compute_init_gradient(clusters)
        S, t, A_new, loss_new = np.eye(d), 0, np.zeros((nb, nb, nb)), np.inf
        G = np.copy(self.G_init)
        
        if verbose:
            loss = []
            S_tot = []
        loss_min = np.inf
        tol = 10**(-4)*alpha
        
        while t < max_it and alpha > tol:
            t += 1
            """ Calcul de A """
            A_old, loss_old = np.copy(A_new), np.copy(loss_new)
            A_new, loss_new = self.derive_A(S)
            if loss_new < loss_min:
                loss_min = loss_new
                S_best = S
            if verbose:
                if t% (int(max_it/20)+1)==0:                
                    print('\nIteration {0:4}, Loss: {1:6}'.format(t,loss_new))
                    print(np.sum(A_new), 'triplets don\'t respect the margin')
                loss.append(loss_new)
                S_tot.append(S)
            A = A_new - A_old
            try:
                G += gamma * self.update_gradient(A)
            except TypeError:
                pass
            if loss_new < loss_old:
                alpha *= 1.01
            else:
                alpha *= .5
            S = self.update_S(S, G, alpha)
        if verbose:
            print('\nMinimal Loss: {0:6}'.format(loss_min))
            return S_tot, loss
        return S_best
    
"""
------------------------- Objective Related Classifier ------------------------
"""
class MetricLearningClassifier:
    def __init__(self, gamma=.1, alpha=10**(-3), max_it=10):
        self.g = gamma
        self.a = alpha
        self.i = max_it

    def set_parameter(self, parameter):
        """
        gamma (gradient descent), alpha (momentum), max_it (nb iterations)
        """
        self.g = parameter[0]
        self.a = parameter[1]
        self.i = parameter[2]

    def fit(self, x_train, y_train, clusters=None, verbose = True):
        self.x_train = x_train
        self.y_train = y_train
        self.algo = MetricLearning(x_train, y_train)
        self.S = self.algo.fit(clusters=clusters, gamma = self.g, 
                               alpha=self.a, max_it=self.i, verbose=verbose)
        if verbose:
            self.S_tot = self.S[0]
            self.S = self.S[0][-1]
      
    def predict_point(self, pt, label):
        self.algo.design = np.vstack((self.x_train, pt))
        self.algo.labels = np.hstack((self.y_train, label))
        self.algo.clusters = self.algo.compute_clusters()
        _, loss = self.algo.derive_A(self.S)
        return loss

    def predict(self, x_test):
        nb = len(x_test)
        soft_pred = -np.ones(nb)
        for i in range(nb):
            x = x_test[i]
            r1 = self.predict_point(x, 1)
            r2 = self.predict_point(x, -1)
            soft_pred[i] = (r2) / (r1+r2)
        return soft_pred
    
    def evaluate(self, x_test, y_test):
        soft_pred = self.predict(x_test)
        return AUC(soft_pred, y_test)
   
"""
-------------------------------- Decision Tree --------------------------------
"""
class Node:
    def __init__(self, test):
        self.test = test # test give a value in [0,1..] correponding to a split
        self.sons = []
                
    def add_sons(self, son):
        self.sons.append(son)
        
    def classify(self, point):
        return self.test(point)
    
    def length_lineage(self):
        best = 0
        for son in self.sons:
            try:
                depth = son.length_lineage() + 1
            except AttributeError:
                depth = 1
            if depth > best:
                best = depth
        return best
    
    def set_depth(self, depth):
        self.depth = depth
        
    def set_depth_lineage(self):
        depth = self.depth
        for son in self.sons:
            son.set_depth(depth+1)
            try:
                son.set_depth_lineage()
            except AttributeError:
                continue

    def explain_linage(self, explanation):
        ind = self.test(None, verbose = True)
        try:
            self.explanation = explanation[ind]
        except (IndexError, TypeError):
            self.explanation = 'Linear Classifier'
        for son in self.sons:
            try:
                son.explain_linage(explanation)
            except AttributeError:
                continue

class Leaf:
    def __init__(self, diagnosis):
        self.diagnosis = diagnosis
        
    def set_depth(self, depth):
        self.depth = depth
        
class Tree:
    def __init__(self, generator):
        self.generator = generator
        
    def find_path(self, path_list):
        current = self.generator
        for i in path_list:
            current = current.sons[i]
        return current

    def set_depth(self):
        self.generator.set_depth(0)
        self.generator.set_depth_lineage()
        self.depth = self.generator.length_lineage() + 1
        
    def classify(self, point):
        current = self.generator
        while True:
            try:
                i = current.test(point)
                current = current.sons[i]
            except AttributeError:
                break
        confidence = current.depth / self.depth
        res = current.diagnosis * confidence
        return res
    
    def explain(self, explanation):
        self.generator.explain_linage(explanation)
        
def show_lineage(ax, node, x_cur, y_cur, size, explain = False):
    if type(node) == Leaf:
        if node.diagnosis > .5:
            ax.scatter(x_cur, y_cur, color = 'b', marker = 's')
        else:
            ax.scatter(x_cur, y_cur, color = 'r', marker = 's')
    else:
        ax.scatter(x_cur, y_cur, color = 'k')
        if explain:
            ax.annotate(node.explanation, (x_cur, y_cur))
        n = len(node.sons)
        son_size = size / n
        for i in range(n):
            son = node.sons[i]
            x, y = (i - (n-1)/2) * size + x_cur, y_cur - 1
            ax.plot([x,x_cur], [y,y_cur], color = 'k', linewidth = .5)
            show_lineage(ax, son, x, y, son_size, explain = explain)
                        
def show_tree(tree, explain = False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    root = tree.generator
    x_cur, y_cur, size = 0, 0, 100
    show_lineage(ax, root, x_cur, y_cur, size, explain = explain)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Decision Tree')
    return ax

def split_func(design, labels, thres = .2):
    ind = np.argsort(design, axis=0)
    order = labels[ind] == 1
    extremity = int(design.shape[0]/10)
    pos_begin = np.mean(order[:extremity], axis = 0)
    pos_end = np.mean(order[-extremity:], axis = 0)
    tmp = pos_begin > pos_end
    score = pos_end
    score[tmp] = pos_begin[tmp]
    ind_test = np.argsort(score)[-1]
    
    test_features = design[:,ind_test]
            
    ind_sort_test = np.argsort(test_features)
    sort_test = test_features[ind_sort_test]
    
    gap = sort_test[1:] - sort_test[:-1]
    thres = np.max(gap) * thres
    big_gap = gap >= thres
    tmp = np.sum(big_gap)
    potential_cut = np.argsort(big_gap)[-tmp:] + 1
    
    dist_to_middle = np.abs(potential_cut - len(sort_test)/2)
    cut = potential_cut[np.argmin(dist_to_middle)]
    cut_thres = sort_test[cut]
    ind1 = test_features >= cut_thres
    if np.mean(ind1)==1:
        label = 2*(np.sum(labels) > 0)-1
        ind1 = labels == label
    ind2 = np.invert(ind1)
    
    design1, labels1 = design[ind1], labels[ind1]
    design2, labels2 = design[ind2], labels[ind2]
    splits = [[design1, labels1], [design2, labels2]]
    
    def test(point, verbose = False):
        if verbose:
            return ind_test
        return (point[ind_test] < cut_thres).astype(np.int) 
    
    return test, splits

def split_met(design, labels, met = FeatureElimination(), thres = None):
    met.fit(design, labels)
    tmp = met.predict(design) 
    maxi = np.max(tmp)
    mini = np.min(tmp)
    if maxi > 0 and mini < 0:
        thres = 0
    else:
        thres = (maxi + mini)/2
    ind1 = tmp > thres
    if np.sum(ind1) == 0:
        label = 2*(np.sum(labels) > 0)-1
        ind1 = labels == label
    ind2 = np.invert(ind1)

    design1, labels1 = design[ind1], labels[ind1]
    design2, labels2 = design[ind2], labels[ind2]
    splits = [[design1, labels1], [design2, labels2]]
    
    def test(point, verbose = False):
        if verbose:
            return met
        tmp = met.predict(np.expand_dims(point, axis=0)) < thres
        return (tmp[0]).astype(np.int) 
    
    return test, splits

def split(design, labels, thres=.2, nb_points=8, split_method = split_func,
          lin_method=MatchingPursuit()):
    if np.mean(labels==labels[0])==1:
        diag = labels[0]
        splits = [[design, labels]]
        return diag, splits
    
    elif len(labels) < nb_points:
        
        lin_method.fit(design, labels)
        
        ind1 = labels==1
        ind2 = np.invert(ind1)
        design1, labels1 = design[ind1], labels[ind1]
        design2, labels2 = design[ind2], labels[ind2]
        splits = [[design1, labels1], [design2, labels2]]
        
        def test(point, verbose = False):
            if verbose:
                return np.nan
            soft = lin_method.predict(np.expand_dims(point, axis=0))[0] 
            return (soft < 0).astype(np.int)
        
        return test, splits

    else:
        test, splits = split_method(design, labels, thres = thres)
        return test, splits
    
def build_tree(design, labels, ctl = True, split_method = split_func, 
               thres = .2, nb_points = 8, lin_method = MatchingPursuit()):
    test_or_diag, design_splits = split(design, labels, thres = thres, 
                                        nb_points=nb_points, 
                                        lin_method = lin_method,
                                        split_method = split_method)
    if len(design_splits)==1:
        # test_or_diag = diag
        generator = Leaf(test_or_diag)
    else:  
        # test_or_diag = test
        generator = Node(test_or_diag)
        for design_split in design_splits:
            d, l = design_split[0], design_split[1]
            son_test = build_tree(d, l, ctl = False, split_method=split_method,
                                  thres = thres, nb_points = nb_points,
                                  lin_method = lin_method)
            generator.add_sons(son_test)
    if ctl:
        tree = Tree(generator)
        return tree
    else:
        return generator
    
class DecisionTree:
    def __init__(self, split_method = split, thres = .2, nb_points = 8, 
                 lin_method = Svm(kernel=poly_kernel, sigma = None)):  
        self.split_method = split_method
        self.thres = thres
        self.nb_points = nb_points
        self.lin_method = lin_method

    def set_parameter(self, parameter):
        """
        split_function, opt: thres (sep.), nb_points and lin_method (classif.)
        """
        self.split_method = parameter[0]
        if len(parameter) > 1:
            self.thres = parameter[1]
        if len(parameter) > 2:
            self.nb_points = parameter[2]
        if len(parameter) > 3:
            self.lin_method = parameter[3]
        
        
    def fit(self, design, labels):
        self.tree = build_tree(design, labels, split_method=self.split_method,
                               thres = self.thres, nb_points = self.nb_points,
                               lin_method = self.lin_method)
        self.tree.set_depth()
        
    def predict(self, design):
        res = []
        for point in design:
            res.append(self.tree.classify(point))
        return np.array(res)
        
    def evaluate(self, design, labels):
        soft_pred = self.predict(design)
        return AUC(soft_pred, labels)
    

Spl = [split_met, split_func]
Thr = [.1,.2,.5]
DECISION_TREE_HYPER = [[s, t] for s in Spl for t in Thr]

"""
-------------------------- Distributions Estimation ---------------------------
"""
def bound_distribution(values, p_value = .05):
    mini_maxi = empirical_bins(values, tmp = np.array([p_value, 1-p_value]))
    return mini_maxi

def compute_range(values, nb_values = 100, mini_maxi = None, p_value=.05):
    if type(mini_maxi)==type(None):
        mini_maxi = bound_distribution(values, p_value = p_value)
    mini, maxi = mini_maxi[0], mini_maxi[1]
    x = np.arange(mini, maxi, (maxi-mini) / nb_values)[:nb_values]
    return x

def gaussian_window_density_estimation(x, values, d = 1, n = None):
    if type(n)==type(None):
        n = values.shape[0]
    if len(x.shape)==0:
        x = np.expand_dims(x, axis=1)
    window_size = np.var(values) * (n**(-1/(d+4)))
    vector = (values - np.expand_dims(x, axis=1)) / window_size
    y = np.exp(- vector**2 / 2 ) 
    y /= window_size*np.sqrt(2*np.pi)
    return y

def fast_estimation(x, values, smoothing = True):
    res = np.mean(gaussian_window_density_estimation(x, values), axis=-1)
    if smoothing:
        filt = conv_filter(scaling(x.shape[0]/SMOOTHING), name='Gaussian')
        res = convolution(res, filt)
        dx = x[1:] - x[:-1]
        normalization = np.sum(dx*res[1:])
        res = res/normalization
    return res

def likelihood(value, x, res):
    i = max(np.sum(x < value) - 1, 0)
    return res[i]
    
def estimate_density(values, nb_values, 
                     mini_maxi = None, p_value=.05, dim=None):
    """
    Estimation with Gaussian windows
    
    min_maxi = [[minimum_axe1, maximum_axe2]]
    """
    if len(values.shape) == 1:
        values = np.expand_dims(values, axis=0)
        
    # Let's get in the proper axis of the covariance
    U, s, V = np.linalg.svd(values, full_matrices=False)
    if dim is None:
        dim = values.shape[0]
    values = np.matmul(U.transpose()[:dim, :], values)
    d, n = values.shape        
        
    if type(mini_maxi) == type(None):
        mini_maxi = [None for i in range(d)]
           
    X = []
    len_res = [nb_values for i in range(d)]
    len_res.append(n)
    res = np.ones(len_res)
    print(d)
    for i in range(d):
        current_axis = values[i,:]
        print(current_axis)
        x = compute_range(current_axis, nb_values, mini_maxi[i], p_value)        
        y = gaussian_window_density_estimation(x, current_axis, d=d, n=n)
        X.append(x)
        if d==1:
            res = y
            break
        for j in range(len(x)):
            for s in range(y.shape[1]):
                ind = [slice(nb_values) for i in range(d)]
                ind[i] = j
                ind.append(s)
                res[tuple(ind)] *= y[j,s]
    res = np.mean(res, axis=-1)
    return X, res, U

"""
----------------------------- Generative Modeling -----------------------------
"""
class GenerativeModel:
    def __init__(self, nb_points = 100, thres = 1):
        self.nb_points = nb_points
        self.thres = thres
    
    def set_parameter(self, parameter):
        """
        threshold (eval), opt: nb_points (estimation)
        """
        self.thres = parameter[0]
        if len(parameter) > 1:
            self.nb_points = parameter[1]
    
    def fit(self, design, labels):
        self.E = []
        for i in range(design.shape[1]):
            tmp = design[:, i]
            tmp1 = tmp[labels==1]
            tmp0 = tmp[labels==-1]
            
            x_max, x_min = np.max(tmp), np.min(tmp)
            delta = x_max - x_min
            x_min, x_max = x_min - delta/2., x_max + delta/2.
            x = np.linspace(x_min, x_max, num=self.nb_points)
            
            res1 = fast_estimation(x, tmp1)
            res0 = fast_estimation(x, tmp0)     
                    
            dx = x[1:] - x[:-1]
            dres = np.abs(res1 - res0)[1:]
            inte = np.sum(dx*dres)
            if inte > self.thres:
                self.E.append([i, x, res1, res0])
        
    def predict(self, x_test):
        pred = []
        for x in x_test:
            res = 0
            for e in self.E:
                i, vals, res1, res0 = e[0], e[1], e[2], e[3]
                p1 = likelihood(x[i], vals, res1)
                p0 = likelihood(x[i], vals, res0)
                res += np.log(p1) - np.log(p0)
            pred.append(res)
        return np.array(pred)
    
    def evaluate(self, x_test, y_test):
        soft_pred = self.predict(x_test)
        return AUC(soft_pred, y_test)
   
Thr = [.1,1]
GENERATIVE_MODEL_HYPER = [[t] for t in Thr]

"""
---------------------------- Singularity Detection ----------------------------
"""    
class DetectionModel:
    def __init__(self, phi_func = None, psi_func = None):
        if phi_func is None:
            self.phi = lambda x: x**(-1)
        else:
            self.phi = phi_func
        if psi_func is None:
             self.psi = lambda x: (x < .1).astype(np.int)
#            self.psi = lambda x: (x < .05).astype(np.int) * (x ** (-1))
        else:
            self.psi = psi_func
    
    def set_parameter(self, parameter):
        """
        psi_func, opt: phi_func 
        """
        self.psi = parameter[0]
        if len(parameter) > 1:
            self.phi = parameter[1]
    
    def fit(self, prob_design, labels):
        design = self.phi(prob_design)
        tmp1 = np.mean(design[labels==1], axis=0)
        tmp0 = np.mean(design[labels==-1], axis=0)
        self.weight = np.log(tmp1) - np.log(tmp0)
        
    def predict(self, x_test):
        X = self.psi(x_test)
        tmp = X * np.expand_dims(self.weight, axis=0)
        return np.mean(tmp,axis=1)

    def evaluate(self, x_test, y_test):
        soft_pred = self.predict(x_test)
        return AUC(soft_pred, y_test)

psi1 = lambda x: (x < .1).astype(np.int)
psi2 = lambda x: (x < .05).astype(np.int) * (x ** (-1))
Psi = [psi1, psi2]
DETECTION_MODEL_HYPER = [[p] for p in Psi]
    
"""
----------------------------- k-means Clustering ------------------------------
""" 
def k_mean_assign(design, center, distance = distance_l2):
    dist_to_center = distance(design, center)
    I = np.argmin(dist_to_center, axis=1)
    return I
    
def k_means(design, k = 10, distance = distance_l2, 
            nb_it=20, tol=10**(-5), nb_try = 20):
    best, best_center = np.inf, None
    for trial in range(nb_try):
        # Choose k center at random
        center_ind = np.random.permutation(design.shape[0])[:k]
        center = design[center_ind]
        
        # Update center iteratively
        err, it = np.inf, 0
        while err > tol and it < nb_it:
            center_old = np.copy(center)
            I = k_mean_assign(design, center, distance = distance)
            i = 0
            while i < k:
                ind = i == I
                tmp = design[ind]
                if tmp.shape[0] == 0:
                    k -= 1
                    if i == 0:
                        center = center[1:]
                        center_old = center_old[1:]
                    elif i==k:
                        center = center[:-1] 
                        center_old = center_old[:-1] 
                    else:
                        center = np.vstack((center[:i], center[i+1:]))
                        center_old=np.vstack((center_old[:i],center_old[i+1:]))
                else:
                    center[i] = np.mean(tmp, axis=0)
                    i += 1
            
            err =  np.mean((center_old - center)**2)
        
        # Look for total distorsion
        dist_to_center = distance(design, center)
        distortion = np.sum(np.min(dist_to_center, axis=1))
        if np.isnan(distortion):
            distortion = np.inf
            
        if distortion <= best:
            best = distortion
            best_center = center
    return best_center

"""
---------------------------- Folds Cut for Scoring ----------------------------
"""    
def get_fold_cut(length, nb_cut):
    incr = length/nb_cut
    tmp = np.arange(0, length + incr/2, incr)
    tmp = np.array([int(i) for i in tmp])
    return tmp

class Folds_Cut:
    def __init__(self, nb_folds = 8):
        self.nb_folds = nb_folds
        
    def design_fold(self, labels, nb_folds = None):
        """
        labels in {-1, 1}
        """
        if type(nb_folds) == type(None):
            nb_folds = self.nb_folds
        self.nb_data = len(labels)
        ind_data = np.arange(self.nb_data)
        
        class_neg = np.random.permutation(ind_data[labels==-1])
        class_pos = np.random.permutation(ind_data[labels==1])
        
        cut_neg = get_fold_cut(len(class_neg), nb_folds)
        cut_pos = get_fold_cut(len(class_pos), nb_folds)
                
        all_folds = []
        for k in range(nb_folds):
            cur_neg = class_neg[cut_neg[k]:cut_neg[k+1]]
            cur_pos = class_pos[cut_pos[k]:cut_pos[k+1]]
            all_folds.append(np.sort(np.concatenate((cur_neg, cur_pos))))
        self.all_folds = np.array(all_folds)
        self.current_ind = -1
    
    def get_fold(self):
        fold = self.all_folds[self.current_ind]
        ind_val = np.array([ind in fold for ind in range(self.nb_data)])
        ind_train = np.invert(ind_val)
        return ind_train, ind_val
        
    def next_fold(self):
        self.current_ind += 1
        ind_train, ind_val = self.get_fold()
        return ind_train, ind_val
    
    def reinit(self):
        self.current_ind = -1
    
"""
----------------------- Cross Evaluation and Validation -----------------------
"""
def cross_evaluation(method, design, labels, evaluation = None,
                     folds = None, nb_folds = 8, verbose = False):
    if folds is None:
        folds = Folds_Cut(nb_folds)
        folds.design_fold(labels)
    else:
        nb_folds = folds.nb_folds
    res = np.zeros(nb_folds)
    for k in range(nb_folds):
        ind_train, ind_val = folds.next_fold()
        train_data = design[ind_train]
        train_labels = labels[ind_train]
        val_data = design[ind_val]
        val_labels = labels[ind_val]
        method.fit(train_data, train_labels)
        if evaluation is None:
            res[k] = method.evaluate(val_data, val_labels)
            y_pred = method.predict(val_data)
        else:
            y_pred = method.predict(val_data)
            res[k] = evaluation(y_pred, val_labels)
    if verbose:
        return res
    else:
        return np.mean(res)

def cross_validation(method, parameter_list, design, labels, evaluation = None,
                     folds = None, nb_folds = 8, verbose = False):
    if folds is None:
        folds = Folds_Cut(nb_folds)
        folds.design_fold(labels)
    else:
        nb_folds = folds.nb_folds
    nb_parameter = len(parameter_list)
    res = np.zeros((nb_folds, nb_parameter))
    for k in range(nb_folds):
        ind_train, ind_val = folds.next_fold()
        train_data = design[ind_train]
        train_labels = labels[ind_train]
        val_data = design[ind_val]
        val_labels = labels[ind_val]
        for l in range(nb_parameter):
            parameter = parameter_list[l]
            method.set_parameter(parameter)
            method.fit(train_data, train_labels)
            if evaluation is None:
                res[k,l] = method.evaluate(val_data, val_labels)
            else:
                y_pred = method.predict(val_data)
                res[k,l] = evaluation(y_pred, val_labels)    
    if verbose:
        return np.array(res)
    else:
        print('best score:', np.max(np.mean(res, axis=0)))
        best_ind = np.argmax(np.mean(res, axis=0))
        best_parameter = parameter_list[best_ind]
        method.set_parameter(best_parameter)
        return best_parameter

"""
------------------------ LOFO Evaluation and Validation -----------------------
"""
class Leave_One_Subject_Out:
    def __init__(self, nb_subject):
        self.all_ind = np.arange(nb_subject)
        self.current_ind = -1
        
    def get_subject(self):
        ind_val = self.all_ind == self.current_ind
        ind_train = np.invert(ind_val)
        return ind_train, ind_val
   
    def next_subject(self):
        self.current_ind += 1
        ind_train, ind_val = self.get_subject()
        return ind_train, ind_val
    
    def reinit(self):
        self.current_ind = -1
    
def scoring_LOSO(method, design, labels, evaluation = AUC, verbose = False):        
    nb_subject = len(labels)
    folds = Leave_One_Subject_Out(len(labels))
    predict = np.zeros(nb_subject)
    true = np.zeros(nb_subject)
    for k in range(nb_subject):
        ind_train, ind_val = folds.next_subject()
        train_data = design[ind_train]
        train_labels = labels[ind_train]
        val_data = design[ind_val]
        val_labels = labels[ind_val]
        method.fit(train_data, train_labels)
        true[ind_val] = val_labels
        tmp = method.predict(val_data)
        predict[ind_val] = tmp
    if verbose:
        return predict, true
    else:
        res = evaluation(predict, true)
        return res
    
def cross_validation_LOFO(method, parameter_list, design, labels,
                          evaluation = AUC, verbose = False):
    res = []
    for parameter in parameter_list:
        method.set_parameter(parameter)
        res.append(scoring_LOSO(method, design, labels, evaluation=evaluation))
    if verbose:
        return np.array(res)
    else:
        best_ind = np.argmax(np.array(res))
        best_parameter = parameter_list[best_ind]
        method.set_parameter(best_parameter)
        return best_parameter
    
"""
---------------------------- Classifier Structure -----------------------------
"""
class classifier:
    def __init__(self, method, ind_sel_func = None, parameter_list = None, 
                 evaluation = None, nb_folds = 6):
        self.method = method
        self.ind_sel_func = ind_sel_func
        self.parameter_list = parameter_list
        self.eval = evaluation
        self.nb_folds = nb_folds
    
    def set_parameter(self, parameter):
        """
        param_list, opt: selection_func, nb folds, evaluation_function
        """
        self.parameter_list = parameter[0]
        if len(parameter) > 1:
            self.ind_sel_func = parameter[1]
        if len(parameter) > 2:
            self.nb_folds = parameter[2]
        if len(parameter) > 3:
            self.eval = parameter[3]
        
    def fit(self, design, labels):
        if not self.ind_sel_func is None:
            self.ind = self.ind_sel_func(design, labels)
        else:
            self.ind = np.ones(design.shape[1]) == 0
        X = design[:,self.ind]
        if not self.parameter_list is None:
            best=cross_validation(self.method, self.parameter_list, X, labels, 
                             evaluation = self.eval, nb_folds = self.nb_folds)
            print('best_parameter:', best)
        self.method.fit(X, labels)
        
    def predict(self, design):
        X = design[:,self.ind]
        soft_pred = self.method.predict(X)
        return soft_pred

    def evaluate(self, design, labels):
        soft_pred = self.predict(design)
        return AUC(soft_pred, labels)

"""
-------------------------- Discriminating Two Childs --------------------------
"""

class Trivial:
    def __init__(self):
        pass
    
    def fit(self, design, labels):
        pass
    
    def predict(self, design):
        return design[:,0]
    
    def evaluate(self, design, labels):
        pred = self.predict(design)
        return AUC(pred, labels)
    
def dual_score(method, design, labels, nb_try = 100, verbose = False):
    ind = np.arange(len(labels))
    ind_aut = ind[labels == 1]
    ind_nor = ind[labels == -1]
    couples = np.random.permutation([[i, j] for i in ind_aut for j in ind_nor])
    if nb_try is None:
        nb_try = len(couples)
    res = []
    for i in range(nb_try):
        couple = couples[i]
        aut_ind, nor_ind = couple[0], couple[1]
        ind_test = (ind == aut_ind) + (ind == nor_ind)
        ind_train = np.invert(ind_test)
        train_data = design[ind_train]
        train_labels = labels[ind_train]
        test_data = design[ind_test]
        test_labels = labels[ind_test]
        method.fit(train_data, train_labels)
        pred = method.predict(test_data)
        comp = pred[test_labels==1] - pred[test_labels==-1]
        if comp > 0:
            res.append(1)
        elif comp == 0:
            res.append(np.nan)
        else:
            res.append(0)
    res = np.array(res)
    if verbose:
        return res
    return np.mean(res[np.invert(np.isnan(res))])


"""
------------------------------ Find Best Methods ------------------------------
"""
def find_AUC_method(design, proba_design, labels, perf = .8, nb_folds = 5):
    to_try = [[SparseClassifier(), SPARSE_CLASSIFIER_HYPER],
              [MatchingPursuit(), MATCHING_PURSUIT_HYPER],
              [FeatureElimination(), FEATURE_ELIMINATION_HYPER],
              [Svm(), SVM_HYPER],
              [NearestNeighbors(), NEAREST_NEIGHBORS_HYPER],
              [GenerativeModel(), GENERATIVE_MODEL_HYPER]]
    
    good = []
    for full_met in to_try:
        met = full_met[0]
        hyper = full_met[1]
        for para in hyper:
            try:
                met.set_parameter(para)
                res = cross_evaluation(met, design, labels, nb_folds=nb_folds)
                if res > perf:
                    good.append([res, met, para])
            except:
                continue
    met = DetectionModel()
    hyper = DETECTION_MODEL_HYPER
    for para in hyper:
        try:
            met.set_parameter(para)
            res = cross_evaluation(met, proba_design,labels, nb_folds=nb_folds)
            if res > perf:
                good.append([res, met, [None]])
        except:
            continue
    return good

def find_dual_method(design, proba_design, labels, perf = .8, nb_try = 50):
    to_try = [[SparseClassifier(), SPARSE_CLASSIFIER_HYPER],
              [MatchingPursuit(), MATCHING_PURSUIT_HYPER],
              [FeatureElimination(), FEATURE_ELIMINATION_HYPER],
              [Svm(), SVM_HYPER],
              [NearestNeighbors(), NEAREST_NEIGHBORS_HYPER],
              [GenerativeModel(), GENERATIVE_MODEL_HYPER]]
    
    good = []
    for full_met in to_try:
        met = full_met[0]
        hyper = full_met[1]
        for para in hyper:
            try:
                met.set_parameter(para)
                res = dual_score(met, design, labels, nb_try = nb_try)
                if res > perf:
                    good.append([res, met, para])
                    print(good[-1])
            except:
                continue
    met = DetectionModel()
    hyper = DETECTION_MODEL_HYPER
    for para in hyper:
        try:
            met.set_parameter(para)
            res = dual_score(met, design, labels, nb_try = nb_try)
            if res > perf:
                good.append([res, met, [None]])
        except:
            continue
    return good

