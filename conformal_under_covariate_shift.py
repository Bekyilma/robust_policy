# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:12:47 2020

@author: muhos601
"""
#distribution parameters
(mup, sp, se) = (0.5, 0.5, 1)
(muq, sq) = (0, 0.2)
#threshold for termination half-interval search
thr = 0.005
#theta_str
theta_str = 2
#x-dependent noise

#Functions for conformal prediction under covariate shift, Barber et al.
import scipy.stats
import numpy as np

#x-dependent noise
def sig_x(x):
    err_var = np.zeros(np.size(x))
    for i in range(np.size(x)):
        if  x[i]>=1 and x[i]<=1.5:
            err_var[i] = 1.15
        else:
            err_var[i] = 0.15

    #large noise for large positive x
    #err_var = np.maximum(np.zeros(np.size(x)), 0.5*x)
    #same noise across all x
    #err_var = 0.15*np.ones(np.size(x))
    return err_var

def point2aug(y_low, y_upp):
    #point to augment
    return (y_low + y_upp)/2

def aug_point(X,yv,y_tilda,phix):
    #augment point y_tilda and x_tilda to X, yv
    Xaug = np.append(X, phix, axis = 1)
    yaug = np.insert(yv,np.size(yv),y_tilda)
    
    return Xaug, yaug

def phi(x_tilda):
    #basis vector d x 1
    phix = np.array([[1], [x_tilda]])
    
    return phix

def phi_mat(x):
    #generate basis matrix d x n
    n = np.size(x)
    Phi = np.zeros((n,2))
    for i in range(n):
        Phi[i] = phi(x[i]).T
    
    Phi = Phi.T
    return Phi

def ls_estimator_trn(X,yv,wt_vec):
    #find weighted least square estimate
    #shape of X matrix d x n
    shp = np.shape(X)
    #Modify X using the weights for weighted LS estimate
    Xm = X*np.sqrt(wt_vec.T)
    #Modify yv n x 1 vector
    yvm = yv*np.sqrt(wt_vec)
    #Moore Penros inverse of X*X.T
    Precs_n  = np.linalg.pinv(Xm.dot(Xm.T))
    #cross correlation vector
    rho_n = Xm.dot(yvm)
    rho_n = rho_n.reshape((shp[0],1))
    #estimate of theta
    theta = Precs_n.dot(rho_n)
    
    return Precs_n, rho_n, theta

def update_ls(Precs_n, rho_n, phix, y_tilda, wt):
    
    #modify phix for weighted ls
    phix_m = phix*np.sqrt(wt)
    #modify y_tilda
    y_tilda_m = y_tilda*np.sqrt(wt)
    
    num = Precs_n.dot(phix_m.dot(phix_m.T)).dot(Precs_n)
    
    den = (1+(phix_m.T.dot(Precs_n)).dot(phix_m))
    
    #updata inverse
    Precs_n1 = Precs_n - num/den
    #update cross correltion vector
    rho_n1 = rho_n + phix_m*y_tilda_m
    #obtain new estimate
    theta = Precs_n1.dot(rho_n1)
    
    return theta
    
def non_cnf_scr(Xaug,yaug,theta):
    #find non-conformity score
    scr = np.abs(theta.T.dot(Xaug)-yaug)
    scr = scr[0]
    return scr

def test_dist(x):
    #evaluates test distribution at array x
    q = scipy.stats.norm(muq,sq).pdf(x)
    
    return q

def train_dist(x):
    #evaluates test distribution at array x
    p = scipy.stats.norm(mup,sp).pdf(x)
    
    return p

def weight(x):
    #liklihood ratio w(x) = q(x)/p(x)
    w = test_dist(x)/train_dist(x)
    return w
    
def qaunt(wt, scr, alpha):
    #quantile such that Pr(Z>=z)>=1-alpha where Z is the R.V for non-conformity score
    prb = wt/np.sum(wt)
    n1 = np.size(scr)
    #scrn = scr[0:n]
    srt_idx = np.argsort(scr)
    srt_prb = prb[srt_idx]
    srt_scr = np.sort(scr)
    total = 0
    i = 0
    while i in range(n1):
    #    total+=1/n1
        total+= srt_prb[i]
        if total>=(1-alpha):
            break
        else:
            i+=1
    #rnk = np.sum(prb[scr[-1]>=scr])
    #rnk = np.sum(scr[-1]>=scr)/np.size(scr)
    #return rnk
    if i==n1:
        return srt_scr[i-1]
    else:
        return srt_scr[i]
    
def train_data_gen(n):
    x = np.random.normal(mup,sp,n)
    #y = -x + x**3 + np.random.normal(0,se,n)
    y = x*theta_str +  sig_x(x)*np.random.normal(0,se,n)
    
    return y, x

def test_data_gen(n):
    x = np.random.normal(muq,sq,n)
    #y = -x + x**3 + np.random.normal(0, se, n)
    y = x*theta_str +  sig_x(x)*np.random.normal(0,se,n)
    
    return y, x

def conf_intr(Precs, rho, theta_ls, Yu, Yl, X, y, x_str, alpha, wt_trn):
    
    #parameter estimate from data
    #Precs, rho, theta_ls = ls_estimator_trn(X,y)
    #prediction at x_str
    phix_str = phi(x_str)
    y_hat = theta_ls.T.dot(phix_str)
    #evaluate weight at augment point
    wt = weight(x_str)
    wtaug = np.append(wt_trn, wt) 
    #upper end of interval
    #use estimate at x_str as the initial lower limit
    y_low = y_hat 
    #check for whether the upper limit of the interval has been found
    ci_upp = False 
    #point to augment as midpoint of y_low and Yu
    y_tilda = point2aug(y_low,Yu) 
    while ci_upp==False:
        #augmented data
        Xaug, yaug = aug_point(X,y,y_tilda,phix_str)
        #estimate with augmented data
        #theta = ls_estimator(Xaug, yaug)
        #find weight at augment point 
        theta = update_ls(Precs, rho, phix_str, y_tilda, wt)
        #non conformity score
        score = non_cnf_scr(Xaug, yaug, theta)
        #1-alpha qauntile s.t Pr(Z<=qtile)>=1-alpha
        qtile = qaunt(wtaug, score, alpha)
        #if y_tilda in interval
        in_itr = score[-1]<=qtile
        #n1 = np.size(score)
        #in_itr = qtile*n1<=np.ceil(n1*(1-alpha))
        #change lower and upper limit for next point to be augmented
        if in_itr == True:
            y_low = y_tilda
        else:
            Yu = y_tilda
        #new point to be augmented    
        y_tilda_n = point2aug(y_low,Yu)
        #check if new point is very close to the previous augmented point
        #print("y_tilda:", y_tilda, ",New y_tilda:", y_tilda_n, ",y_low:",y_low)
        in_thr = np.abs(y_tilda_n-y_tilda)<=thr
        #if the previous point is in interval and the new point is close,
        #upper end of interval has been found hence terminate
        if in_thr==True and in_itr==True:
            ci_upp = True
        else:
            y_tilda = y_tilda_n
        
        #print("Upper limit : Threshold:", in_thr,", In Interval:", in_itr)
    
    yci_u = y_tilda
    
    #lower end interval
    #use estimate at x_str as the initial upper limit
    y_upp = y_hat 
    #check for whether the upper limit of the interval has been found
    ci_low = False 
    #point to augment as midpoint of Yl and y_upp
    y_tilda = point2aug(Yl,y_upp) 
    while ci_low==False:
        #augmented data
        Xaug, yaug = aug_point(X,y,y_tilda,phix_str)
        #estimate with augmented data
        #theta = ls_estimator(Xaug, yaug)
        theta = update_ls(Precs, rho, phix_str, y_tilda, wt)
        #non conformity score
        score = non_cnf_scr(Xaug, yaug, theta)
        #1-alpha qauntile s.t Pr(Z<=qtile)>=1-alpha
        qtile = qaunt(wtaug, score, alpha)
        #if y_tilda in interval
        in_itr = score[-1]<=qtile
        #change lower and upper limit for next point to be augmented
        if in_itr == True:
            y_upp = y_tilda
        else:
            Yl = y_tilda
        #new point to be augmented    
        y_tilda_n = point2aug(Yl,y_upp)
        #check if new point is very close to the previous augmented point
        in_thr = np.abs(y_tilda_n-y_tilda)<=thr
        #if the previous point is in interval and the new point is close,
        #upper end of interval has been found hence terminate
        #print("Lower limit : Threshold:", in_thr,", In Interval:", in_itr)
        if in_thr==True and in_itr==True:
            ci_low = True
        else:
            y_tilda = y_tilda_n
    
    yci_l = y_tilda

    return (np.append(yci_l, yci_u, axis=0)).reshape((1,2))

        
    
    
            
        
        