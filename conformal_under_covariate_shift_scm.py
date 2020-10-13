# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:46:44 2020

@author: muhos601
"""

import numpy as np
import scipy.stats
import conformal_under_covariate_shift as func
from sklearn import mixture

#parameters of covariate distribution p(c)
(muc, sc) = (30, 10)
#treatment reduces outcome by 10 and not treating increases outcome by 5 
(mu1, mu0) = (-10, 5)
#noise in output
se = 0.5
#parameters for the test distribution of covariate
(muq, sq) = (10, 5)


def sigmoid(c):
    #evaluates sigmoid function at covariate c
    c = c.reshape(-1,1)
    shp = np.shape(c)
    sig = np.zeros((shp[0],1))
    for i in range(shp[0]):
        sig[i] = 1/(1+np.exp(-(c[i]-30)/5))
    
    return sig

def covariate_dist(c):
    #evaluates training covariate distribution p(c) at array c
    pc = scipy.stats.norm(muc,sc).pdf(c)
    
    return pc

def px_given_c(x, c):
    #evaluate true distribution p(x|c) at a given exposure x and covariate c
    px_gc = np.zeros((np.size(x),1))
    for i in range(np.size(x)):
        if x[i]==1:
            #probability of treatment assignment
            px_gc[i] = 1-sigmoid(c[i])
        else:
            #probability of not assigning treatment
            px_gc[i] = sigmoid(c[i])
        
    return px_gc    

def train_covariate_gen(n):
    #generates covariate from training distribution p(c)
    c = np.random.normal(muc, sc, n)
    c = c.reshape(-1,1)
    return c    

def train_exposure_gen(n, c):
    #generates exposure from bernoulli distribution with success probability 
    #p(x=1|c)=1-sigmoid(c)
    
    p = 1-sigmoid(c)
    #treatment
    x = scipy.stats.bernoulli.rvs(p)
    
    return x

def outcome_gen(x, c):
    
    y = (c + 70 + mu1)*x + (c + 70 + mu0)*(1-x) #+ np.random.normal(0, se, n)
    
    return y

def SCM(n):
    #Generate n data points from a structural causal model
    #generate covariate
    c = train_covariate_gen(n)
    #assign treatment 
    x = train_exposure_gen(n, c)
    #outcome
    y = outcome_gen(x,c)
    
    return y, c, x

def test_covariate_dist(c):
    #evaluate test distribution q(c) at covariat c
    qc = scipy.stats.norm(muq,sq).pdf(c)
    
    return qc 

def test_cond_dist(x,c):
    #evaluates test condition distribution q(x|c) at a given x and c
    #make it same as p(x|c)
    p = px_given_c(x, c)
    #another test policy-the probability of being treated is q(x=1|c) = 1
    #if x==1:
    #    qx1 = 1
    #    return qx1
    #else:
    #    qx0 = 0
    #    return qx0
    return p

def test_covariate_gen(n):
    #generate covariate from the test distribution q(c)
    c = np.random.normal(muq,sq,n)
    c = c.reshape(-1,1)
    
    return c

def test_exposure_gen(n, c):
    #either same as training bernoulli draw with success probability q(x=1|c)=1-sigmoid(c)
    p = 1-sigmoid(c)
    x = scipy.stats.bernoulli.rvs(p)
    #bernoulli draw with success probability q(x=1|c)=1
    #p = np.ones((n,1))
    #x = scipy.stats.bernoulli.rvs(p)
    
    return x

def fit_gmm(num_mix, data):
    #data should be n x d where d is dimension of covariates
    #define GMM
    gmm = mixture.GaussianMixture(n_components = num_mix, covariance_type = 'full')
    #fit GMM
    gmm.fit(data)
    #read mixture weights, means and variances
    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_
    
    return weights, means, covars

def eval_dergmm_latent(X, weights, means, covars):
    #X should be n x 1 shape, only for latent variable
    X = X.reshape(-1,1)
    sz = np.shape(X)
    pdf_gmm = np.zeros((sz[0],1))
    for j in range(np.size(weights)): 
        tmp_var = -weights[j]*scipy.stats.multivariate_normal.pdf(X, means[j], covars[j])
        tmp_var = tmp_var.reshape(-1,1)*(X-means[j])
        pdf_gmm += tmp_var
    
    return pdf_gmm

def eval_gmm(X, weights, means, covars):
    #X should be n x d shape
    shp = np.shape(X)
    pdf_gmm = np.zeros((shp[0],1))
    for j in range(np.size(weights)): 
        tmp_var = weights[j]*scipy.stats.multivariate_normal.pdf(X, means[j], covars[j])
        tmp_var = tmp_var.reshape(-1,1)
        pdf_gmm += tmp_var
    
    return pdf_gmm

def eval_cdf_gmm(h, dh, weights, means, covars):
    #evaluates probability of GMM Pr(h<H<=h)
    #only works with one-dimensional GMM for now
    h = h.reshape(-1,1)
    sz = np.shape(h)
    pr_gmm = np.zeros((sz[0],1))
    for j in range(np.size(weights)): 
        tmp_var = weights[j]*(scipy.stats.norm.cdf(h+dh, means[j], covars[j])-\
                         scipy.stats.norm.cdf(h, means[j], covars[j]))
        tmp_var = tmp_var.reshape(-1,1)
        pr_gmm += tmp_var
    
    return pr_gmm

def est_px(x):
    #estimate p(x==0) and p(x==1) from observed treatment vector x
    px_est = np.zeros((2,1))
    #probability of treated patients
    px_est[1] = np.sum(x==1)/np.size(x)
    #probability of untreated patients
    px_est[0] = 1-px_est[1]
    
    return px_est

def eval_denOfwt_usingDer(xtest, ctest, px_est, weights, means, covars):
    
    den = np.zeros((np.size(xtest),1))
    for i in range(np.size(xtest)):
        if xtest[i]==0:
            den[i] = px_est[0]*eval_dergmm_latent(ctest[i],weights[0], means[0], covars[0])
        else:
            den[i] = px_est[1]*eval_dergmm_latent(ctest[i],weights[1], means[1], covars[1])
            
    return den

def eval_denOfwt(xtest, ctest, px_est, weights, means, covars):
    
    shp = ctest.shape
    n = shp[0]
    d = shp[1]
    den = np.zeros((n,1))
    for i in range(np.size(xtest)):
        if xtest[i]==0:
            den[i] = px_est[0]*eval_gmm(ctest[i].reshape((1,d)),weights[0], means[0], covars[0])
        else:
            den[i] = px_est[1]*eval_gmm(ctest[i].reshape((1,d)),weights[1], means[1], covars[1])
            
    return den

def eval_denUsingProb(xtest, ctest, dh, px_est, weights, means, covars):
    #evaluates denomiator using probability approximations
    den = np.zeros((np.size(xtest),1))
    for i in range(np.size(xtest)):
        if xtest[i]==0:
            den[i] = px_est[0]*eval_cdf_gmm(ctest[i], dh[i], weights[0], means[0], covars[0])
        else:
            den[i] = px_est[1]*eval_cdf_gmm(ctest[i], dh[i], weights[1], means[1], covars[1])
            
    return den

def eval_numOfwt(xtest,ctest):
    num = np.zeros((np.size(xtest),1))
    for i in range(np.size(xtest)):
        num[i] = test_covariate_dist(ctest[i])*test_cond_dist(xtest[i],ctest[i])
        
    return num

def wt_appx(x, c, px_est, weights, means, covars):
    den_hat = eval_denOfwt(x, c, px_est, weights, means, covars)
    wt_hat = eval_numOfwt(x, c)/den_hat
    
    return wt_hat, den_hat

def wt_true(x,c):
    px_c = px_given_c(x,c)
    true_den = px_c*covariate_dist(c)
    wt = eval_numOfwt(x,c)/true_den

    return wt, true_den

def phi_treat(x):
    #x is a scalar exposure for a single datapoint
    phi = np.zeros((2,1))
    if x==0:
        phi[0]=1
    else:
        phi[1]=1

    return phi

def phi_mat_treat(x):
    #x is vector of exposure for all n data points
    n = np.size(x)
    Phi = np.zeros((n,2))
    for i in range(n):
        Phi[i] = phi_treat(x[i]).T
    
    Phi = Phi.T
    return Phi

def conf_intr_scm(Precs, rho, theta_ls, px_est, weights, means, covars,wt_trn,
                  Yu, Yl, Phi, y, x_str, c_str, alpha, thr):
    
    #basis at x_str
    phix_str = phi_treat(x_str)
    #prediction at x_str
    y_hat = theta_ls.T.dot(phix_str)
    #evaluate weight at x_str, c_str
    #wt, tmp = wt_true([x_str], [c_str])
    wt, tmp = wt_appx([x_str], [c_str], px_est, weights, means, covars)
    #augment weight of x_str, c_str to weights of training points
    wtaug = np.append(wt_trn, wt) 
    #upper end of interval
    #use estimate at x_str as the initial lower limit
    y_low = y_hat 
    #check for whether the upper limit of the interval has been found
    ci_upp = False 
    #point to augment as midpoint of y_low and Yu
    y_tilda = func.point2aug(y_low, Yu) 
    while ci_upp==False:
        #augmented data
        Phiaug, yaug = func.aug_point(Phi, y, y_tilda, phix_str)
        #update theta 
        theta = func.update_ls(Precs, rho, phix_str, y_tilda, wt)
        #non conformity score
        score = func.non_cnf_scr(Phiaug, yaug, theta)
        #1-alpha qauntile s.t Pr(Z<=qtile)>=1-alpha
        qtile = func.qaunt(wtaug, score, alpha)
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
        y_tilda_n = func.point2aug(y_low,Yu)
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
    y_tilda = func.point2aug(Yl,y_upp) 
    while ci_low==False:
        #augmented data
        Phiaug, yaug = func.aug_point(Phi,y,y_tilda,phix_str)
        #estimate with augmented data
        #theta = ls_estimator(Xaug, yaug)
        theta = func.update_ls(Precs, rho, phix_str, y_tilda, wt)
        #non conformity score
        score = func.non_cnf_scr(Phiaug, yaug, theta)
        #1-alpha qauntile s.t Pr(Z<=qtile)>=1-alpha
        qtile = func.qaunt(wtaug, score, alpha)
        #if y_tilda in interval
        in_itr = score[-1]<=qtile
        #change lower and upper limit for next point to be augmented
        if in_itr == True:
            y_upp = y_tilda
        else:
            Yl = y_tilda
        #new point to be augmented    
        y_tilda_n = func.point2aug(Yl,y_upp)
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
