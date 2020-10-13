
import numpy as np
import random
import csv
import conformal_with_2D_covariates as func2d
import conformal_under_covariate_shift as func
import conformal_under_covariate_shift_scm as func_scm
#import pathlib
import os, sys

#true treatment effect on the treated
tau = 1.5
#standard deviation of Osama_Uppsalatcome distribution
sigma_y0 = 5
sigma_y1 = 1
#values for sampling coefficient vector beta 
#data size and number of covariates
(n, p, w) = (747, 25, 0.5)

#####PROVIDE PATH TO FILE HERE#####
#data_folder = pathlib.Path('ihdp_test').parent.absolute()
#print('sys.argv[0] =', sys.argv[0])             
data_folder = os.path.dirname(sys.argv[0])        
path = data_folder+'/ihdp_npci_1.csv'

#read treatment and covariates
def read_treat_cov(path):
    #treatment variable
    x = np.zeros((n,1))
    #covariate
    Z = np.zeros((n, p))
    with open(path, 'r') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            x[i] = row[0]
            Z[i,:] = row[5:]
            i+=1
    #standardize the covariates        
    Zstd = (Z-np.mean(Z, axis=0))/np.std(Z,axis=0)
    #augment column of 1s
    Zaug = np.concatenate([np.ones((n,1)), Zstd], axis = 1)
    
    return x, Z, Zaug             

def draw_beta(val_beta, prob_beta, p):
    
    beta = np.random.choice(val_beta, p = prob_beta, size = (p, 1) )
    
    beta0 = np.random.uniform(-1,1,1).reshape(-1,1)
    
    beta = np.concatenate([beta0, beta], axis=0)
    
    return beta

def means(beta):
    
    x, Z, Zaug = read_treat_cov(path)
    
    #beta = draw_beta(val_beta, prob_beta, p)
    
    W = np.concatenate([np.zeros((n,1)), w*np.ones((n,p))], axis = 1)
    
    mu0 = np.exp((Zaug+W).dot(beta))
    
    mu1 = Zaug.dot(beta)
    
    omega = np.mean(mu1[x==1]-mu0[x==1])-tau
    
    return omega
         
def gen_outcome(x, z, zo, beta, omega):
    #generate Osama_Uppsalatcome for a given treatment x and standardized covariate z
    #z is a row vector
    z = z.reshape((1,p+1))
    wvec = np.concatenate([np.zeros((1,1)), w*np.ones((1,p))], axis = 1)
    mu0 = np.exp((z+wvec).dot(beta))
    mu1 = z.dot(beta)-omega
    if x==0:#not treated
        #weight vector added for non-linearity
        
        #sigma = sigma_y0*(np.abs(z[0,1])**2)
        #if zo[6]==1:
        #y = np.random.normal(mu[0], sigma_y0*10, 1)
        #elif zo[6]==0:
        y = np.random.normal(mu0[0], sigma_y0, 1)
    else:#treated
        #Osama_Uppsalatcome
        #if zo[0]>0:
        #    y = np.random.normal(mu[0], sigma_y1, 1)
        #elif zo[0]<=0:
        y = np.random.normal(mu1[0], sigma_y1, 1)
    
    return y, mu0, mu1    
        
def generate_data(n, beta, omega):
    #generate Osama_Uppsalatcome, treatment and augmented and standardized covariates except
    #that the first column is a column of 1s
    x, Z, Zaug = read_treat_cov(path)
    y = np.zeros((n,1))
    mu0 = np.zeros((n,1))
    mu1 = np.zeros((n,1))
    for i in range(n):
        y[i], mu0[i], mu1[i] = gen_outcome(x[i], Zaug[i,:], Z[i,:], beta, omega)
        
    return y, x, Z, Zaug, mu0, mu1    

def split_data(y, x, Zaug, mu0, mu1, ntrain):
    n = np.size(y)
    #sample indices for training data withOsama_Uppsalat replacement
    idx = np.array(random.sample(range(n), ntrain))
    lic = np.zeros(n, dtype = bool)
    for i in range(n):
        if np.sum(i==idx)==0:
            lic[i] = True
        else:
            lic[i] = False
    #lic = lic.reshape(Zaug.shape[0])        
    y_test = y[lic]
    x_test = x[lic]
    Z_test = Zaug[lic,:]    
    #mu0_test = mu0[lic]
    #mu1_test = mu1[lic]
    y_train = y[~lic]
    mu0_train = mu0[~lic]
    mu1_train = mu1[~lic]
    x_train = x[~lic]
    Z_train = Zaug[~lic,:]
    
    return y_train, x_train, Z_train, y_test, x_test, Z_test, mu0_train, mu1_train

def learn_linear_model(y, x, Z):
    #estimate parameters of model y = x*(z.T*\beta1) + (1-x)(z.T*\beta0)
    n = np.size(y) #phi(z) = [1, z]
    trt = x==1
    trt = trt.reshape((n,))
    utrt = ~trt
    y0 = y[utrt]
    Z0 = Z[utrt, :]
    y1 = y[trt]
    Z1 = Z[trt, :]
    beta0 = np.linalg.pinv(Z0).dot(y0)
    beta1 = np.linalg.pinv(Z1).dot(y1)
    
    return beta0, beta1

def eval_linear_model_policy(beta0, beta1, Z):
    n = Z.shape[0]
    exp_opt = np.zeros((n,1))
    y1 = Z.dot(beta1)
    y0 = Z.dot(beta0)
    for i in range(n):
        if y0[i]<=y1[i]:
            exp_opt[i] = 0
        else:
            exp_opt[i] = 1
            
    return exp_opt, y0, y1 

def wt_rand_cntrl(x, px_est, test_treat):
    n = np.size(x)
    den = np.zeros((n,1))
    for i in range(n):
        if x[i]==0:
            den[i] = px_est[0]
        elif x[i]==1:
            den[i] = px_est[1]
    
    num = func2d.test_expo_dist_delta_latent(x, test_treat)
    
    wt = num/den
    
    return wt

#conformal interval using latent space
def conf_intr_scm_latent(Precs, rho, theta_ls, px_est, wt_trn, Yu, Yl, Phi, y, 
                         x_str, test_treat, alpha, thr):
    
    #basis at x_str
    phix_str = func_scm.phi_treat(x_str)
    #prediction at x_str
    y_hat = theta_ls.T.dot(phix_str)
    #evaluate weight at x_str, c_str
    #wt, tmp = wt_true([x_str], [c_str])
    wt = wt_rand_cntrl(x_str, px_est, test_treat)
    #augment weight of x_str, c_str to weights of training points
    wtaug = np.append(wt_trn, wt) 
    #upper end of interval
    #use estimate at x_str as the initial lower limit
    y_low = y_hat 
    #check for whether the upper limit of the interval has been fOsama_Uppsaland
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
        #check if new point is very close to the previOsama_Uppsalas augmented point
        #print("y_tilda:", y_tilda, ",New y_tilda:", y_tilda_n, ",y_low:",y_low)
        in_thr = np.abs(y_tilda_n-y_tilda)<=thr
        #if the previOsama_Uppsalas point is in interval and the new point is close,
        #upper end of interval has been fOsama_Uppsaland hence terminate
        if in_thr==True and in_itr==True:
            ci_upp = True
        else:
            y_tilda = y_tilda_n
        
        #print("Upper limit : Threshold:", in_thr,", In Interval:", in_itr)
    
    yci_u = y_tilda
    
    #lower end interval
    #use estimate at x_str as the initial upper limit
    y_upp = y_hat 
    #check for whether the upper limit of the interval has been fOsama_Uppsaland
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
        #check if new point is very close to the previOsama_Uppsalas augmented point
        in_thr = np.abs(y_tilda_n-y_tilda)<=thr
        #if the previOsama_Uppsalas point is in interval and the new point is close,
        #upper end of interval has been fOsama_Uppsaland hence terminate
        #print("Lower limit : Threshold:", in_thr,", In Interval:", in_itr)
        if in_thr==True and in_itr==True:
            ci_low = True
        else:
            y_tilda = y_tilda_n
    
    yci_l = y_tilda

    return (np.append(yci_l, yci_u, axis=0)).reshape((1,2))
