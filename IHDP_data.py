##############THIS CODE CORRESPONDS TO THE INFANT HEALTH DATA IN SECTION 4.2 OF THE PAPER#######
##############THE SETTINGS CAN BE CHANGED IN FILE ihdp.test.py###############################
############# CURRENTLY THE SETTINGS CORRESPOND TO FIG. 4E AND 4F IN THE PAPER###############
#######YOU ALSO NEED TO PROVIDE PATH TO THE ihdp_npci_1.csv file in the ihdp_test.py FILE####
#######THEN SIMPLY RUN TO GET THE PLOTS IN FIGURE 4E AND 4F##########

############################
import random
import numpy as np
import ihdp_test as ihdp
############################
##############INITIALIZATION###################
#number of data points and number of covariates
(n, p) = (747, 25)
#set numpy seed
seed = 0
################################################
########READ DATA###################
#weights used in IHDP data generator
val_beta = np.array([0,0.1,0.2,0.3,0.4])
#probabilities to sample beta with
prob_beta = np.array([0.6,0.1,0.1,0.1,0.1])
np.random.seed(3)
beta = ihdp.draw_beta(val_beta, prob_beta, p)
omega = ihdp.means(beta)
#generate data
y, x, Z, Zaug, mu0, mu1 = ihdp.generate_data(n, beta, omega)
y = -1*y
mu0 = -1*mu0
mu1 = -1*mu1
#split intro training and test
ntrain = 600
random.seed(10)
y_train, x_train, Z_train, y_test, x_test, Z_test, mu0_train, mu1_train = ihdp.split_data(y, x, Zaug, mu0, mu1, ntrain)
####################################
#########AUTOENCODER################
#from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
#set tensorflow seed
tf.compat.v1.set_random_seed(seed)
# this is the size of our encoded representations
encoding_dim = 4
#dimension of covariates
d = 25
#place holder for covariate
input_img = Input(shape=(d,))
#encoder layer
encoded_c = Dense(encoding_dim, activation = 'elu')(input_img)
#decoder layer
decoded_c = Dense(d, activation = 'linear')(encoded_c)
#
autoencoder_trn_c = Model(input_img, decoded_c)
#
encoder_trn_c = Model(input_img, encoded_c)
#
decoder_input_c = Input(shape=(encoding_dim,))
#
decoder_layer_trn_c = autoencoder_trn_c.layers[-1]
#
decoder_trn_c = Model(decoder_input_c, decoder_layer_trn_c(decoder_input_c))
#compile the model
autoencoder_trn_c.compile(loss='mean_squared_error', optimizer='adam')
#fit autoencoder
autoencoder_trn_c.fit(Z_train[:,1:], Z_train[:,1:], epochs=150, batch_size=20, shuffle=True)
#latent variable
latent_h =  encoder_trn_c.predict(Z_train[:,1:])
#reconstruct
#temp2 = L4_trn.predict(Z_train[:,1:])
Z_tilda = decoder_trn_c.predict(latent_h)
#error
error = Z_tilda-Z_train[:,1:]
#########################################
#############FIT GMM#####################
import conformal_under_covariate_shift_scm as func_scm
num_mix = 4
train_trt = (x_train==0).reshape((ntrain,))
train_utrt = ~train_trt
#p(z|x=0)
weights_0, means_0, covars_0 = func_scm.fit_gmm(num_mix, latent_h[train_utrt,:])
#p(z|x=1)
weights_1, means_1, covars_1 = func_scm.fit_gmm(num_mix, latent_h[train_trt,:])
#append weights, means and covars in a single array
weights = np.append([weights_0],[weights_1],axis=0)
means = np.append([means_0],[means_1],axis=0)
covars = np.append([covars_0],[covars_1],axis=0)
##########################################
########ESTIMATE p(x)############
px_hat = func_scm.est_px(x_train)
#################################
###EVALUATE CONFORMAL INTERVAL FOR EACH INDIVIDUAL TREATEMENT 0 AND 1###
#basis matrix of treatment
Phi = func_scm.phi_mat_treat(x_train)
import conformal_with_2D_covariates as func2d
import conformal_under_covariate_shift as func
#nos. of test points
ntest = n-ntrain
#parameters for conformal interval
(Yu, Yl, alpha, thr) = (100, -100, 0.2, 0.1)
#latent variable for test covariates
latent_h_test = encoder_trn_c.predict(Z_test[:,1:])
#array for conformal interval for individual action
CI_test = np.zeros((ntest, 2, 2))
CI_train = np.zeros((ntrain,2,2))
#loop over treatments
for j in range(2):
    #policy 1(x==j)
    test_policy = j*np.ones((ntrain, 1))
    #weights w(x,z) for training data and above test policy
    wt_trn, _ , _ = func2d.weights_appx_latent(x_train, latent_h, latent_h, test_policy,
                                               px_hat, weights, means, covars)
    #Esimate theta on training data
    Precs, rho,  theta = func.ls_estimator_trn(Phi, y_train, wt_trn)
    #loope over test data
    count = 0
    for i in range(ntest):
        x_test_under_above = np.array([j])
        CI_test[i,j] = func2d.conf_intr_scm_latent(Precs, rho, theta, px_hat, weights, means,
                                              covars, wt_trn, Yu, Yl, Phi, y_train, x_test_under_above,
                                              latent_h_test[i], latent_h_test[i].reshape((1,encoding_dim)),
                                              np.array([j]), alpha, thr, False)
    #interval treating the test data as training                        
    for i in range(ntrain):
        x_test_under_above = np.array([j])
        CI_train[i,j] = func2d.conf_intr_scm_latent(Precs, rho, theta, px_hat, weights, means,
                                              covars, wt_trn, Yu, Yl, Phi, y_train, x_test_under_above,
                                              latent_h[i], latent_h[i].reshape((1,encoding_dim))
                                              ,np.array([j]), alpha, thr, False)
####################################################################
#######ROBUST EXPOSURE########
x_str, temp = func2d.robust_exposure(CI_test)  
x_str_train, temp = func2d.robust_exposure(CI_train)      
####################################### 

####COMPUTE THE CONFORMAL INTERVAL FOR THE ROBUST POLICY
wt_trn, _, _ = func2d.weights_appx_latent(x_train, latent_h, latent_h, x_str_train,
                                               px_hat, weights, means, covars)
Precs, rho, theta = func.ls_estimator_trn(Phi, y_train, wt_trn)
CI_robust = np.zeros((ntest,2))
for i in range(ntest):
    CI_robust[i] = func2d.conf_intr_scm_latent(Precs, rho, theta, px_hat, weights, means,
                                              covars, wt_trn, Yu, Yl, Phi, y_train, x_str[i],
                                              latent_h_test[i], latent_h_test[i].reshape((1,encoding_dim))
                                              ,x_str[i], alpha, thr, False) 
###########################################################

###COMPUTE THE CONFORMAL INTERVAL FOR THE LOGGINF POLICY###
CI_logg = np.zeros((ntest,2))
wt_trn,_,_ = func2d.weights_appx_latent(x_train, latent_h, latent_h, x_train, 
                                        px_hat, weights, means, covars)
Precs, rho, theta = func.ls_estimator_trn(Phi, y_train, wt_trn)
for i in range(ntest):
    CI_logg[i] =  func2d.conf_intr_scm_latent(Precs, rho, theta, px_hat, weights, means,
                                              covars, wt_trn, Yu, Yl, Phi, y_train, x_test[i],
                                              latent_h_test[i], latent_h_test[i].reshape((1,encoding_dim))
                                              , x_test[i], alpha, thr, False) 
############################################################
####COMPUTE THE MAXIMUM FOR THE ROBUST AND LOGGIN POLICY####
y_alpha_rob = np.max(CI_robust, axis=1)
y_alpha_logg = np.max(CI_logg, axis=1)
############################################################    
####GENERATE OUTCOME FOR TEST COVARIATE AND ABOVE ROBUST EXPOSURE###
y_test_rob = np.zeros((ntest, 1))
for i in range(ntest):
    y_test_rob[i],_,_ = ihdp.gen_outcome(x_str[i], Z_test[i], Z[i], beta, omega)

y_test_rob = -1*y_test_rob   

#####Linear model#######
beta0, beta1 = ihdp.learn_linear_model(y_train, x_train, Z_train)
#Evaluate exposure on test data
x_lin,_, _ = ihdp.eval_linear_model_policy(beta0, beta1, Z_test)
#
y_test_lin = np.zeros((ntest,1))
for i in range(ntest):
    y_test_lin[i],_,_ = ihdp.gen_outcome(x_lin[i], Z_test[i], Z[i], beta, omega)

y_test_lin = -1*y_test_lin 
#######################################
#####CCDF#####
Np = 500    
yax = np.linspace(np.min([np.min(y_test), np.min(y_test_rob)]),
                  np.max([np.max(y_test), np.max(y_test_rob)]),Np)    
ccdf_logg = np.zeros((Np,1))
ccdf_rob = np.zeros((Np,1))
ccdf_lin = np.zeros((Np,1))
for i in range(Np):
    ccdf_logg[i] = np.sum(y_test>yax[i])/ntest
    ccdf_rob[i] = np.sum(y_test_rob>yax[i])/ntest
    ccdf_lin[i] = np.sum(y_test_lin>yax[i])/ntest
###############
######PLOT CCDF#########
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
#    
fig = plt.figure(figsize=(8,6))
plt.plot(yax, ccdf_logg, c = 'b', linewidth=2, label=(r'$x\sim p(x|z)$'))
plt.plot(yax, ccdf_rob, c = 'r', linewidth = 2, label = (r'$x=\pi_{\alpha}(z)$'))
plt.plot(yax, ccdf_lin, c ='green', linewidth = 2, label=(r'$x=\pi_{\gamma}(z)$'))
plt.plot(yax,0.2*np.ones((Np,1)),'--', c='k')
plt.xlabel(r'$\tilde{y}$', fontsize=20)
plt.ylabel(r'$P\{y>\tilde{y}\}$', fontsize=20)
plt.grid()
plt.xlim(-8,0)
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.9))
plt.show()
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\ihdp\\ccdf_case1.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\ihdp\\ccdf_case2.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\ihdp\\ccdf_case3.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\ihdp\\ccdf_case5.pdf',format='pdf')
###########################
####PlOT MAXIMUM OF INTERVALS####
z1 = Z_test[:,1]
z1_sort = np.sort(z1)
idx = np.argsort(z1)
y_alpha0 = CI_test[:,0,1]
y_alpha1 = CI_test[:,1,1]

fig = plt.figure(figsize=(8,6))
plt.plot(z1_sort, y_alpha0[idx], c='b',linewidth=2, 
         label=(r'$y_{\alpha}(0,z)$'))
plt.plot(z1_sort, y_alpha1[idx], c='k',linewidth=2, 
         label=(r'$y_{\alpha}(1,z)$'))
plt.xlabel(r'$z_{1}$',fontsize=20)
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.8))
plt.grid()
plt.show()
#plt.savefig('C:\\OU_latest\\Off_policy_idea\\figs\\ihdp\\y_alpha_case3.pdf',format='pdf')

####PLOT EXOPSURES#########
fig = plt.figure(figsize=(8,6))
plt.scatter(Z_test[:,1],x_str, c='k', marker = 'x', s=100, 
            label = 'Robust exposure')
plt.scatter(Z_test[:,1], x_test, c='darkorange', marker = 'o', s=25, 
            label = ('Logging exposure'))
plt.scatter(Z_test[:,1], x_lin, marker='^', c='green', s = 50, label=(r'mean optimal'))
plt.xlabel(r'$z_{1}$', fontsize=20)
plt.ylabel('Treatment',fontsize=20)
plt.grid()
fig.legend(loc='upper right',bbox_to_anchor=(0.9,0.8))    
plt.show()
#plt.savefig('C:\\OU_latest\\Off_policy_idea\\figs\\ihdp\\exposures_case5.pdf',format='pdf')

####PLOT y_alpha_robust and y_alpha_logg#########
fig= plt.figure(figsize=(8,6))
plt.scatter(Z_test[:,5], Z_test[:,6], c=y_alpha_rob, marker='o', s=100, label=(r'Robust'), cmap='coolwarm')    
#plt.scatter(Z_test[:,1],Z_test[:,2],c=y_alpha_logg, marker='o', s=25, label = (r'Logging'))
plt.colorbar()
plt.grid()
plt.xlabel(r'Neo natal index',fontsize=20)
plt.ylabel(r'Mother age',fontsize=20)
plt.show()
#plt.xlabel(r'Birth weight',fontsize=20)
#plt.ylabel(r'Head circumference',fontsize=20)
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\ihdp\\y_alpha_robust_case1.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\ihdp\\y_alpha_robust_case2.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\ihdp\\y_alpha_robust_case3.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\ihdp\\y_alpha_robust_case5b.pdf',format='pdf')

####PLOT histogram of training data
#fig = plt.figure(figsize=(8,6))
#plt.hist(mu0_train-mu1_train, density=True)
#plt.hist(, density=True, label = 'Treated')
#plt.grid()
#fig.legend()
#plt.xlabel(r'$E[y|x=0,z]-E[y|x=1,z]$', fontsize=20)
#plt.ylabel('Density', fontsize=20)
####PLOT UPPER LIMIT FOR ROBUST AND LOGG EXPOSURE####

    
#autoencoder for binary covariates
#seed
#tf.compat.v1.set_random_seed(seed)
#
#encoding_dim = 1
#dimension of covariates
#d = 19
#place holder for covariate
#input_img = Input(shape=(d,))
#encoder layer
#encoded_b = Dense(encoding_dim, activation = 'elu')(input_img)
#decoder layer
#decoded_b = Dense(d, activation = 'linear')(encoded_b)
#
#autoencoder_trn_b = Model(input_img, decoded_b)
#
#encoder_trn_b = Model(input_img, encoded_b)
#
#decoder_input_b = Input(shape=(encoding_dim,))
#
#decoder_layer_trn_b = autoencoder_trn_b.layers[-1]
#
#decoder_trn_b = Model(decoder_input_b, decoder_layer_trn_b(decoder_input_b))
#compile
#autoencoder_trn_b.compile(loss='mean_squared_error', optimizer='adam')
#fit
#autoencoder_trn_b.fit(Z[:,6:], Z[:,6:], epochs = 200, batch_size=20, shuffle = True)
#latent
#latent_h_b = encoder_trn_b.predict(Z[:,6:])
#reconstruct
#Z_tilda_b = decoder_trn_b.predict(latent_h_b)
#error
#error_bin = Z_tilda_b-Z[:,6:]
#plot histogram of reconstruction error
#for col in range(25):
#    fig = plt.figure()
#    plt.hist(error[:, col], density=True, label = 'Histogram of error')
#    plt.grid()
#    plt.legend()
#    plt.xlabel(r'$c-\hat{c}$')
    #print("[", np.max(Z_train[:,col+1]), ", ", np.min(Z_train[:,col+1]),"]")
    
#plot latent variables
#fig = plt.figure()
#plt.scatter(latent_h[:,0], latent_h[:,1], label='latent variable')
#plt.grid()
