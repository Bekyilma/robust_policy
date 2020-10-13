#########THIS CODE CORRESPONDS TO SECTION 4.1 SYNTHETIC DATA IN THE PAPER#########

#####THE PARAMETERS ARE SET IN THE FILE conformal_with_2D_covariates.py AS PER
                    #####THE SETTINGS IN THE PAPER###########
                    
#######SIMPLY RUN TO PRODUCE RESULTS IN FIGURE 3 EXCEPT FOR FIGUERE 3D############                    

import conformal_under_covariate_shift as func
import conformal_with_2D_covariates as func2d
import conformal_under_covariate_shift_scm as func_scm
import numpy as np
#from scipy.stats import norm

#########set seed############
seed = 0
np.random.seed(seed)
######generate training data##########
n = 200
y,x,c = func2d.SCM(n)
#Extract columns
#age covariate
c1 = c[:,0]
c1 = c1.reshape(-1,1)
#gender covariate
c2 = c[:,1]
c2 = c2.reshape(-1,1)
######Learn distribution p(z|x)#######
#p(z_age|z_gender, x)
mu = np.zeros((2,2))
s = np.zeros((2,2))
p = np.zeros((2,1))
#males and untreated
#mu[0,0], s[0,0] = norm.fit(c1[np.logical_and(c2==0, x==0)])
_, mu[0,0], s[0,0] = func_scm.fit_gmm(1,c1[np.logical_and(c2==0, x==0)].reshape(-1,1))
#males and treated
#mu[0,1], s[0,1] = norm.fit(c1[np.logical_and(c2==0, x==1)])
_, mu[0,1], s[0,1] = func_scm.fit_gmm(1,c1[np.logical_and(c2==0, x==1)].reshape(-1,1))
#females and untreated
#mu[1,0], s[1,0] = norm.fit(c1[np.logical_and(c2==1, x==0)])
_, mu[1,0], s[1,0] = func_scm.fit_gmm(1,c1[np.logical_and(c2==1, x==0)].reshape(-1,1))
#females and treated
#mu[1,1], s[1,1] = norm.fit(c1[np.logical_and(c2==1, x==1)])
_, mu[1,1], s[1,1] = func_scm.fit_gmm(1,c1[np.logical_and(c2==1, x==1)].reshape(-1,1))
#gmm returns variances so take sqrt for standard deviation
s = np.sqrt(s)
#p(z_gender|x)
#females|untreated
p[0] = np.sum(c2[x==0]==1)/np.sum(x==0)
#females|treated
p[1] = np.sum(c2[x==1]==1)/np.sum(x==1)
###########estimate p(x)###########
px_est = func_scm.est_px(x)
#######Generate test covariates##########
ntest = 100
#draw covariates from distribution q(z) = p(z)
ctest = func2d.test_dist_pz_gen(ntest, px_est, mu, s, p)
ctest_grid = func2d.test_data_on_grid(ntest)
xtest_logg = func2d.train_exposure_gen(ctest)
ytest_logg = func2d.outcome_gen(xtest_logg, ctest)        
#######form basis matrix###########
Phi = func_scm.phi_mat_treat(x)
######Evaluate conformal interval for each treatment############
#number of treatments
Na = 2
#conformal interval 
CI_test = np.zeros((ntest,2,Na))    
CI_test_grid = np.zeros((ntest,2,Na))    
#parameters for conformal interval
(Yu, Yl, alpha, thr) = (30, -30, 0.2, 0.5)
#
for j in range(Na):
    #test policy
    test_treat = j*np.ones((ntest,1))
    #approximate weights using in latent space for training data
    wt_trn = func2d.weights_appx_dist(c, x, j*np.ones((n,1)), mu, s, p, px_est)
    ###############estimate theta using LS######################
    Precs, rho, theta_ls = func.ls_estimator_trn(Phi, y, wt_trn)
    ############################################################
    #treatment of test covariates under test policy
    xtest_a = j*np.ones((ntest,1))
    #interval for ctest
    for i in range(ntest):
    #evaluate conformal interval
        CI_test[i,j] = func2d.conf_intr_scm_dist(Precs, rho, theta_ls, Phi, y, wt_trn, 
                       xtest_a[i], ctest[i].reshape((1,2)), mu, s, p, px_est, test_treat[i], 
                       Yu, Yl, alpha, thr)
        CI_test_grid[i,j] = func2d.conf_intr_scm_dist(Precs, rho, theta_ls, Phi, y, wt_trn, 
                       xtest_a[i], ctest_grid[i].reshape((1,2)), mu, s, p, px_est, test_treat[i], 
                       Yu, Yl, alpha, thr)
        
#robust exposure
xtest_robust, temp = func2d.robust_exposure(CI_test)
xtest_grid_robust, temp = func2d.robust_exposure(CI_test_grid)
#corresponding outcome
ytest_robust = func2d.outcome_gen(xtest_robust, ctest)    

###########linear model#############
beta0, beta1 = func2d.learn_linear_model(y, x, c)
#optimal policy with the linear model
exp_opt, mu0_lin, mu1_lin = func2d.eval_linear_model_policy(beta0, beta1, ctest)
exp_opt_grid, _, _ = func2d.eval_linear_model_policy(beta0, beta1, ctest)
#corresponding outcome
ytest_lin_opt = func2d.outcome_gen(exp_opt, ctest)
    
#################################################################################################################
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('legend', fontsize=20)

#######indicators for training data#############
treated = x==1
untreated = ~treated
female = c2==1
male = ~female
trt_fem = np.sum(treated*female)
trt_mal  = np.sum(treated*male)
utrt_fem = np.sum(untreated*female)
utrt_mal = np.sum(untreated*male)
#######indicators for test data##########
c1_str = ctest[:,0]
c1_str = c1_str.reshape(-1,1)
c2_str = ctest[:,1]
c2_str = c2_str.reshape(-1,1)
fem_str = c2_str==1
mal_str = ~fem_str
#########################################
c1_str_grd = ctest_grid[:,0]
c1_str_grd = c1_str_grd.reshape(-1,1)
c2_str_grd = ctest_grid[:,1]
c2_str_grd = c2_str_grd.reshape(-1,1)
fem_str_grd = c2_str_grd==1
mal_str_grd = ~fem_str_grd
##########################################

#plot robust policy vs loggin policy vs mean optimal policy
px1_cstr = func2d.train_exposure_dist(np.ones((ntest,1)), ctest_grid)
px1_cstr_mal = px1_cstr[mal_str_grd]
px1_cstr_fem = px1_cstr[fem_str_grd]
xstr_ht_mal = xtest_grid_robust[mal_str_grd]
xstr_ht_fem = xtest_grid_robust[fem_str_grd]
exp_lin_mal = exp_opt_grid[mal_str_grd]
exp_lin_fem = exp_opt_grid[fem_str_grd]
c1_str_mal_grd = c1_str_grd[mal_str_grd]
c1_str_fem_grd = c1_str_grd[fem_str_grd]


#males
fig = plt.figure(figsize=(8,6))
plt.plot(c1_str_mal_grd, px1_cstr_mal, '--',c = 'b', linewidth = 2, label = (r'$p(x=1|z)$'))
plt.plot(c1_str_mal_grd, xstr_ht_mal, c = 'r', linewidth=2, label = (r'$\pi_{\alpha}(z)$') )
plt.plot(c1_str_mal_grd, exp_lin_mal, c='green', linewidth=2, label=(r'$\pi_{\gamma}(z)$'))
plt.xlabel(r'Age', fontsize=20)
plt.grid()
plt.ylim((-0.1,1.05))
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.7))
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\Numerical_example\\p_vs_pi_mal_lin.pdf',format='pdf')

#females
fig = plt.figure(figsize=(8,6))
plt.plot(c1_str_fem_grd, px1_cstr_fem, '--', c='b', linewidth=2, label=(r'$p(x=1|z)$'))
plt.plot(c1_str_fem_grd, xstr_ht_fem, c = 'r', linewidth=2, label = (r'$\pi_{\alpha}(z)$'))
plt.plot(c1_str_fem_grd, exp_lin_fem, c='green', linewidth=2, label=(r'$\pi_{\gamma}(z)$'))
plt.xlabel(r'Age', fontsize=20)
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.7))
plt.grid()
plt.ylim((-0.1,1.05))
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\Numerical_example\\p_vs_pi_fem_lin.pdf',format='pdf')

##plot y_{\alpha}
#c1_str_mal = c1_str[mal_str]
#c1_str_fem = c1_str[fem_str]
#idx_mal = np.argsort(c1_str_mal)
#idx_fem = np.argsort(c1_str_fem)

#y_alpha0 = CI_test[:,0,1]
#y_alpha0 = y_alpha0.reshape(-1,1)
#y_alpha0_mal = y_alpha0[mal_str]
#y_alpha0_fem = y_alpha0[fem_str]
#y_alpha1 = CI_test[:,1,1]
#y_alpha1 = y_alpha1.reshape(-1,1)
#y_alpha1_mal = y_alpha1[mal_str]
#y_alpha1_fem = y_alpha1[fem_str]

#males
#fig = plt.figure(figsize=(8,6))
#plt.plot(c1_str_mal[idx_mal], y_alpha0_mal[idx_mal], c='orangered', linewidth=2, label=(r'$x=0$'))
#plt.plot(c1_str_mal[idx_mal], y_alpha1_mal[idx_mal], c='grey', linewidth = 2, label = ('$x=1$'))
#plt.scatter(c1[treated*male], 10*np.ones((trt_mal,1)), c='grey', marker = '^', s = 100)
#plt.scatter(c1[male*untreated], 10*np.ones((utrt_mal,1)), c='orangered', marker = 'o', s = 20)
#plt.ylabel(r'$y_{\alpha}(x,z)$',fontsize=20)
#plt.xlabel('Age', fontsize=20)
#plt.grid()
#plt.ylim(2,30)
#plt.xlim((np.min(c1_str_mal),np.max(c1_str_mal)))
#fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\problem_illustration\\y_alpha_mal.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\Numerical_example\\y_alpha_mal_lin.pdf',format='pdf')

#females
#fig= plt.figure(figsize=(8,6))
#plt.plot(c1_str_fem[idx_fem], y_alpha0_fem[idx_fem], c='orangered', linewidth=2, label=(r'$x=0$'))
#plt.plot(c1_str_fem[idx_fem], y_alpha1_fem[idx_fem], c='grey', linewidth = 2, label = ('$x=1$'))
#plt.scatter(c1[treated*female], 10*np.ones((trt_fem,1)), c='grey', marker = '^', s = 100)
#plt.scatter(c1[female*untreated], 10*np.ones((utrt_fem,1)), c='orangered', marker = 'o', s = 20)
#plt.ylabel(r'$y_{\alpha}(x,z)$',fontsize=20)
#plt.xlabel('Age', fontsize=20)
#plt.ylim(2,30)
#plt.xlim((np.min(c1_str_fem),np.max(c1_str_fem)))
#plt.grid()
#fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\problem_illustration\\y_alpha_fem.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\Numerical_example\\y_alpha_fem_lin.pdf',format='pdf')

#plot means for females and males
#mu0_lin_mal = mu0_lin[mal_str]
#mu0_lin_fem = mu0_lin[fem_str]
#mu1_lin_mal = mu1_lin[mal_str]
#mu1_lin_fem = mu1_lin[fem_str]

#fig = plt.figure(figsize=(8,6))
#plt.plot(c1_str_mal[idx_mal], mu0_lin_mal, c='b', linewidth=2, label=(r'$\mu_{0}$ Male'))
#plt.plot(c1_str_mal[idx_mal], mu1_lin_mal, '--', c='b', linewidth=2, label=(r'$\mu_{1}$ Male'))
#plt.plot(c1_str_fem[idx_fem], mu0_lin_fem, c='r', linewidth=2, label=(r'$\mu_{0}$ Female'))
#plt.plot(c1_str_fem[idx_fem], mu1_lin_fem, '--', c='r', linewidth=2, label=(r'$\mu_{1}$ Female'))
#plt.grid()
#plt.xlabel('Age', fontsize=20)
#plt.ylabel('Mean', fontsize=20)
#fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

##############Evaluate CCDF##########################
#nos. of point to evaluate the CCDF at
Np = 100
yax = np.linspace(-30,30,Np)
#ccdf under robust policy
ccdf_robust = np.zeros((Np,1))
#ccdf under training policy
ccdf_train_policy = np.zeros((Np,1))
#ccdf lin optimal policy
ccdf_lin_opt = np.zeros((Np,1))
for i in range(Np):
    ccdf_robust[i] = 1/ntest*np.sum(ytest_robust>yax[i])
    ccdf_train_policy[i] = 1/ntest*np.sum(ytest_logg>yax[i])
    ccdf_lin_opt[i] = 1/ntest*np.sum(ytest_lin_opt>yax[i])
######################################################
#plot ccdfs         
fig = plt.figure(figsize=(8,6))
plt.plot(yax, ccdf_train_policy, label=(r'$x\sim~p(x|z)$'), linewidth=2, c='b')
plt.plot(yax, ccdf_robust, label=(r'$x=\pi_{\alpha}(z)$'),linewidth=2, c='r')
plt.plot(yax, ccdf_lin_opt, label = (r'$x=\pi_{\gamma}(z)$'), linewidth=2, c='green')
plt.plot(yax,alpha*np.ones(np.size(yax)),'--',c='k')
plt.xlim(-30,30)
plt.xlabel(r'$\tilde{y}$', fontsize=20)
plt.ylabel(r'$P\{y>\tilde{y}\}$', fontsize=20)
fig.legend(loc='upper right',bbox_to_anchor=(0.9,0.9))
plt.grid()
plt.show()
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\problem_illustration\\ccdf_illustration.pdf',format='pdf')
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\Numerical_example\\ccdf_lin.pdf',format='pdf')

####illustrating overlap#######
#px1_c = func2d.train_exposure_dist( np.ones((n,1)), c)
#px1_c_mal = px1_c[male]
#px1_c_fem = px1_c[female]

#c1_mal = c1[male]
#c1_fem = c1[female]
#idx_mal = np.argsort(c1_mal)
#idx_fem = np.argsort(c1_fem)

#fig = plt.figure(figsize=(8,6))
#plt.plot(c1_mal[idx_mal], px1_c_mal[idx_mal], c = 'b', linewidth = 2, label = (r'$p(x=1~|~z)$~Male'))
#plt.plot(c1_fem[idx_fem], px1_c_fem[idx_fem] , c='green', linewidth=2, label=(r'$p(x=1~|~z)$~Female'))
#plt.scatter(c1[treated*female],0.5*np.ones(trt_fem), s=100, marker = '^', c='grey')
#plt.scatter(c1[untreated*female],0.5*np.ones(utrt_fem), s=20, marker = 'o', c= 'orangered')
#plt.scatter(c1[treated*male],0*np.ones(trt_mal),s=100, marker = '^', c='grey')
#plt.scatter(c1[untreated*male], 0*np.ones(utrt_mal),s=20, marker = 'o', c= 'orangered')
#plt.grid()
#plt.xlabel(r'Age',fontsize=20)
#plt.ylabel(r'Probability', fontsize=20)
#fig.legend(loc='upper right',bbox_to_anchor=(0.9,0.9))
#plt.savefig('C:\\OU\\Off_policy_idea\\figs\\problem_illustration\\illustrating_overlap.pdf',format='pdf')


