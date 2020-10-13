
#parameters for the training covariate distribution
#covariate = [age, gender] = [z1, z2], z2=0=male, z2=1=female
#probability of z2=1 i.e. p(z2=1)=0.5
s = 0.5
#parameters for p(z1|z2=0)
(mu0, s0) = (45, 5)
#parameter for p(z1|z2=1)
(mu1, s1) = (30, 5)
#parameters for exposure distribution
#p(x=1|z2=0) = p0, p(x=1|z2=1) = p1 
(p0, p1) = (0.2,0.92) #for equation (13) in the paper
#effect of gender
gender_eff = 0
#effect of exposure/treatment
(exp_eff_0, exp_eff_1) = (0,-5)
#variance of outcome 'y'
(sigma1, sigma2) = (0.2, 20)

#parameters for distribution q(c2) the proability of female
qs = 0.5
#parameters for distribution q(c1)
(muq0, sq0) = (45, 5)
(muq1, sq1) = (30, 5)
#parameters for conditional exposure distribution q(x|c)
##q(x=1|c2=0) = qp0, q(x=1|c2=1) = qp1 
(qp0,qp1) = (0.9, 0.1)
 
import numpy as np
import scipy.stats
import conformal_under_covariate_shift as func
import conformal_under_covariate_shift_scm as func_scm
#from sklearn.ensemble import RandomForestRegressor

def train_covariate_gen(n):
    #generate  n x d (d=2 here) matrix of covariates
    #p(c2)
    s_vec = s*np.ones((n,1))
    c2 = scipy.stats.bernoulli.rvs(s_vec)
    #p(c1|c2)
    c1 = np.zeros((n,1))
    for i in range(n):
        if c2[i]==0:
            c1[i]=np.random.normal(mu0,s0)
        else:
            c1[i]=np.random.normal(mu1,s1)
            
    c = np.append(c1, c2, axis = 1)
    return c

def train_exposure_prob(c):
    #probability of being treated based on covariates
    n = c.shape[0]
    px1_c = np.zeros((n,1))
    for i in range(n):
        if c[i][1]==1:#female
            #if c[i][0]>30:
            #    px1_c[i] = 1
            #else:
            #    px1_c[i] = 0
            px1_c[i] = p1*1/(1+np.exp(-(c[i][0]-20)/6))
        else:
            px1_c[i] = p0*1/(1+np.exp(-(c[i][0]-45)/2))
    
    return px1_c

def train_exposure_gen(c):
    #generate n x1 vector of exposure
    #c is n x d (d=2 here) matrix of covariates
    n = c.shape[0]
    x = np.zeros((n,1))
    px1_c = train_exposure_prob(c)
    for i in range(n):
        #if c[i][1]==1:#c2=1 (female)
        x[i] = scipy.stats.bernoulli.rvs(px1_c[i]) #assign treatment with probability p1
        #else: #otherwise c2=0 (male)
        #x[i] = scipy.stats.bernoulli.rvs(px1_c[]) #assign treatment with probability p0
    
    return x

def outcome_gen(x,c):
    #generate outcome corresponds to equation (14) in the paper
    n = np.size(x)
    y = np.zeros((n,1))
    beta0 = np.array([-46, 1, 0]).reshape(-1,1)
    beta1 = np.array([-45, 1, 0]).reshape(-1,1)
    mu = x*(np.append(np.ones((n,1)), c, axis=1).dot(beta1)) + \
        (1-x)*(np.append(np.ones((n,1)), c, axis=1).dot(beta0))
    sigma = np.zeros((n,1))
    for i in range(n): 
        if x[i]==0:
            sigma[i] = sigma2
        elif x[i]==1:
            sigma[i] = sigma1
            
    y = np.random.normal(mu, sigma)
        #y[i] = (c[i][0] + 70 + gender_eff*c[i][1] + exp_eff_1 + np.random.normal(0,0.2,1))*x[i] + \
        #(c[i][0] + 70 + gender_eff*c[i][1] + exp_eff_0+np.random.normal(0,0.2,1))*(1-x[i]) - 110 
    
    return y
    
def SCM(n):
    #generate outcome y 
    y = np.zeros((n,1))
    #covariates
    c = train_covariate_gen(n)
    #exposure
    x = train_exposure_gen(c)
    #outcome
    y = outcome_gen(x,c)
        
    return y, x, c

def learn_linear_model(y, x, z):
    #estimate parameters of model y = x*(z.T*\beta1) + (1-x)(z.T*\beta0)
    n = np.size(y)
    Phiz = np.append(np.ones((n,1)), z, axis=1) #phi(z) = [1, z]
    trt = x==1
    trt = trt.reshape((n,))
    utrt = ~trt
    y0 = y[utrt]
    Phiz0 = Phiz[utrt, :]
    y1 = y[trt]
    Phiz1 = Phiz[trt, :]
    beta0 = np.linalg.pinv(Phiz0).dot(y0)
    beta1 = np.linalg.pinv(Phiz1).dot(y1)
    
    return beta0, beta1

def eval_linear_model_policy(beta0, beta1, z):
    n = z.shape[0]
    exp_opt = np.zeros((n,1))
    Phiz = np.append(np.ones((n,1)),z,axis=1)
    y1 = Phiz.dot(beta1)
    y0 = Phiz.dot(beta0)
    for i in range(n):
        if y0[i]<=y1[i]:
            exp_opt[i] = 0
        else:
            exp_opt[i] = 1
            
    return exp_opt, y0, y1

def train_cov1_dist_male(c1):
    
    return scipy.stats.norm(mu0,s0).pdf(c1)

def train_cov1_dist_female(c1):
    
    return scipy.stats.norm(mu1,s1).pdf(c1)

def train_covariate_dist(c):
    #evaluate train covariate distribution at c n x d
    n = np.shape(c)[0]
    pc = np.zeros((n,1))
    for i in range(n):
        if c[i][1]==1:#c2=1 i.e. female
            pc[i] = scipy.stats.norm(mu1,s1).pdf(c[i][0]) * s
        else:#male
            pc[i] = scipy.stats.norm(mu0,s0).pdf(c[i][0]) * (1-s)
            
    return pc

def train_dist_zx(z, x, mu, s, p, px_est):
    #evaluates p(z,x) = p(z|x)p(x) at a given z and x
    n = np.size(x)
    p_zx = np.zeros((n,1))
    for i in range(n):
        gen = z[i][1] 
        trt = x[i]
        m = (1-gen)*(1-trt)*mu[0,0] + (1-gen)*trt*mu[0,1] + gen*(1-trt)*mu[1,0] +\
            gen*trt*mu[1,1]
        std = (1-gen)*(1-trt)*s[0,0] + (1-gen)*trt*s[0,1] + gen*(1-trt)*s[1,0] +\
            gen*trt*s[1,1]    
        pz1_z2x = scipy.stats.norm.pdf(z[i][0], m, std)
        pz2_x = (1-gen)*(1-trt)*(1-p[0])+(1-gen)*trt*(1-p[1]) + gen*(1-trt)*p[0]+\
                gen*trt*p[1]
        px = trt*px_est[1]+(1-trt)*px_est[0]
        p_zx[i] = pz1_z2x*pz2_x*px 
        
    return p_zx

def train_dist_pz(z, px_est, mu, s, p):
    #evaluates distribution p(z) = \sum_{x} p(z|x)p(x) at a given z
    #z size nxd
    sh = np.shape(z)
    n = sh[0]
    pz = np.zeros((n,1))
    for i in range(n):
        pz[i] = train_dist_zx(z[i].reshape((1,2)), np.array([0]),mu, s, p, px_est)+\
                train_dist_zx(z[i].reshape((1,2)), np.array([1]),mu, s, p, px_est)
                
    return pz

def test_data_on_grid(n):
    
    n_males = int(np.floor(n/2))
    n_females = n-n_males
    #age males
    c1_males = np.linspace(38,58,n_males).reshape(-1,1)
    #
    c2_males = np.zeros((n_males,1))
    c_males = np.append(c1_males, c2_males, axis=1)
    #age females
    c1_fem = np.linspace(20,40,n_females).reshape(-1,1)
    #
    c2_fem = np.ones((n_females,1))
    c_females = np.append(c1_fem, c2_fem, axis=1)
    
    c = np.append(c_males, c_females, axis = 0)
    
    return c

def test_dist_pz_gen(n, px_est, mu, s, p):
    #Generates data from p(z) = \sum_{x}p(z|x)p(x)
    #age covariate
    c1 = np.zeros((n,1))
    #gender covariate
    c2 = np.zeros((n,1))
    #which is less p(x==0) or p(x==1)
    idx_min = np.argmin(px_est)
    idx_max = np.argmax(px_est)
    for i in range(n):
        u =np.random.uniform(0,1,1)
        if u<=px_est[idx_min]:
            temp = scipy.stats.bernoulli.rvs(p[idx_min])
            c2[i] = temp
            c1[i] = np.random.normal(mu[temp, idx_min], s[temp, idx_min], 1)
        else:
            temp = scipy.stats.bernoulli.rvs(p[idx_max])
            c2[i] = temp
            c1[i] = np.random.normal(mu[temp, idx_max], s[temp, idx_max], 1)
    
    z = np.append(c1,c2, axis=1)
    
    return z

def train_exposure_dist(x, c):
    #evaluate training conditional distribution p(x|c)
    n = np.size(x)
    px_c = np.zeros((n,1))
    px1_c = train_exposure_prob(c)
    for i in range(n):
        #if c[i][1]==1:#female
        if x[i]==1:
            px_c[i]=px1_c[i]
        else:
            px_c[i]=1-px1_c[i]
        #else:#male
        #    if x[i]==1:
        #        px_c[i] = p0
        #    else:
        #        px_c[i] = 1-p0
                
    return px_c

def test_covariate_gen(n):
    #generate test covariate data from test distribution
    #gender
    qs_vec = qs*np.ones((n,1))
    #draw from q(c2)
    c2 = scipy.stats.bernoulli.rvs(qs_vec)
    if n==1:
        c2 = np.array([c2])
        c2 = c2.reshape(-1,1)
    #draw from q(c1|c2)
    c1 = np.zeros((n,1))
    for i in range(n):
        if c2[i]==0:
            c1[i]=np.random.normal(muq0,sq0)
        else:
            c1[i]=np.random.normal(muq1,sq1)
            
    c = np.append(c1, c2, axis = 1)
    return c

    return c  

def draw_from_gmm(n, weights, means, covars):
    #draws from a Gaussian mixture model specified by weight, mean and covariances
    #nos. of mixtures, 
    K = np.size(weights)
    sorted_weights = np.sort(weights)
    idx = np.argsort(weights)
    #array of uniform random numbers
    u = np.random.uniform(0,1,n)
    #array for latent variable
    h=np.zeros((n,1))
    for j in range(n):
        for i in range(K):
            if u[j]>=np.sum(sorted_weights[0:i]) and u[j]<np.sum(sorted_weights[0:i+1]):
                h[j] = scipy.stats.multivariate_normal.rvs(means[idx[i]], covars[idx[i]])
                break
    
    return h        
    
def test_covariate_gen_latent(n, px_est, weights, means, covars):
    #draws latent covariate h from q(h)=p(h)
    #sort px_est
    sorted_px = np.sort(px_est, axis=0)
    #sorted index
    idx = np.argsort(px_est, axis=0)
    #array of uniform random number
    u = np.random.uniform(0,1,n)
    #nos. of mixtures
    K = np.size(px_est)
    #array of latent variables
    H = []
    #
    h = np.zeros((n,1))
    #
    for i in range(K):
        H = H + [draw_from_gmm(n, weights[i], means[i], covars[i])]
        
    for j in range(n):
        for i in range(K):
            if u[j]>=np.sum(sorted_px[0:i]) and u[j]<np.sum(sorted_px[0:i+1]):
                h[j] = H[idx[i][0]][j]
                break
    return h
                  

def test_exposure_gen(c):
    #generate n x1 vector of test exposure
    #c is n x d (d=2 here) matrix of covariates
    n = c.shape[0]
    x = np.zeros((n,1))
    for i in range(n):
        if c[i][1]==1:#c2=1 (female)
            x[i] = scipy.stats.bernoulli.rvs(qp1) #assign treatment with probability qp1
        else: #otherwise c2=0 (male)
            x[i] = scipy.stats.bernoulli.rvs(qp0) #assign treatment with probability qp0
    
    return x

def test_exposure_gen_latent(c_tilda):
    #generate n x1 vector of test exposure based on c_tilda = decoder(latent_h)
    #c is n x d (d=2 here) matrix of covariates
    n = c_tilda.shape[0]
    x = np.zeros((n,1))
    for i in range(n):
        if c_tilda[i][1]==1:#c2=1 (female)
            x[i] = scipy.stats.bernoulli.rvs(qp1) #assign treatment with probability qp1
        else: #otherwise c2=0 (male)
            x[i] = scipy.stats.bernoulli.rvs(qp0) #assign treatment with probability qp0
    
    return x        

def test_cov1_dist_male(c1):
    
    return scipy.stats.norm(muq0,sq0).pdf(c1)

def test_cov1_dist_female(c1):
    
    return scipy.stats.norm(muq1,sq1).pdf(c1)    

def test_covariate_dist(c):
    #evaluate test covariate distribution at c n x d
    n = np.shape(c)[0]
    qc = np.zeros((n,1))
    for i in range(n):
        if c[i][1]==1:#c2=1 i.e. female
            qc[i] = scipy.stats.norm(muq1,sq1).pdf(c[i][0]) * qs
        else:#male
            qc[i] = scipy.stats.norm(muq0,sq0).pdf(c[i][0]) * (1-qs)
            
    return qc

def test_exposure_dist(x, c):
    #evaluate training conditional distribution p(x|c)
    n = np.size(x)
    qx_c = np.zeros((n,1))
    for i in range(n):
        if c[i][1]==1:#female
            if x[i]==1:
                qx_c[i]=qp1
            else:
                qx_c[i]=1-qp1
        else:#male
            if x[i]==1:
                qx_c[i] = qp0
            else:
                qx_c[i] = 1-qp0
                
    return qx_c

def test_covariate_probability(c,dc):
    #evaluates probablity Pr(c1<=C1<c1+dc1|C2=c2)*Pr(C2=c2) for the test distribution
    #dc1 is an array of small change in age covariate
    n = np.shape(c)[0]
    qc = np.zeros((n,1))
    for i in range(n):
        if c[i][1]==1:#c2=1 i.e. female
            qc[i] = (scipy.stats.norm(muq1,sq1).cdf(c[i][0]+dc[i][0])-\
            scipy.stats.norm(muq1,sq1).cdf(c[i][0])) * qs
        else:#male
            qc[i] = (scipy.stats.norm(muq0,sq0).cdf(c[i][0]+dc[i][0])-\
            scipy.stats.norm(muq0,sq0).cdf(c[i][0]))* (1-qs)
            
    return qc

def test_covariate_dist_latent(latent_h, px_est, weights, means, covars):
    #evaluates q(h) = sum_{x}p(h|x)p(x)
    sh = np.shape(latent_h)
    n = sh[0]
    d = sh[1]
    #linear weight vector w= [w1,w2]
    q_h = np.zeros((n,1))
    for i in range(n):
        q_h[i] = px_est[0]*func_scm.eval_gmm(latent_h[i].reshape((1,d)), weights[0], means[0], covars[0])+\
                px_est[1]*func_scm.eval_gmm(latent_h[i].reshape((1,d)), weights[1], means[1], covars[1])
    return q_h
    
def test_exposure_dist_latent(x, c_tilda):
    #evaluate q(x|h) using the decoded_c
    #size of array
    n = np.size(x)
    qx_h = np.zeros((n,1))
    for i in range(n):
        if c_tilda[i][1]==1: #implies female
            if x[i]==1:
                qx_h[i]=qp1
            else:
                qx_h[i]=1-qp1
        else:#male
            if x[i]==1:
                qx_h[i]=qp0
            else:
                qx_h[i]=1-qp0
      
    return qx_h

def test_expo_dist_delta_latent(x, test_treatment):
    #evaluates distribution q(x|h) = delta(x-test_treatment)
    n = np.size(x)
    qx_h = np.zeros((n,1))
    for i in range(n):
        if x[i] == test_treatment[i]:
            qx_h[i] = 1
     
    return qx_h

def robust_exposure(CI):
    #evaluates the optimal policy x*(h) given conformal interval for each 
    #individual action
    shp = np.shape(CI)
    #data size
    n = shp[0]
    CI_robust = np.zeros((n,2))
    x_str_h = np.zeros((n,1))
    for i in range(n):
        if np.max(CI[i,0])>np.max(CI[i,1]): #if max(CI) under policy 0 > max(CI) under policy 1
            x_str_h[i] = 1
            CI_robust[i] = CI[i,1]
        else:
            x_str_h[i] = 0
            CI_robust[i] = CI[i,0]
            
    return x_str_h, CI_robust
         
                    
def weights_true(x, c):
    #evaluates w(x,c)=q(x|c*q(c)/p(c|x)p(x)
    num = test_covariate_dist(c)*test_exposure_dist(x,c)
    den = train_covariate_dist(c)*train_exposure_dist(x, c)
    wt = num/den
    
    return wt, den, num

def weights_true_2(x, c, test_treat):
    #evaluates w(x,c)=q(x|c*q(c)/p(c|x)p(x)
    num = test_covariate_dist(c)*test_expo_dist_delta_latent(x, test_treat)
    den = train_covariate_dist(c)*train_exposure_dist(x, c)
    wt = num/den
    
    return wt, den, num

def weights_appx(x, c, latent_h, px_est, weights, means, covars):
    #evaluates w(x,c)=q(x|c*q(c)/p(h|x)p(x)
    num = test_covariate_dist(c)*test_exposure_dist(x,c)
    den = func_scm.eval_denOfwt(x, latent_h, px_est, weights, means, covars)
    wt = num/den
    
    return wt, den, num

def weights_appx_latent(x, c_tilda, latent_h, test_treat, px_est, weights, means, covars):
    #evaluates w(x,h)=q(x|decoder(latent_h))*q(h)/p(h|x)p(x)
    num = test_covariate_dist_latent(latent_h, px_est, weights, means, covars)*\
            test_expo_dist_delta_latent(x, test_treat) #test_exposure_dist_latent(x, c_tilda)
    den = func_scm.eval_denOfwt(x, latent_h, px_est, weights, means, covars)
    wt = num/den

    return wt, den, num    

def weights_appx_dist(z, x, test_treat, mu, s, p, px_est):
    #evauates weights when estimating distribution of age by 1D-Gaussian fit
    num = train_dist_pz(z, px_est, mu, s, p)*\
    test_expo_dist_delta_latent(x, test_treat)
    den = train_dist_zx(z, x, mu, s, p, px_est)
    wt = num/den
    
    return wt

def weights_apprx_prob(x, c, dc, latent_h, dh, px_est, weights, means, covars):
    #evaluates weights #evaluates w(x,c)=q(x|c*q(c)/p(h|x)p(x) using probabilty approximation
    num = test_exposure_dist(x,c)*test_covariate_probability(c,dc)
    den = func_scm.eval_denUsingProb(x, latent_h, dh, px_est, weights, means, covars)
    wt = num/den
    
    return wt, den, num     

#conformal interval based on w(x,c) = q(x|c)q(c)/p(h|x)p(h) with directly density evaluattion OR
#using probabilities in a small area            
def conf_intr_scm(Precs, rho, theta_ls, px_est, weights, means, covars, wt_trn,
                  Yu, Yl, Phi, y, x_str, c_str, latent_h_str, alpha, thr, wt_ind, dc, dh_str):
    
    #basis at x_str
    phix_str = func_scm.phi_treat(x_str)
    #prediction at x_str
    y_hat = theta_ls.T.dot(phix_str)
    #evaluate weight at x_str, c_str
    #wt, tmp = wt_true([x_str], [c_str])
    if wt_ind == False:
        wt, tmp, tmp2 = weights_apprx_prob([x_str], [c_str], [dc], latent_h_str, 
                                           dh_str, px_est, weights, means, covars)
        #wt,tmp, tmp2 = weights_appx([x_str], [c_str], latent_h_str, px_est, weights, means, covars)
    else:
        wt,tmp, tmp2 = weights_true([x_str],[c_str])
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

#conformal interval using latent space
def conf_intr_scm_latent(Precs, rho, theta_ls, px_est, weights, means, covars, wt_trn,
                  Yu, Yl, Phi, y, x_str, c_str_tilda, latent_h_str, test_treat, alpha, thr, wt_ind):
    
    #basis at x_str
    phix_str = func_scm.phi_treat(x_str)
    #prediction at x_str
    y_hat = theta_ls.T.dot(phix_str)
    #evaluate weight at x_str, c_str
    #wt, tmp = wt_true([x_str], [c_str])
    if wt_ind == False:
        wt, tmp, tmp2 = weights_appx_latent(x_str, c_str_tilda, latent_h_str, test_treat, 
                                            px_est, weights, means, covars)
    else:
        wt,tmp, tmp2 = weights_true_2([x_str],[c_str_tilda], [test_treat])
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

#conformal interval using not latent space
def conf_intr_scm_dist(Precs, rho, theta_ls, Phi, y, wt_trn, x_str, z_str, 
                       mu, s, p, px_est, test_treat, Yu, Yl, alpha, thr):
    
    #basis at x_str
    phix_str = func_scm.phi_treat(x_str)
    #prediction at x_str
    y_hat = theta_ls.T.dot(phix_str)
    #evaluate weight at x_str, c_str
    wt = weights_appx_dist(z_str, x_str, test_treat, mu, s, p, px_est)
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