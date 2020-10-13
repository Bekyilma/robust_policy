This repository is the official implementation of "Learning Robust Decision Policies from Observational Data"

-REQUIREMENTS

To install requirements
 
--pip install -r <path-to-requirements.txt>

-TRAINING

--Simulated data:

To get results for the synthetic data in section 4.1

python <path-to-robust_policy_not_latent.py>

This will produce results in Figure 3a-3c

--Real Data:

To get results for the real data in file ihdp_npci_1.csv in section 4.2

python <path-to-IHDP_data.py>

This will produce results in Figure 4 for the setting when sigma_1 = 1, sigma_0 = 5. These settings can be changed to get the different plots in Figure 4 in the ihdp_test.py file by changign variables sigma_y1 and sigma_y0. 
 
The csv data file should be in the same folder as all the other .py files

-DESCRIPTION OF PYTHON FILES

-robust_policy_not_latent.py: main script for simulated data

-IHDP_data.py: main script for real data

-conformal_under_covariate_shift.py, conformal_under_covariate_shift_scm.py, 
conformal_with_2D_covariates, ihdp_test.py : These files define different functions which are used by the main scripts for simulated and real data above. 