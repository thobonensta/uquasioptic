import time

import numpy as np
import chaospy as cp
import math
from tqdm import tqdm

def lognormal_params_from_mean_std(m, s):
    # m = mean, s = std
    sigma2 = np.log(1 + (s**2)/(m**2))
    sigma  = np.sqrt(sigma2)
    mu     = np.log(m) - 0.5 * sigma2
    return mu, sigma
#################### Parameters (without uncertainty) ###############################
# Calculate the projection
freq_GHz = np.linspace(220, 330, 101, endpoint=True)
##### Parameters for the three layers MUT (mat/mid/mar) - mut = material under test
eps_mid = 1. - 0j


##################### Uncertainty over the parameters ###############################
# Model the variation for the air gap (logN since positive definite)
beta_1 = 30/100 # variation coefficient
m1 = 50e-6  # mean
sig1 = beta_1*m1 # standard-deviation
m_c1,sig_c1 = lognormal_params_from_mean_std(m1,sig1)


# Model the variation for the angle (N since can be positive or negative)
m2 = 0
sig2 = 0.1


# Model the variation for the thickness of the material thickness_mut = 0.00582
beta_3 = 1/100 # variation coefficient
m3 = 0.00582 # mean
sig3 = beta_3*m3 # standard-deviation
m_c3,sig_c3 = lognormal_params_from_mean_std(m3,sig3)

# # Model the variation for the eps of MUT eps_mut = 2. - 0.025j
# beta_re =0/100
# mre = 2
# sigre = beta_re*mre
# m_cre,sig_cre = lognormal_params_from_mean_std(mre,sigre)
#
# beta_im =0/100
# mim = 0.025
# sigim = beta_im*mim
# m_cim,sig_cim = lognormal_params_from_mean_std(mim,sigim)
eps_mut = 2. - 0.025j


# Use chaospy for the decomposition
t0 = time.perf_counter()
X1 = cp.LogNormal(m_c1,sig_c1)
X2 = cp.Normal(m2,sig2)
X3 = cp.LogNormal(m_c3,sig_c3)
# X4 = cp.LogNormal(m_cre,sig_cre)
# X5 = cp.LogNormal(m_cim,sig_cim)

# Joint distribution assuming independance
X = cp.J(X1,X2,X3)

# Decomposition using Wiener-Askey
p = 6
NbRV = 5
P = math.comb(p+NbRV,p)
decomp = cp.generate_expansion(p,X)
decomp = decomp.round(10)
# Gauss quadrature adapted to our joint distribution
d = 8
tg,wg = cp.generate_quadrature(d,X,rule="gaussian")
# Extract physical values
x1_vals = tg[0, :]  # quadrature nodes for X1
x2_vals = tg[1, :]  # quadrature nodes for X2
x3_vals = tg[2, :]  # quadrature nodes for X3
# x4_vals = tg[3, :]  # quadrature nodes for X4
# x5_vals = tg[4, :]  # quadrature nodes for X5

wg = wg[:, None]    # quadrature weigths

# Normalization
norms = np.array([cp.E(poly**2, X) for poly in decomp])

######################### PCE UQ ########################################################
# Compute the projection
from utils.PlaneWave import SimPlaneWaveList # The blackbox code we use here for test
lst_S11_PC = []
lst_S12_PC = []

for x1, x2, x3 in tqdm(zip(x1_vals, x2_vals, x3_vals)):
    thickness_mid = x1
    angle_deg = x2
    thickness_mut = x3
    lst_S11 = []
    lst_S12 = []
    for f_GHz in freq_GHz:
        # Plane wave model
        S11, S12, _, _ = SimPlaneWaveList(f_GHz, angle_deg, eps_mut, thickness_mut, eps_mid, thickness_mid)
        lst_S11.append(S11)
        lst_S12.append(S12)
    lst_S11_PC.append(lst_S11)
    lst_S12_PC.append(lst_S12)


coeffsS11 = []
for phi in decomp:
    # shape (n_freqs,)
    phi_vals = phi(*tg)[:, None]
    numers = np.sum(wg * lst_S11_PC * phi_vals, axis=0)
    coeffsS11.append(numers)

coeffsS11 = np.array(coeffsS11)/norms[:, None]

coeffsS12 = []
for phi in decomp:
    phi_vals = phi(*tg)[:, None]
    numers = np.sum(wg * lst_S12_PC * phi_vals, axis=0)
    coeffsS12.append(numers)

coeffsS12 = np.array(coeffsS12)/norms[:, None]
t_PCE = time.perf_counter() - t0
print('Time PCE : ',t_PCE)
S11_m = coeffsS11[0, :]
S12_m = coeffsS12[0, :]

S11_std = np.sum(coeffsS11[1:, :] ** 2, axis=-1)
S12_std = np.sum(coeffsS12[1:, :] ** 2, axis=-1)

######################## MONTE-CARLO ##########################################
t0 = time.perf_counter()
N_sample = 100000
# Create the samples for the different variables
thickness_mid_samples = np.random.lognormal(m_c1,sig_c1,int(N_sample))
angle_deg_samples = np.random.normal(m2,sig2,int(N_sample))
thickness_mut_samples = np.random.lognormal(m_c3,sig_c3,int(N_sample))

lst_S11 = []
lst_S12 = []

for i in tqdm(range(N_sample)):
    lst_S11_tmp = []
    lst_S12_tmp = []
    for f_GHz in freq_GHz:
        # Plane wave model
        S11, S12, _, _ = SimPlaneWaveList(f_GHz, angle_deg_samples[i], eps_mut, thickness_mut_samples[i], eps_mid, thickness_mid_samples[i])
        lst_S11_tmp.append(S11)
        lst_S12_tmp.append(S12)
    lst_S11.append(np.array(lst_S11_tmp))
    lst_S12.append(np.array(lst_S12_tmp))

lst_S11 = np.array(lst_S11)
lst_S12 = np.array(lst_S12)
t_MC = time.perf_counter()-t0
print('Time MC : ',t_MC)
print('Gain in time : ', t_MC/t_PCE)

m_S11 = np.sum(lst_S11,axis=0)/N_sample
m_S12 = np.sum(lst_S12,axis=0)/N_sample
std_S11 = (np.sum(lst_S11**2-m_S11**2,axis=0))/(N_sample-1)
std_S12 = (np.sum(lst_S12**2-m_S12**2,axis=0))/(N_sample-1)

############################### DETERMINIST CASE #############################################
dS11 = []
dS12 = []
for f_GHz in freq_GHz:
    S11, S12, _, _ = SimPlaneWaveList(f_GHz, m2, eps_mut, m3, eps_mid, m1)
    dS11.append(S11)
    dS12.append(S12)

######################## PLOT THE RESULTS #####################################

import matplotlib
matplotlib.use("macOSX")
import matplotlib.pyplot as plt

plt.figure()
plt.plot(freq_GHz,np.abs(S11_m),'-',color='red',label='PCE')
plt.plot(freq_GHz,np.abs(m_S11),'--',color='green',label='MC')
plt.plot(freq_GHz,np.abs(dS11),'--',color='black',label='Det')

plt.legend()
plt.figure()
plt.plot(freq_GHz,np.abs(S12_m),'-',color='red',label='PW - PCE')
plt.plot(freq_GHz,np.abs(m_S12),'--',color='green',label='MC')
plt.plot(freq_GHz,np.abs(dS12),'--',color='black',label='Det')
plt.ylabel(r'$|S_{12}|$',fontsize=16)
plt.xlabel('f (GHz)',fontsize=16)
plt.legend()
plt.figure()
plt.plot(freq_GHz, np.angle(S11_m),'-',color='red', label='PCE')
plt.plot(freq_GHz,np.angle(m_S11),'--',color='green',label='MC')
plt.plot(freq_GHz,np.angle(dS11),'--',color='black',label='Det')
plt.ylabel(r'phase $S_{11}$', fontsize=16)
plt.xlabel('f (GHz)', fontsize=16)
plt.legend()
plt.figure()
plt.plot(freq_GHz, np.angle(S12_m),'-',color='red', label='PCE')
plt.plot(freq_GHz,np.angle(m_S12),'--',color='green',label='MC')
plt.plot(freq_GHz,np.angle(dS12),'--',color='black',label='Det')
plt.ylabel(r'phase $S_{12}$', fontsize=16)
plt.xlabel('f (GHz)', fontsize=16)
plt.legend()
plt.show()

print('err S11 l2 : ', np.linalg.norm(S11_m - m_S11) / np.linalg.norm(m_S11))
print('err S12 l2 : ', np.linalg.norm(S12_m - m_S12) / np.linalg.norm(m_S12))
print(len(std_S11),len(S11_std))
print('err S11 l2 [m2] : ', np.linalg.norm(std_S11 - S11_std))
print('err S12 l2 [m2] : ', np.linalg.norm(std_S12 - S12_std))