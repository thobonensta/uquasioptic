import time
import matplotlib
for backend in [ 'MACOSX','Qt5Agg', 'TkAgg', 'Agg']:
    try:
        matplotlib.use(backend, force=True)
        import matplotlib.pyplot as plt

        print(f"Using backend: {matplotlib.get_backend()}")
        break
    except Exception as e:
        print(f"Failed to use backend {backend}: {e}")
import matplotlib.pyplot as plt
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
freq_GHz = np.linspace(220, 330, 201, endpoint=True)
##### Parameters for the three layers MUT (mat/mid/mar) - mut = material under test
eps_mid = 1. - 0j


##################### Uncertainty over the parameters ###############################
# Model the variation for the air gap (logN since positive definite)
beta_1 = 20/100 # variation coefficient
m1 = 50e-6  # mean
sig1 = beta_1*m1 # standard-deviation
m_c1,sig_c1 = lognormal_params_from_mean_std(m1,sig1)

# Model the variation of take-off angle (N since around 0)
m2 = 0
sig2 = 0.1

# Model the variation of the MUT thickness
m3 = 0.00582
beta_3 = 0.5/100
sig3 = beta_3*m3
m_c3,sig_c3 = lognormal_params_from_mean_std(m3,sig3)


# Other
eps_mut = 2. - 0.025j
eps_mid = 1. - 0j
thickness_mut = m3


# Use chaospy for the decomposition
t0 = time.perf_counter()
X1 = cp.LogNormal(m_c1,sig_c1)
X2 = cp.Normal(m2,sig2)
X3 = cp.LogNormal(m_c3,sig_c3)

X = cp.J(X1,X2,X3)
# Decomposition using Wiener-Askey
p = 4
NbRV = 3
P = math.comb(p+NbRV,p)
decomp = cp.generate_expansion(p,X)
# Gauss quadrature adapted to our joint distribution
d = 10
tg,wg = cp.generate_quadrature(d,X,rule="gaussian")
# Extract physical values
x1_vals = tg[0, :]  # quadrature nodes for X1
x2_vals = tg[1, :]
x3_vals = tg[2, :]


wg = wg[:, None] # quadrature weigths

# Normalization
norms = np.array([cp.E(poly**2, X) for poly in decomp])

######################### PCE UQ ########################################################
# Compute the projection
from utils.PlaneWave import SimPlaneWaveList # The blackbox code we use here for test
lst_S11_PC = []
lst_S12_PC = []

for x1,x2,x3 in tqdm(zip(x1_vals,x2_vals,x3_vals)):
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
# After computing coeffsS11 and coeffsS12...

t_PCE = time.perf_counter() - t0
print('Time PCE : ',t_PCE)

# Mean of complex S-parameters (zeroth order coefficient)
S11_m = coeffsS11[0, :]
S12_m = coeffsS12[0, :]

# Compute E[|S|²] directly from coefficients
# For complex S = Σ s_i Ψ_i, we have |S|² = Σ |s_i|² (by orthogonality)
E_abs_S11_squared = np.sum(np.abs(coeffsS11)**2 * norms[:, None], axis=0)
E_abs_S12_squared = np.sum(np.abs(coeffsS12)**2 * norms[:, None], axis=0)
E_angle_S11_squared = np.sum(np.angle(coeffsS11)**2 * norms[:, None], axis=0)
E_angle_S12_squared = np.sum(np.angle(coeffsS12)**2 * norms[:, None], axis=0)
# Compute E[|S|] using quadrature
# Evaluate |S| at quadrature points and integrate
S11_at_quad = np.zeros((len(x1_vals), len(freq_GHz)), dtype=complex)
S12_at_quad = np.zeros((len(x1_vals), len(freq_GHz)), dtype=complex)

for i, phi in enumerate(decomp):
    phi_vals = phi(*tg)[:, None]  # shape (n_quad_points, 1)
    S11_at_quad += coeffsS11[i, :] * phi_vals
    S12_at_quad += coeffsS12[i, :] * phi_vals



# Compute E[|S|] using quadrature
E_abs_S11 = np.sum(wg * np.abs(S11_at_quad), axis=0)
E_abs_S12 = np.sum(wg * np.abs(S12_at_quad), axis=0)
E_angle_S11 = np.sum(wg * np.angle(S11_at_quad), axis=0)
E_angle_S12 = np.sum(wg * np.angle(S12_at_quad), axis=0)

# Variance of magnitude: Var(|S|) = E[|S|²] - E[|S|]²
S11_std = E_abs_S11_squared - E_abs_S11**2
S12_std = E_abs_S12_squared - E_abs_S12**2
S11_ang_std = E_angle_S11_squared - E_angle_S11**2
S12_ang_std = E_angle_S12_squared - E_angle_S12**2

# Mean of magnitude (for plotting)
S11_m_mag = E_abs_S11
S12_m_mag = E_abs_S12

######################## MONTE-CARLO ##########################################
t0 = time.perf_counter()
N_sample = 100000
thickness_mid_samples = np.random.lognormal(m_c1, sig_c1, int(N_sample))
angle_deg_samples = np.random.normal(m2, sig2, int(N_sample))
thickness_mut_samples = np.random.lognormal(m_c3, sig_c3, int(N_sample))

lst_S11 = []
lst_S12 = []

for i in tqdm(range(N_sample)):
    lst_S11_tmp = []
    lst_S12_tmp = []
    for f_GHz in freq_GHz:
        S11, S12, _, _ = SimPlaneWaveList(f_GHz, angle_deg_samples[i], eps_mut, thickness_mut_samples[i],
                                          eps_mid, thickness_mid_samples[i])
        lst_S11_tmp.append(S11)
        lst_S12_tmp.append(S12)
    lst_S11.append(np.array(lst_S11_tmp))
    lst_S12.append(np.array(lst_S12_tmp))

lst_S11 = np.array(lst_S11)
lst_S12 = np.array(lst_S12)
t_MC = time.perf_counter() - t0
print('Time MC : ', t_MC)
print('Gain in time : ', t_MC / t_PCE)

m_S11 = np.mean(np.abs(lst_S11), axis=0)
m_S12 = np.mean(np.abs(lst_S12), axis=0)
std_S11 = np.var(np.abs(lst_S11), axis=0, ddof=1)
std_S12 = np.var(np.abs(lst_S12), axis=0, ddof=1)

############################### DETERMINIST CASE #############################################
dS11 = []
dS12 = []
for f_GHz in freq_GHz:
    S11, S12, _, _ = SimPlaneWaveList(f_GHz, m2, eps_mut, thickness_mut, eps_mid, m1)
    dS11.append(S11)
    dS12.append(S12)

######################## PLOT THE RESULTS #####################################
plt.figure()
plt.plot(freq_GHz, S11_m_mag, '-', color='red', label='PCE')
plt.plot(freq_GHz, m_S11, '--', color='green', label='MC')
plt.plot(freq_GHz, np.abs(dS11), '--', color='black', label='Det')
plt.fill_between(freq_GHz,
                 S11_m_mag - 1.64 * np.sqrt(S11_std),
                 S11_m_mag + 1.64 * np.sqrt(S11_std),
                 color='red', alpha=0.3)
plt.ylabel(r'$|S_{11}|$', fontsize=16)
plt.xlabel('f (GHz)', fontsize=16)
plt.legend()

plt.figure()
plt.plot(freq_GHz, S12_m_mag, '-', color='red', label='PW - PCE')
plt.plot(freq_GHz, m_S12, '--', color='green', label='MC')
plt.plot(freq_GHz, np.abs(dS12), '--', color='black', label='Det')
plt.fill_between(freq_GHz,
                 S12_m_mag - 1.64 * np.sqrt(np.abs(S12_std)),
                 S12_m_mag + 1.64 * np.sqrt(np.abs(S12_std)),
                 color='red', alpha=0.3)
plt.ylabel(r'$|S_{12}|$', fontsize=16)
plt.xlabel('f (GHz)', fontsize=16)
plt.legend()

plt.figure()
plt.plot(freq_GHz, np.angle(S11_m), '-', color='red', label='PCE')
# plt.fill_between(freq_GHz,
#                  E_angle_S11 - 1.64 * np.sqrt(S11_ang_std),
#                  E_angle_S11 + 1.64 * np.sqrt(S11_ang_std),
#                  color='red', alpha=0.3)
plt.plot(freq_GHz, np.angle(np.mean(lst_S11, axis=0)), '--', color='green', label='MC')
plt.plot(freq_GHz, np.angle(dS11), '--', color='black', label='Det')
plt.ylabel(r'phase $S_{11}$', fontsize=16)
plt.xlabel('f (GHz)', fontsize=16)
plt.legend()

plt.figure()
plt.plot(freq_GHz, np.angle(S12_m), '-', color='red', label='PCE')
plt.plot(freq_GHz, np.angle(np.mean(lst_S12, axis=0)), '--', color='green', label='MC')
# plt.fill_between(freq_GHz,
#                  E_angle_S12 - 1.64 * np.sqrt(S12_ang_std),
#                  E_angle_S12 + 1.64 * np.sqrt(S12_ang_std),
#                  color='red', alpha=0.3)
plt.plot(freq_GHz, np.angle(dS12), '--', color='black', label='Det')
plt.ylabel(r'phase $S_{12}$', fontsize=16)
plt.xlabel('f (GHz)', fontsize=16)
plt.legend()
plt.show()

print('err S11 mean l2 : ', np.linalg.norm(S11_m_mag - m_S11) / np.linalg.norm(m_S11))
print('err S12 mean l2 : ', np.linalg.norm(S12_m_mag - m_S12) / np.linalg.norm(m_S12))
print('err S11 var l2 : ', np.linalg.norm(std_S11 - S11_std) / np.linalg.norm(std_S11))
print('err S12 var l2 : ', np.linalg.norm(std_S12 - S12_std) / np.linalg.norm(std_S12))
print('Variance S11 [0] - MC vs PCE:', std_S11[0], S11_std[0])
print('Variance S12 [0] - MC vs PCE:', std_S12[0], S12_std[0])