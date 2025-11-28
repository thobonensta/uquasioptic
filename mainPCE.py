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
from utils.PlaneWave import SimPlaneWaveList
import numpy as np
import scipy.special as sps
from utils.Statistics import cramerVonMises, KStest
from utils.PCE import computeNormalization, computeNormalization2
from utils.PCE import decompPCELogNormGH,coeffsPCELogN,decompPCEHermite, decompPCEquadoneRV
from utils.PCE import decompPCELogNormGH2, alpha2,decompPCEHermite2, decompPCENormandLogNormGH2
from scipy.special import eval_hermitenorm
import math



if __name__ == '__main__':
    #####-----------------------------------------------------------------------------#####
    #####-----------------------------PARAMETERS--------------------------------------#####
    #####-----------------------------------------------------------------------------#####

    ##### Frequency span for the simulation #####
    freq_GHz = np.linspace(220, 250, 21, endpoint=True)
    ##### Parameters for the three layers MUT (mat/mid/mar) - mut = material under test
    eps_mut = 2. - 0.025j
    thickness_mut = 0.00582
    eps_mid = 1. - 0j
    thickness_mid_min = 99e-6
    thickness_mid_max = 101e-6

    ##### angle in deg #####
    angle_deg_min = 0
    angle_deg_max = 2


    #####-----------------------------------------------------------------------------#####
    #####-----------------------------RANDOM VAR--------------------------------------#####
    #####-----------------------------------------------------------------------------#####
    law = 'lognorm'
    # Compute the PC decomposition for logN law of thickness mid
    beta_1 = 10/100 # variation coefficient
    m1 = (thickness_mid_min+thickness_mid_max)/2  # mean
    sig1 = beta_1*m1 # standard-deviation
    m_c1 = np.log(m1) - 1 / 2 * np.log(1 + sig1 ** 2 / m1 ** 2)
    sig_c1 = np.sqrt(np.log(1 + sig1 ** 2 / m1 ** 2))
    # Compute the PC decomposition for logN law of theta
    if law == 'loglog':
        beta_2 = 10/100 # variation coefficient
        m2 = (angle_deg_min+angle_deg_max)/2  # mean
        sig2 = beta_2*m2 # standard-deviation
        m_c2 = np.log(m2) - 1 / 2 * np.log(1 + sig2 ** 2 / m2 ** 2)
        sig_c2 = np.sqrt(np.log(1 + sig2 ** 2 / m2 ** 2))
    # Compute the PC decomposition for logN law of theta
    elif law == 'lognorm':
        m2 = 0  # mean
        sig2 = 0.1 # standard-deviation


    #####-----------------------------------------------------------------------------#####
    #####-----------------------------PCE PW -----------------------------------------#####
    #####-----------------------------------------------------------------------------#####

    t0 = time.perf_counter()
    NbRV = 2 # number of random variable
    p = 6 # decomposition order for one RV
    P = math.comb(p+NbRV,p)
    d = 10 # quadrature order
    ti, wi = sps.roots_hermitenorm(d) # Hermite roots and weigths for the quadrature
    print('point de Gauss', ti)
    print('poids de Gauss', wi)

    if NbRV == 1:
        alpha = range(p+1) # list of Hermite degree up to order P
    elif NbRV == 2:
        alpha = alpha2(p)
    else:
        print('Not coded yet for more RVs')

    if NbRV == 1:
        X = decompPCELogNormGH(p+1,m_c1,sig_c1,ti) # evalutation of Hermite poly at gauss points
    elif NbRV == 2:
        if law == 'loglog':
            X = decompPCELogNormGH2(P,alpha,m_c1,sig_c1,m_c2,sig_c2,ti)
        elif law=='lognorm':
            X = decompPCENormandLogNormGH2(P,alpha,m_c1,sig_c1,m2,sig2,ti)

    # ##### SIMULATION  for one RV #####
    if NbRV == 1:
        dec_S11_PC = decompPCEquadoneRV(p, d, ti, wi, X, freq_GHz, angle_deg_min, eps_mut, thickness_mut, eps_mid, SimPlaneWaveList)
        dec_S12_PC = decompPCEquadoneRV(p, d, ti, wi, X, freq_GHz, angle_deg_min, eps_mut, thickness_mut, eps_mid, SimPlaneWaveList)

        S11_m = dec_S11_PC[:, 0]
        S12_m = dec_S12_PC[:, 0]

        S11_std = np.sum(dec_S11_PC[:, 1:]**2,axis=-1)
        S12_std = np.sum(dec_S12_PC[:, 1:]**2,axis=-1)

    ##### SIMULATION  for two RV #####
    elif NbRV == 2:
        lst_S11_PC = []
        lst_S12_PC = []
        for i in range(len(X)):
            thickness_mid = X[i][0]
            angle_deg = X[i][1]
            lst_S11 = []
            lst_S12 = []
            for f_GHz in freq_GHz:
                # Plane wave model
                S11, S12, _, _ = SimPlaneWaveList(f_GHz, angle_deg, eps_mut, thickness_mut, eps_mid, thickness_mid)
                lst_S11.append(S11)
                lst_S12.append(S12)
            lst_S11_PC.append(lst_S11)
            lst_S12_PC.append(lst_S12)

        #den_list = computeNormalization(p + 1)
        den_list = computeNormalization2(alpha)


        dec_S11_PC = np.zeros((len(lst_S11_PC[0]), P), dtype='complex')
        dec_S12_PC = np.zeros((len(lst_S12_PC[0]), P), dtype='complex')

        for i in range(P):
            S_11_tmp = np.zeros(len(lst_S11_PC[0]), dtype='complex')
            S_12_tmp = np.zeros(len(lst_S12_PC[0]), dtype='complex')
            alpha1, alpha2 = alpha[i]
            m = 0
            for k1 in range(d):
                for k2 in range(d):
                    t1 = ti[k1]
                    w1 = wi[k1]
                    t2 = ti[k2]
                    w2 = wi[k2]

                    S_11_tmp = S_11_tmp + w1 * w2* np.array(lst_S11_PC)[m, :] * eval_hermitenorm(alpha1, t1) * eval_hermitenorm(alpha2, t2)* 1 / (
                        np.sqrt(2 * np.pi)**NbRV)
                    S_12_tmp = S_12_tmp + w1 * w2 * np.array(lst_S12_PC)[m, :] * eval_hermitenorm(alpha1, t1) * eval_hermitenorm(alpha2, t2)* 1 / (
                        np.sqrt(2 * np.pi)**NbRV)
                    m+=1
            dec_S11_PC[:, i] = S_11_tmp / den_list[i]
            dec_S12_PC[:, i] = S_12_tmp / den_list[i]

        S11_m = dec_S11_PC[:, 0]
        S12_m = dec_S12_PC[:, 0]

        S11_std = np.sum(dec_S11_PC[:, 1:] ** 2, axis=-1)
        S12_std = np.sum(dec_S12_PC[:, 1:] ** 2, axis=-1)

    t_PCE = time.perf_counter() - t0
    print('Temps PCE total : ', t_PCE)

    # Monte Carlo with two RV
    N_sample = 100000
    t0 = time.perf_counter()
    if law == 'loglog':
        thickness_mid_samples = np.random.lognormal(m_c1,sig_c1,int(N_sample))
        angle_deg_samples = np.random.lognormal(m_c2,sig_c2,int(N_sample))
    elif law == 'lognorm':
        thickness_mid_samples = np.random.lognormal(m_c1,sig_c1,int(N_sample))
        angle_deg_samples = np.random.normal(m2,sig2,int(N_sample))

    lst_S11 = []
    lst_S12 = []

    lst_S11_tmp = []
    lst_S12_tmp = []
    for f_GHz in freq_GHz:
        # Plane wave model
        S11, S12, _, _ = SimPlaneWaveList(f_GHz, angle_deg_samples[0], eps_mut, thickness_mut, eps_mid, thickness_mid_samples[0])
        lst_S11_tmp.append(S11)
        lst_S12_tmp.append(S12)
    lst_S11.append(np.array(lst_S11_tmp))
    lst_S12.append(np.array(lst_S12_tmp))

    for i in range(1,int(N_sample)):
        lst_S11_tmp = []
        lst_S12_tmp = []
        for f_GHz in freq_GHz:
            # Plane wave model
            S11, S12, _, _ = SimPlaneWaveList(f_GHz, angle_deg_samples[i], eps_mut, thickness_mut, eps_mid, thickness_mid_samples[i])
            lst_S11_tmp.append(S11)
            lst_S12_tmp.append(S12)
        lst_S11.append(np.array(lst_S11_tmp))
        lst_S12.append(np.array(lst_S12_tmp))

    lst_S11 = np.array(lst_S11)
    lst_S12 = np.array(lst_S12)


    m_S11 = np.sum(lst_S11,axis=0)/N_sample
    m_S12 = np.sum(lst_S12,axis=0)/N_sample
    std_S11 = (np.sum(lst_S11**2-m_S11**2,axis=0))/(N_sample-1)
    std_S12 = (np.sum(lst_S12**2-m_S12**2,axis=0))/(N_sample-1)

    t_MC = time.perf_counter() - t0
    print('Temps Monte Carlo {0}: '.format(N_sample), t_MC)

    print('gain en temps', t_MC/t_PCE)

    lst_coeffs = coeffsPCELogN(p, m_c1, sig_c1)
    if law == 'loglog':
        lst_coeffsa = coeffsPCELogN(p, m_c2, sig_c2)
    elif law == 'lognorm':
        lst_coeffsa = coeffsPCELogN(p, m2, sig2)


    thicknessPCE_samples = decompPCEHermite(lst_coeffs,N_s=10000)
    anglePCE_samples = decompPCEHermite(lst_coeffsa,N_s=10000)

    res = cramerVonMises(thicknessPCE_samples, thickness_mid_samples)
    resKS = KStest(thicknessPCE_samples, thickness_mid_samples)
    resa = cramerVonMises(anglePCE_samples, angle_deg_samples)
    resaKS = KStest(anglePCE_samples, angle_deg_samples)
    print('p-value CVM test [thick] : ',res.pvalue )
    print('p-value KS test [thick] : ', resKS.pvalue)
    print('p-value CVM test [angle] : ',resa.pvalue )
    print('p-value KS test [angle] : ', resaKS.pvalue)


    lst_S11_tmp = []
    lst_S12_tmp = []
    for f_GHz in freq_GHz:
        # Plane wave model
        S11, S12, _, _ = SimPlaneWaveList(f_GHz, 1, eps_mut, thickness_mut, eps_mid, (thickness_mid_min+thickness_mid_max)/2)
        lst_S11_tmp.append(S11)
        lst_S12_tmp.append(S12)
    plt.figure()
    plt.plot(freq_GHz,np.abs(S11_m),'-',color='red',label='PCE')
    plt.plot(freq_GHz,np.abs(m_S11),'--',color='blue', alpha = 0.6,label='MC [{0}]'.format(N_sample))
    plt.plot(freq_GHz,np.abs(lst_S11_tmp),':',color='black', alpha = 0.6,label='Deterministe')
    plt.fill_between(freq_GHz,np.abs(S11_m-1.96*np.sqrt(S11_std)),np.abs(S11_m+1.96*np.sqrt(S11_std)),color='red',alpha=0.3)
    plt.ylabel(r'$|S_{11}|$',fontsize=16)
    plt.xlabel('f (GHz)',fontsize=16)
    plt.legend()
    plt.figure()
    plt.plot(freq_GHz,np.abs(S12_m),'-',color='red',label='PW - PCE')
    plt.plot(freq_GHz,np.abs(m_S12),'--',color='blue', alpha = 0.6,label='PW - MC [{0}]'.format(N_sample))
    plt.fill_between(freq_GHz,np.abs(S12_m-1.96*np.sqrt(S12_std)),np.abs(S12_m+1.96*np.sqrt(S12_std)),color='red',alpha=0.3)
    plt.plot(freq_GHz,np.abs(lst_S12_tmp),':',color='black', alpha = 0.6,label='Deterministe')
    plt.ylabel(r'$|S_{12}|$',fontsize=16)
    plt.xlabel('f (GHz)',fontsize=16)
    plt.legend()
    plt.figure()
    plt.plot(freq_GHz, np.angle(S11_m),'-',color='red', label='PW - PCE')
    plt.plot(freq_GHz, np.angle(m_S11),'--',color='blue', alpha = 0.6, label='PW - MC [{0}]'.format(N_sample))
    plt.plot(freq_GHz,np.angle(lst_S11_tmp),':',color='black', alpha = 0.6,label='Deterministe')
    plt.ylabel(r'phase $S_{11}$', fontsize=16)
    plt.xlabel('f (GHz)', fontsize=16)
    plt.legend()
    plt.figure()
    plt.plot(freq_GHz, np.angle(S12_m),'-',color='red', label='PW - PCE')
    plt.plot(freq_GHz, np.angle(m_S12),'--',color='blue', alpha = 0.6, label='PW - MC [{0}]'.format(N_sample))
    plt.plot(freq_GHz,np.angle(lst_S12_tmp),':',color='black', alpha = 0.6,label='Deterministe')
    plt.ylabel(r'phase $S_{12}$', fontsize=16)
    plt.xlabel('f (GHz)', fontsize=16)
    plt.legend()
    print('err S11 l2 : ', np.linalg.norm(S11_m-m_S11)/np.linalg.norm(m_S11))
    print('err S12 l2 : ', np.linalg.norm(S12_m-m_S12)/np.linalg.norm(m_S12))
    print('err S11 l2 [m2] : ', np.linalg.norm(std_S11-S11_std))
    print('err S12 l2 [m2] : ', np.linalg.norm(std_S12-S12_std))
    plt.show()