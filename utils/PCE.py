import numpy as np
from math import factorial
from scipy.special import eval_hermitenorm
from random import gauss
def nthMomentGauss(n):
    ''' function that computes the n-th order moment of a normal distribution
    Inputs : n - int : order
    output : float : moment
    '''

    if n % 2 == 1:
        return 0
    else:
        return factorial(n) / (2 ** (int(n / 2))) / factorial(int(n / 2))


def probabilistic_hermite_coeff_numpy(n):
    ''' function that computes the coefficient of the Hermite polynomial of order n'''
    coeff = np.zeros(n + 1)
    for k in range(int(n / 2) + 1):
        coeff[n - 2 * k] = (-1) ** k * factorial(n) / (2 ** k * factorial(k) * factorial(n - 2 * k))
    return np.array(coeff)


def computeNormalization(p):
    ''' function that computes the normalization factor for each element of the Hermite polynomial
    basis'''
    den_list = [1]
    for i in range(1, p):
        prod = 1
        h = probabilistic_hermite_coeff_numpy(i)
        h_1 = np.poly1d(np.flip(h))
        h_2 = np.polymul(h_1, h_1)
        n2 = len(h_2)
        S = sum([h_2[k] * nthMomentGauss(k) for k in range(n2 + 1)])
        prod *= S
        den_list.append(prod)
    return den_list

def computeNormalization2(alpha):
    den_list = [1]
    for i in range(1,len(alpha)):
        alpha_i = alpha[i]
        prod = 1
        for deg in alpha_i:
            h = probabilistic_hermite_coeff_numpy(deg)
            h_1 = np.poly1d(np.flip(h))
            h_2 = np.polymul(h_1, h_1)
            n2 = len(h_2)
            S = sum([h_2[k] * nthMomentGauss(k) for k in range(n2 + 1)])
            prod *= S
        den_list.append(prod)
    return den_list

def decompPCELogNormGH(p,m_c1,sig_c1,ti):

    lst_decomp = []
    for t1 in ti:
        tmp = 0
        for i in range(p):
            a = sig_c1 ** i / factorial(i) * np.exp(m_c1 + sig_c1 ** 2 / 2)
            tmp += a*eval_hermitenorm(i,t1)
        lst_decomp.append([tmp])

    return lst_decomp

def decompPCENormGH(p,m_c1,sig_c1,ti):

    lst_decomp = []
    for t1 in ti:
        tmp = 0
        for i in range(p):
            if i == 0:
                a = m_c1
            elif i == 1:
                a = sig_c1**2
            else:
                a=0
            tmp += a*eval_hermitenorm(i,t1)
        lst_decomp.append([tmp])

    return lst_decomp
def decompPCELogNormGH2(P,alpha,m_c1,sig_c1,m_c2,sig_c2,ti):
    X = []
    for t1 in ti:
        for t2 in ti:
            X1 = 0
            X2 = 0
            for i in range(P):
                alpha1, alpha2 = alpha[i]
                if alpha2 == 0:
                    a = sig_c1 ** i / factorial(i) * np.exp(m_c1 + sig_c1 ** 2 / 2)
                    X1 += a * eval_hermitenorm(alpha1, t1) *  eval_hermitenorm(alpha2, t2)
                if alpha1 == 0:
                    a = sig_c2 ** i / factorial(i) * np.exp(m_c2 + sig_c2 ** 2 / 2)
                    X2 += a * eval_hermitenorm(alpha1, t1) *  eval_hermitenorm(alpha2, t2)
            X.append([X1,X2])
    return X

def decompPCENormandLogNormGH2(P,alpha,m_c1,sig_c1,m_c2,sig_c2,ti):
    X = []
    for t1 in ti:
        for t2 in ti:
            X1 = 0
            X2 = 0
            for i in range(P):
                alpha1, alpha2 = alpha[i]
                if alpha1 == 0:
                    if alpha2 == 0:
                        a = m_c2
                    elif alpha2 == 1:
                        a = sig_c2 ** 2
                    else:
                        a = 0
                    X2 += a * eval_hermitenorm(alpha1, t1) *  eval_hermitenorm(alpha2, t2)
                if alpha2 == 0:
                    a = sig_c2 ** i / factorial(i) * np.exp(m_c1 + sig_c1 ** 2 / 2)
                    X1 += a * eval_hermitenorm(alpha1, t1) *  eval_hermitenorm(alpha2, t2)
            X.append([X1,X2])
    return X

def decompPCEHermite(coeffs,N_s):
    res = []
    for i in range(N_s):
        sample = gauss(0,1)
        u_s = 0
        for j in range(len(coeffs)):
            u_s += coeffs[j]*eval_hermitenorm(j,sample)
        res.append(u_s)
    return res

def decompPCEHermite2(coeffs,alpha,N_s):
    res = []
    for i in range(N_s):
        sample1 = gauss(0,1)
        sample2 = gauss(0,1)
        u_s = 0
        for j in range(len(coeffs)):
            alpha1, alpha2 = alpha[j]
            u_s += coeffs[j]*eval_hermitenorm(alpha1,sample1)*eval_hermitenorm(alpha2,sample2)
        res.append(u_s)
    return res

def coeffsPCELogN(p,m_c1,sig_c1):
    lst_coeffs = []
    for i in range(p):
        a = sig_c1 ** i / factorial(i) * np.exp(m_c1 + sig_c1 ** 2 / 2)
        lst_coeffs.append(a)
    return lst_coeffs

def coeffsPCEN(p,m1,sig1):
    lst_coeffs = []
    for i in range(p):
        if i == 0:
            a = m1
        elif i == 1:
            a = sig1**2
        else:
            a=0
        lst_coeffs.append(a)
    return lst_coeffs

def alpha2(p):
    alpha_list = []
    for pp in range(p+1):
        for m in range(pp+1):
            alpha_list.append([pp-m,m])
    return alpha_list


def decompPCEquadoneRV(p,d,ti,wi,X,freq_GHz, angle_deg_min,eps_mut,thickness_mut,eps_mid,fun):

    lst_S11_PC = []
    lst_S12_PC = []
    for i in range(d):
        tmp = X[i]
        lst_S11 = []
        lst_S12 = []
        for f_GHz in freq_GHz:
            # Plane wave model
            S11, S12, _,_ = fun(f_GHz, angle_deg_min,eps_mut,thickness_mut,eps_mid,tmp[0])
            lst_S11.append(S11)
            lst_S12.append(S12)
        lst_S11_PC.append(lst_S11)
        lst_S12_PC.append(lst_S12)


    den_list = computeNormalization(p+1)

    dec_S11_PC = np.zeros((len(lst_S11_PC[0]), p+1),dtype='complex')
    dec_S12_PC = np.zeros((len(lst_S12_PC[0]), p+1),dtype='complex')

    for i in range(p+1):
        S_11_tmp = np.zeros(len(lst_S11_PC[0]),dtype='complex')
        S_12_tmp = np.zeros(len(lst_S12_PC[0]),dtype='complex')
        for k1 in range(d):
            t1 = ti[k1]
            w1 = wi[k1]

            S_11_tmp = S_11_tmp + w1  * np.array(lst_S11_PC)[k1, :] * eval_hermitenorm(i,t1) * 1 / (
                            np.sqrt(2 * np.pi))
            S_12_tmp = S_12_tmp + w1  * np.array(lst_S12_PC)[k1, :] * eval_hermitenorm(i,t1)* 1 / (
                            np.sqrt(2 * np.pi))
        dec_S11_PC[:, i] = S_11_tmp / den_list[i]
        dec_S12_PC[:, i] = S_12_tmp / den_list[i]

    return dec_S11_PC, dec_S12_PC




