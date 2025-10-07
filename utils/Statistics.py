import scipy.stats as sts
import numpy as np
def cramerVonMises(datatest,datetrue):
    ''' Compute the Cramer's Von Mises test for a log Normal distribution '''
    return sts.cramervonmises_2samp(datatest,datetrue)

def KStest(datatest,datatrue):
    return sts.ks_2samp(datatest,datatrue,alternative='two_sided')