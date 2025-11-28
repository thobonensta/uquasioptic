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
from utils.PlaneWave import SimPlaneWave
import numpy as np
from utils.GBQuasiOptic import SimGB

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ##### Frequency span for the simulation #####
    freq_GHz = np.linspace(220, 330, 1001, endpoint=True)
    ##### Parameters for the three layers MUT (mat/mid/mar) - mut = material under test
    eps_mut = 2. - 0.025j
    thickness_mut = 0.00582
    eps_mid = 21. - 0j
    thickness_mid = 0.6e-10
    dict_mut= [ # samples
        {'epsilon_r': eps_mut, 'thickness': thickness_mut},
        {'epsilon_r': eps_mid, 'thickness': thickness_mid},
        {'epsilon_r': eps_mut, 'thickness': thickness_mut}
]
    # In the case of only one material its eps_mut and x2 for the thickness
    ##### angle in deg #####
    angle_deg = 0

    ##### SIMULATION #####
    lst_S11 = []
    lst_S12 = []
    lst_S11_GB = []
    lst_S12_GB = []
    for f_GHz in freq_GHz:
        # Plane wave model
        S11, S12, _,_ = SimPlaneWave(f_GHz, angle_deg,dict_mut)
        lst_S11.append(S11)
        lst_S12.append(S12)
        # GB model
        S11, S12 = SimGB(f_GHz,angle_deg,eps_mut,thickness_mut*2,0.000250,0.0023,0.0023)
        #S11,S12, S21,S22 = SimGB_multi(f_GHz,angle_deg,dict_mut,0.000250,0.0023,0.0023)
        lst_S11_GB.append(S11)
        lst_S12_GB.append(S12)


    plt.figure()
    plt.plot(freq_GHz,np.abs(lst_S11),label='PW')
    plt.plot(freq_GHz, np.abs(lst_S11_GB),label='GB')
    plt.ylabel(r'$|S_{11}|$',fontsize=16)
    plt.xlabel('f (GHz)',fontsize=16)
    plt.legend()
    plt.figure()
    plt.plot(freq_GHz,np.abs(lst_S12),label='PW')
    plt.plot(freq_GHz, np.abs(lst_S12_GB),label='GB')
    plt.ylabel(r'$|S_{12}|$',fontsize=16)
    plt.xlabel('f (GHz)',fontsize=16)
    plt.legend()
    plt.figure()
    plt.plot(freq_GHz, np.angle(lst_S11), label='PW')
    plt.plot(freq_GHz, np.angle(lst_S11_GB), label='GB')
    plt.ylabel(r'phase $S_{11}$', fontsize=16)
    plt.xlabel('f (GHz)', fontsize=16)
    plt.legend()
    plt.figure()
    plt.plot(freq_GHz, np.angle(lst_S12), label='PW')
    plt.plot(freq_GHz, np.angle(lst_S12_GB), label='GB')
    plt.ylabel(r'phase $S_{12}$', fontsize=16)
    plt.xlabel('f (GHz)', fontsize=16)
    plt.legend()
    plt.show()