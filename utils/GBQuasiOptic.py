import numpy as np
from .FresnelCoeff import compute_FresnelCoeff
from scipy.constants import c
from .Convert import TtoS, StoT

def SimGB(freq_GHz,slab_angle_deg,slab_epsr,slab_L,delta_z0,w0h,w0r,d=15,polar='te',Rref=-1):
    # Extract simulation parameters from configuration
    slab_angle_rad = slab_angle_deg * np.pi / 180.  # Convert to radians for calculations
    freq_Hz = freq_GHz * 1e9  # Convert frequency to Hz
    k0 = 2 * np.pi * freq_Hz / c  # Free space wave number (rad/m)
    lamb = c/freq_Hz

    # Initialize medium properties for air (before first interface)
    n1 = eta1 = 1.  # Air: refractive index = 1, impedance = 1

    # Calculate incident angle parameters
    cos_theta1 = np.cos(slab_angle_rad)  # Cosine of incident angle in air

    # Calculate material properties for current layer
    n2 = np.sqrt(slab_epsr)  # Complex refractive index: n = sqrt(εr)
    eta2 = 1 / n2  # Wave impedance in medium: η = 1/n (normalized to free space)

    # Apply Snell's law to find transmission angle in current layer
    cos_theta2 = np.sqrt(1 - (n1 / n2) ** 2 * np.sin(slab_angle_rad) ** 2)

    # Calculate Fresnel reflection coefficient based on polarization
    rho_1 = compute_FresnelCoeff(eta1, eta2, cos_theta1, cos_theta2, polar)
    rho_2 = -rho_1
    # Calculate Fresnel transmission coefficient based on polarizatio
    tau_1 = 1 + rho_1
    tau_2 = 1 - rho_1

    # Compute the different parameters of GB
    Pair = np.exp(-1j*k0*slab_L)
    P0 = np.exp(-1j*k0*delta_z0)
    cax_z0 = 1/(w0h/w0r + w0r/w0h -1j*lamb*delta_z0/(np.pi*w0h*w0r))
    cax_Ref = 2*P0/(1/cax_z0)

    cMUT_T = 0
    cMUT_R = rho_1*cax_Ref
    Pm = np.exp(-1j*k0*n2*slab_L)
    for i in range(d):
        delta_zs_p = (1-1/np.real(n2))*slab_L - i * 2*slab_L/(np.real(n2))
        delta_zs = delta_z0-delta_zs_p
        cax_zs = 1/(w0h/w0r + w0r/w0h -1j*lamb*delta_zs/(np.pi*w0h*w0r))

        if slab_angle_deg !=0:
            deltax = slab_L*(1-cos_theta1**2)*(np.sqrt(n2**2-(1-cos_theta1**2))-cos_theta1)/(cos_theta1*np.sqrt(n2**2-(1-cos_theta1**2)))
            Pdp = np.exp(-1j*k0*(n2*slab_L/cos_theta2+deltax))
            Pm = np.exp(-1j * k0 * n2 * slab_L*cos_theta2)

        cMUT_T += 2 * tau_1 * tau_2 * Pm * P0 * Pm ** (2 * i) * rho_2 ** (2 * i) / (1 / cax_z0)
    for i in range(1,d):
        delta_zs_p = (1-1/np.real(n2))*slab_L - i * 2*slab_L/(np.real(n2))
        delta_zs = delta_z0-delta_zs_p
        cax_zs = 1/(w0h/w0r + w0r/w0h -1j*lamb*delta_zs/(np.pi*w0h*w0r))
        cMUT_R += 2 * tau_1 * tau_2 * P0 * Pm ** (2 * i) * rho_2 ** (2 * i-1) / (1 / cax_z0)

    S12 = cMUT_T/(cax_Ref*Pair)
    S11 = cMUT_R/(cax_Ref)

    return S11, S12

def SimGB_multi(freq_GHz,slab_angle_deg,dict_mut,delta_z0,w0h,w0r,d=10,polar='te',Rref=-1):
    # Extract simulation parameters from configuration
    slab_angle_rad = slab_angle_deg * np.pi / 180.  # Convert to radians for calculations
    freq_Hz = freq_GHz * 1e9  # Convert frequency to Hz
    k0 = 2 * np.pi * freq_Hz / c  # Free space wave number (rad/m)
    lamb = c/freq_Hz

    # Initialize medium properties for air (before first interface)
    n1 = eta1 = 1.  # Air: refractive index = 1, impedance = 1
    T = None  # Initialize transfer matrix accumulator

    # Calculate incident angle parameters
    cos_theta1 = np.cos(slab_angle_rad)  # Cosine of incident angle in air
    thickness_total = 0

    for slab in dict_mut:
        # Update total thickness for phase reference calculations
        thickness_total += slab['thickness']
        slab_thickness = slab['thickness']  # Current layer thickness

        # Calculate material properties for current layer
        n2 = np.sqrt(slab['epsilon_r'] + 0j)  # Complex refractive index: n = sqrt(εr)
        eta2 = 1 / n2  # Wave impedance in medium: η = 1/n (normalized to free space)

        # Apply Snell's law to find transmission angle in current layer
        cos_theta2 = np.sqrt(1 - (n1 / n2) ** 2 * np.sin(slab_angle_rad) ** 2)

        # Calculate Fresnel reflection coefficient based on polarization
        rho_1 = compute_FresnelCoeff(eta1, eta2, cos_theta1, cos_theta2, polar)
        rho_2 = -rho_1
        # Calculate Fresnel transmission coefficient based on polarizatio
        tau_1 = 1 + rho_1
        tau_2 = 1 - rho_1

        # Compute the different parameters of GB
        Pair = np.exp(-1j * k0 * slab_thickness)
        P0 = np.exp(-1j * k0 * delta_z0)
        cax_z0 = 1 / (w0h / w0r + w0r / w0h - 1j * lamb * delta_z0 / (np.pi * w0h * w0r))
        cax_Ref = 2 * P0 / (1 / cax_z0)

        cMUT_T = 0
        cMUT_R = rho_1 * cax_Ref
        Pm = np.exp(-1j * k0 * n2 * slab_thickness)
        for i in range(d):
            delta_zs_p = (1 - 1 / np.real(n2)) * slab_thickness - i * 2 * slab_thickness / (np.real(n2))
            delta_zs = delta_z0 - delta_zs_p
            cax_zs = 1 / (w0h / w0r + w0r / w0h - 1j * lamb * delta_zs / (np.pi * w0h * w0r))
            cMUT_T += 2 * tau_1 * tau_2 * Pm * P0 * Pm ** (2 * i) * rho_2 ** (2 * i) / (1 / cax_z0)
        for i in range(1, d):
            # delta_zs_p = (1 - 1 / np.real(n2)) * slab_thickness - i * 2 * slab_thickness / (np.real(n2))
            # delta_zs = delta_z0 - delta_zs_p
            # cax_zs = 1 / (w0h / w0r + w0r / w0h - 1j * lamb * delta_zs / (np.pi * w0h * w0r))
            cMUT_R += 2 * tau_1 * tau_2 * P0 * Pm ** (2 * i) * rho_2 ** (2 * i - 1) / (1 / cax_z0)

        S21 = cMUT_T / (cax_Ref * Pair)
        S11 = cMUT_R / (cax_Ref)

        # Convert layer S-parameters to T-parameters for cascading
        T11, T21, T12, T22 = StoT(S11, S21, S21, S11)

        # Cascade transfer matrices (matrix multiplication for series connection)
        if T is None:
            # First layer: initialize total transfer matrix
            T = np.array([[T11, T12], [T21, T22]])
        else:
            # Subsequent layers: multiply transfer matrices (order matters!)
            T = T @ np.array([[T11, T12], [T21, T22]])

    # Convert final cascaded T-matrix back to S-parameters
    S11, S12, S21, S22 = TtoS(T)

    return S11, S12, S21, S22







