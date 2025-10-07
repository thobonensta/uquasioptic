import numpy as np
from scipy.constants import c
from .Convert import StoT, TtoS
from .FresnelCoeff import compute_FresnelCoeff

def SimPlaneWave(freq_GHz,slab_angle_deg,dict_mut, polar='te'):
    """
    Simulate S-parameters using plane wave theory for multilayer structures.

    Calculates electromagnetic scattering parameters by modeling plane wave
    propagation through stratified media using Fresnel coefficients and
    transfer matrix method. Handles oblique incidence and multiple layers.

    Parameters
    ----------
    frequency_GHz : float
        Operating frequency in GHz for the simulation.

    Returns
    -------
    tuple of complex
        Four-element tuple containing (S11, S12, S21, S22) S-parameters
        calculated from plane wave theory.

    Notes
    -----
    The simulation process:
    1. Calculate wave parameters (wavelength, impedances, angles)
    2. For each layer: compute Fresnel coefficients and phase delays
    3. Build transfer matrix for each interface
    4. Cascade all transfer matrices
    5. Convert final T-matrix to S-parameters
    6. Apply phase corrections for reference plane positioning
    """

    # Extract simulation parameters from configuration
    slab_angle_rad = slab_angle_deg * np.pi / 180.  # Convert to radians for calculations
    freq_Hz = freq_GHz * 1e9  # Convert frequency to Hz
    k0 = 2 * np.pi * freq_Hz / c  # Free space wave number (rad/m)

    # Initialize medium properties for air (before first interface)
    n1 = eta1 = 1.  # Air: refractive index = 1, impedance = 1
    T = None  # Initialize transfer matrix accumulator

    # Calculate incident angle parameters
    cos_theta1 = np.cos(slab_angle_rad)  # Cosine of incident angle in air
    thickness_total = 0  # Accumulator for total structure thickness

    # Process each material layer in the multilayer structure
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
        rho = compute_FresnelCoeff(eta1,eta2,cos_theta1,cos_theta2,polar)

        # Calculate phase delay through current layer
        P = np.exp(-1j * k0 * n2 * slab_thickness * cos_theta2)  # Propagation phase factor

        # Calculate single-layer S-parameters using transmission line theory
        den = (1 - (P * rho) ** 2)  # Common denominator (1 - Γ²P²)
        S11 = ((1 - P ** 2) * rho) / den  # Layer reflection coefficient
        S21 = P * (1 - rho ** 2) / den  # Layer transmission coefficient

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

    # Apply phase corrections for reference plane positioning
    F = np.exp(1j * k0 * thickness_total)  # Phase factor for total thickness
    S12 *= F
    S21 *= F  # Correct transmission phase
    S22 = S11 * F ** 2  # Correct output reflection phase (double path)

    return S11, S12, S21, S22

def SimPlaneWaveList(freq_GHz,slab_angle_deg,eps_mut,thickness_mut,eps_mid,thickness_mid, polar='te'):
    """
    Simulate S-parameters using plane wave theory for multilayer structures.

    Calculates electromagnetic scattering parameters by modeling plane wave
    propagation through stratified media using Fresnel coefficients and
    transfer matrix method. Handles oblique incidence and multiple layers.

    Parameters
    ----------
    frequency_GHz : float
        Operating frequency in GHz for the simulation.

    Returns
    -------
    tuple of complex
        Four-element tuple containing (S11, S12, S21, S22) S-parameters
        calculated from plane wave theory.

    Notes
    -----
    The simulation process:
    1. Calculate wave parameters (wavelength, impedances, angles)
    2. For each layer: compute Fresnel coefficients and phase delays
    3. Build transfer matrix for each interface
    4. Cascade all transfer matrices
    5. Convert final T-matrix to S-parameters
    6. Apply phase corrections for reference plane positioning
    """

    # Extract simulation parameters from configuration
    slab_angle_rad = slab_angle_deg * np.pi / 180.  # Convert to radians for calculations
    freq_Hz = freq_GHz * 1e9  # Convert frequency to Hz
    k0 = 2 * np.pi * freq_Hz / c  # Free space wave number (rad/m)

    # Initialize medium properties for air (before first interface)
    n1 = eta1 = 1.  # Air: refractive index = 1, impedance = 1
    T = None  # Initialize transfer matrix accumulator

    # Calculate incident angle parameters
    cos_theta1 = np.cos(slab_angle_rad)  # Cosine of incident angle in air
    thickness_total = 0  # Accumulator for total structure thickness

    # Create the dictionary from the given parameters
    dict_mut = [  # samples
        {'epsilon_r': eps_mut, 'thickness': thickness_mut},
        {'epsilon_r': eps_mid, 'thickness': thickness_mid},
        {'epsilon_r': eps_mut, 'thickness': thickness_mut}
    ]

    # Process each material layer in the multilayer structure
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
        rho = compute_FresnelCoeff(eta1,eta2,cos_theta1,cos_theta2,polar)

        # Calculate phase delay through current layer
        P = np.exp(-1j * k0 * n2 * slab_thickness * cos_theta2)  # Propagation phase factor

        # Calculate single-layer S-parameters using transmission line theory
        den = (1 - (P * rho) ** 2)  # Common denominator (1 - Γ²P²)
        S11 = ((1 - P ** 2) * rho) / den  # Layer reflection coefficient
        S21 = P * (1 - rho ** 2) / den  # Layer transmission coefficient

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

    # Apply phase corrections for reference plane positioning
    F = np.exp(1j * k0 * thickness_total)  # Phase factor for total thickness
    S12 *= F
    S21 *= F  # Correct transmission phase
    S22 = S11 * F ** 2  # Correct output reflection phase (double path)

    return S11, S12, S21, S22