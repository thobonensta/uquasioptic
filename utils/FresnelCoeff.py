def compute_FresnelCoeff(eta1,eta2,cos_theta1,cos_theta2,polar):
    # Calculate Fresnel reflection coefficient based on polarization
    if polar == 'te':
        # TE polarization (H-field parallel to interface, magnetic field transverse)
        den = eta2 * cos_theta1 + eta1 * cos_theta2  # Denominator for TE case
        rho = (eta2 * cos_theta1 - eta1 * cos_theta2) / den  # TE Fresnel coefficient
    elif polar == 'tm':
        # TM polarization (E-field parallel to incidence plane, electric field transverse)
        den = eta2 * cos_theta2 + eta1 * cos_theta1  # Denominator for TM case
        rho = (eta2 * cos_theta2 - eta1 * cos_theta1) / den  # TM Fresnel coefficient
    else:
        rho = 0  # Fallback (should not occur)

    return rho