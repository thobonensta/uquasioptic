def StoT(S11, S21, S12, S22):
    """
    Convert S-parameters to T-parameters (transfer matrix).
    Parameters
    ----------
    S11 : complex
    S21 : complex
    S12 : complex
    S22 : complex

    Returns
    -------
    tuple of complex
        Four-element tuple containing (T11, T21, T12, T22) transfer parameters.
    """
    # Convert S-parameters to T-parameters using standard formulas
    T11 = 1 / S21  # Forward voltage transfer ratio
    T22 = S12 - S11 * S22 / S21  # Reverse current transfer ratio
    T21 = S11 / S21  # Reverse voltage transfer ratio
    T12 = -T21  # Forward current transfer ratio (reciprocal network)
    return T11, T21, T12, T22

def TtoS(T):
        """
        Convert T-parameters (transfer matrix) to S-parameters.

        Parameters
        ----------
        T : ndarray
            2x2 transfer matrix with elements [[T11, T12], [T21, T22]].

        Returns
        -------
        tuple of complex
            Four-element tuple containing (S11, S12, S21, S22) scattering parameters.
        """
        # Convert T-parameters back to S-parameters
        S11 = T[1, 0] / T[0, 0]  # Input reflection coefficient
        S21 = 1 / T[0, 0]  # Forward transmission coefficient
        S12 = T[1, 1] - T[0, 1] * T[1, 0] / T[0, 0]  # Reverse transmission coefficient
        S22 = -T[0, 1] / T[0, 0]  # Output reflection coefficient
        return S11, S12, S21, S22