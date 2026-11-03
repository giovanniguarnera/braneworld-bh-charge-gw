"""
ISCO of the remnant black hole and cutoff frequency of the waveform.

See https://arxiv.org/pdf/0801.4297

The remnant mass and spin are computed with the non-precessing fits of pySEOBNR, so this
module only works for aligned or antialigned initial spins.
"""

import numpy as np
from pyseobnr.eob.utils.nr_utils import bbh_final_spin_non_precessing_HBR2016
from pyseobnr.eob.utils.nr_utils import bbh_final_mass_non_precessing_UIB2016
from config.params import MSUN_SEC


def compute_r_isco(chi_f, M):
    """
    ISCO radius of a Kerr black hole (Boyer-Lindquist coordinates).

    Parameters
    ----------
    chi_f   : float (with sign) - dimensionless spin of the black hole
    M       : float [M_sun]     - mass of the black hole

    Returns
    -------
    r       : float [M_sun]     - ISCO radius (prograde orbit if chi_f > 0, retrograde otherwise)
    """
    Z_1 = 1.0 + (1.0 - chi_f**2)**(1/3) * (
        (1.0 + chi_f)**(1/3) + (1.0 - chi_f)**(1/3)
    )
    Z_2 = np.sqrt(3.0 * chi_f**2 + Z_1**2)
    if chi_f > 0:
        r = M * (3.0 + Z_2 - np.sqrt((3.0 - Z_1) * (3.0 + Z_1 + 2.0 * Z_2)))  # prograde ISCO
    else:
        r = M * (3.0 + Z_2 + np.sqrt((3.0 - Z_1) * (3.0 + Z_1 + 2.0 * Z_2)))  # retrograde ISCO
    return r   # in units of M_sun


def compute_f_isco(M, chi_f, r_isco):
    """
    Orbital frequency at the ISCO.

    Parameters
    ----------
    M       : float [M_sun]     - mass of the black hole
    chi_f   : float (with sign) - dimensionless spin of the black hole
    r_isco  : float [M_sun]     - ISCO radius

    Returns
    -------
    f_isco  : float [M_sun^-1]  - ISCO orbital frequency (divide by MSUN_SEC to get Hz)
    """
    omega = M**(1/2) / (r_isco**(3/2) + abs(chi_f) * M**(3/2))
    f_isco = (1.0 / (2 * np.pi)) * omega
    return f_isco   # in units of M_sun^-1


def isco_cutoff(M, q, chi_1, chi_2):
    """
    Cutoff frequency of the waveform: twice the ISCO orbital frequency of the remnant.

    Parameters
    ----------
    M         : float [M_sun]        - total mass of the binary
    q         : float                - mass ratio, q = m_1/m_2 >= 1
    chi_1     : float (with sign)    - spin of the first component
    chi_2     : float (with sign)    - spin of the second component

    Returns
    -------
    f_cutoff  : float [Hz]           - cutoff frequency = 2 * f_ISCO of the remnant
    chi_f     : float (with sign)    - spin of the remnant
    r_isco    : float [M_sun]        - ISCO radius of the remnant
    """
    mass_2 = M / (1.0 + q)
    mass_1 = M - mass_2
    chi_f  = bbh_final_spin_non_precessing_HBR2016(mass_1, mass_2, chi_1, chi_2)
    mass_f = bbh_final_mass_non_precessing_UIB2016(mass_1, mass_2, chi_1, chi_2)

    if not -1.0 <= chi_f <= 1.0:
        raise ValueError(f"chi_f = {chi_f} out of physical range [-1, 1]")

    if mass_f <= 0:
        raise ValueError(f"mass_f = {mass_f} out of physical range (0, inf)")

    r_isco = compute_r_isco(chi_f, mass_f)
    f_cutoff = 2 * compute_f_isco(mass_f, chi_f, r_isco) / MSUN_SEC   # in Hz
    return f_cutoff, chi_f, r_isco
