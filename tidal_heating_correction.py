"""
Tidal heating phase correction delta_psi(f) for TaylorF2, in the presence of a braneworld
charge q_BH = Q_BH / M^2.

See https://arxiv.org/pdf/2402.15336,
    https://arxiv.org/pdf/2202.08661, Eq. (5).

The correction has contributions at 2.5PN (with a logarithm), 3.5PN and 4PN (with a
logarithm). Only the primitive in v is kept: the terms coming from the lower limit of
integration are proportional to v^0 and v^3 and are absorbed into phi_c and t_c.

Spins are aligned or antialigned with the orbital angular momentum: the factor (L.J_i) = +1 (-1)
is encoded in the sign of chi_i.
"""

import numpy as np
from config.params import MSUN_SEC


def A5_single(m_i, M, chi_i, q_BH):
    """
    Contribution of the i-th black hole to the coefficient Psi_5.

    Parameters
    ----------
    m_i     : float [M_sun]     - mass of the i-th component
    M       : float [M_sun]     - total mass
    chi_i   : float (with sign) - spin of the i-th component
    q_BH    : float (with sign) - braneworld charge in range [-1, 1]

    Returns
    -------
    A5      : float             - 2.5PN term of the i-th component
    """
    x = (M / m_i)**2
    brackets = 1.0 + 3.0 * chi_i**2 + q_BH * x * (2.0 + 3.0 * chi_i**2 + q_BH * x)
    return (m_i / M)**3.0 * chi_i * brackets


def A8_single(m_i, M, chi_i, q_BH):
    """
    Contribution of the i-th black hole to the coefficient Psi_8.

    Parameters
    ----------
    m_i     : float [M_sun]     - mass of the i-th component
    M       : float [M_sun]     - total mass
    chi_i   : float (with sign) - spin of the i-th component
    q_BH    : float (with sign) - braneworld charge in range [-1, 1]

    Returns
    -------
    A8      : complex           - 4PN term of the i-th component
                                  (complex when 1 - chi_i^2 + q_BH (M/m_i)^2 < 0, i.e. no horizon)
    """
    x = (M / m_i)**2.0
    arg_sqrt = 1.0 - chi_i**2.0 + q_BH * x
    sqrt_term = np.sqrt(complex(arg_sqrt))  # imaginary if arg_sqrt < 0

    first_brackets  = 1.0 + sqrt_term + q_BH * x / 2.0
    second_brackets = 1.0 + 3.0 * chi_i**2.0 + q_BH * x * (2.0 + 3.0 * chi_i**2.0 + q_BH * x)
    return (m_i / M)**4 * first_brackets * second_brackets


def Psi5_and_Psi8(m_1, m_2, chi_1, chi_2, q_BH):
    """
    2.5PN and 4PN coefficients of the binary, summed over the two components.

    Parameters
    ----------
    m_1     : float [M_sun]     - mass of the first component
    m_2     : float [M_sun]     - mass of the second component
    chi_1   : float (with sign) - spin of the first component
    chi_2   : float (with sign) - spin of the second component
    q_BH    : float (with sign) - braneworld charge in range [-1, 1]

    Returns
    -------
    Psi5    : float             - sum of the 2.5PN terms of the two components
    Psi8    : complex           - sum of the 4PN terms of the two components
    """
    M = m_1 + m_2
    Psi_5 = A5_single(m_1, M, chi_1, q_BH) + A5_single(m_2, M, chi_2, q_BH)
    Psi_8 = A8_single(m_1, M, chi_1, q_BH) + A8_single(m_2, M, chi_2, q_BH)
    return Psi_5, Psi_8


def delta_psi_tidal(freqs, m_1, m_2, chi_1, chi_2, q_BH):
    """
    Tidal heating phase correction, to be added to the TaylorF2 phase psi(f).

    Parameters
    ----------
    freqs        : array-like (float) [Hz] - frequency grid
    m_1, m_2     : float [M_sun]           - component masses, m_1 >= m_2
    chi_1, chi_2 : float (with sign)       - dimensionless (anti)aligned spins
    q_BH         : float (with sign)       - braneworld charge in range [-1, 1]

    Returns
    -------
    delta_psi    : np.ndarray (complex)    - phase correction [rad]
    """
    freqs = np.asarray(freqs, dtype=float)
    M_sec = (m_1 + m_2) * MSUN_SEC          # total mass in units of time (s)
    eta   = (m_1 * m_2) / (m_1 + m_2)**2.0  # symmetric mass ratio
    q = m_1 / m_2                           # mass ratio

    pMf   = np.pi * M_sec * freqs           # dimensionless pi M f, so that v = pMf**(1/3)

    Psi_5, Psi_8 = Psi5_and_Psi8(m_1, m_2, chi_1, chi_2, q_BH)

    prefactor = 3.0 / (128.0 * eta)

    # Spin-orbit coefficient entering the 4PN term
    F_SO        = - 1.0 / 4.0 * (73.0 / (3.0 * (1.0 + q)**2.0) * (q**2.0 * chi_1 + chi_2) + 15.0 * q / (1.0 + q)**2.0 * (chi_1 + chi_2))

    # log(pMf) = 3 log(v)
    psi_2_dot_5 = - 10.0 / 9.0 * Psi_5 * (np.log(pMf) + 1.0)
    psi_3_dot_5 = - 5.0 / 168.0 * Psi_5 * (952.0 * eta + 995.0)
    psi_4       = - 20.0 / 9.0 * (np.log(pMf) - 1.0) * (Psi_5 * (F_SO + 4.0 * np.pi) + Psi_8)

    delta_psi = prefactor * pMf**(-5.0 / 3.0) * (psi_2_dot_5 * pMf**(5.0 / 3.0) + psi_3_dot_5 * pMf**(7.0 / 3.0) + psi_4 * pMf**(8.0 / 3.0))

    return delta_psi
