"""
Point-particle TaylorF2 waveform (stationary phase approximation), used as the GR baseline
of TaylorF2_charged.
"""

import numpy as np

from config.params import MSUN_SEC
from config.params import MPC_SEC


def TaylorF2(freqs, m_1, m_2, chi_1, chi_2, distance, time, phi, iota, f_ref=0.0):
    """
    TaylorF2 waveform in the stationary phase approximation, in the point-particle limit, with
    the phase up to 3.5 PN and the amplitude at the restricted (Newtonian) order A0 f^(-7/6),
    as in LALSimulation.

    Phase: Eq. (A2)-(A9) of https://arxiv.org/pdf/2311.17554. The non-spinning coefficients
        coincide with Eq. (3.18) of https://arxiv.org/pdf/0907.0700; the cubic-in-spin terms
        at 3.5 PN are NOT implemented by the TaylorF2 approximant of LALSimulation, which only
        adds the spin-orbit coefficient at that order.

    Amplitude: the PN corrections a_1 ... a_7 of Eq. (5.7) of https://arxiv.org/pdf/1107.1267,
        obtained by expanding [-E'(v)/F(v)]^(1/2) in powers of v, are kept in the code for
        reference but are not used (restricted TaylorF2).

    Convention: h = A(f) exp(-i psi(f)), the same as LALSimulation.

    Parameters
    ----------
    freqs        : array-like (float) [Hz] - frequency grid
    m_1, m_2     : float [M_sun]           - component masses, m_1 >= m_2
    chi_1, chi_2 : float (with sign)       - dimensionless (anti)aligned spins
    distance     : float [Mpc]             - luminosity distance
    time         : float [s]               - reference time
    phi          : float [rad]             - reference phase
    iota         : float [rad]             - inclination angle
    f_ref        : float [Hz], default=0.0 - reference frequency; if 0, the reference point is
                                             coalescence (as in LALSimulation)

    Returns
    -------
    hp : np.ndarray (complex) - plus (+) polarization
    hc : np.ndarray (complex) - cross (x) polarization
    """
    GammaE = 0.577215664901532   # Euler-Mascheroni constant

    freqs = np.asarray(freqs, dtype=float)

    M = m_1 + m_2
    M_sec = M * MSUN_SEC
    distance_sec = distance * MPC_SEC

    # Symmetric and antisymmetric spin combinations
    chi_s = 0.5 * (chi_1 + chi_2)
    chi_a = 0.5 * (chi_1 - chi_2)

    eta = (m_1 * m_2) / (m_1 + m_2)**2
    delta = (1. - 4. * eta)**0.5
    v  = (np.pi * M_sec * freqs)**(1. / 3.)

    # Spin combinations entering the PN coefficients
    beta = (113. / 12.)*(chi_s + delta * chi_a - (76. * eta / 113.) * chi_s)
    sigma = chi_a**2 * ((81. / 16.) - 20 * eta) + (81. * chi_a * chi_s * delta)/8. + chi_s**2 * ((81. / 16.) - eta / 4.)
    epsilon = (502429. / 16128. - 907. * eta / 192.) * delta * chi_a + (5. * eta**2 / 48. - 73921. *eta / 2016. + 502429. / 16128.) * chi_s
    gamma = (732985. / 2268. - 24260. * eta / 81. - 340. * eta ** 2 / 9.) * chi_s + (732985. / 2268. + 140. * eta / 9.) * delta * chi_a

    # Amplitude (see https://arxiv.org/pdf/1107.1267)
    A = ((5. * eta / 24.)**0.5 / np.pi**(2. / 3.)) * (M_sec**(5. / 6.) / distance_sec)

    # PN amplitude corrections: not used (restricted TaylorF2), kept for reference
    a0 = 1.

    a1 = 0

    a2 = 11. * eta / 8. + 743. / 672.

    a3 = beta / 2. - 2. * np.pi

    a4 = 1379. * eta**2 / 1152. + 18913. * eta / 16128. + 7266251. / 8128512. - sigma / 2.

    a5 = 57. * np.pi * eta / 16. - 4757. * np.pi / 1344. + epsilon

    a6 = 856. * GammaE / 105. + 67999. * eta**3 / 82944. - 1041557. * eta**2 / 258048. - 451. * np.pi**2 * eta / 96. + 10. * np.pi**2 / 3. + 3526813753. * eta / 27869184. - 29342493702821. / 500716339200. + 856. * np.log (4 * v) / 105.

    a7 = - 1349. * np.pi * eta**2 / 24192. - 72221. * np.pi * eta / 24192. - 5111593. * np.pi / 2709504.

    # 3.5PN phasing, point-particle limit
    # (see https://arxiv.org/pdf/0907.0700, https://arxiv.org/pdf/2311.17554,
    #  https://arxiv.org/pdf/1107.1267 and https://arxiv.org/pdf/1601.05588)
    p0 = 1.

    p1 = 0

    p2 = (3715. / 756. + (55. * eta) / 9.)

    p3 = (-16. * np.pi + (113. * delta * chi_a) / 3. + (113. / 3. - (76. * eta) / 3.) * chi_s)

    p4 = (15293365. / 508032. + (27145. * eta) / 504. + (3085. * eta**2) / 72. + (-405. / 8. + 200. * eta) * chi_a**2 - (405. * delta * chi_a * chi_s) / 4. + (-405. / 8. + (5. * eta) / 2.) * chi_s**2)

    p5 = (38645. * np.pi / 756. - 65. * np.pi * eta / 9. - gamma)

    p6 = (11583231236531. / 4694215680. - 640. / 3.* np.pi**2 - 6848. / 21. * GammaE + eta * (-15737765635. / 3048192. + 2255. / 12.* np.pi**2) + eta**2 * 76055. / 1728. - eta**3 * 127825. /1296. \
         - (6848. / 21.) * np.log(4.) + np.pi * (2270. * delta * chi_a / 3. + (2270. / 3. - 520. * eta) * chi_s) + (75515. / 144. - 8225. * eta / 18.) * delta * chi_a * chi_s \
         + (75515. / 288. - 263245. * eta / 252. - 480. * eta**2) * chi_a**2 + (75515. / 288. - 232415. * eta / 504. + 1255. * eta**2 / 9.) * chi_s**2)

    p7 = (((77096675. * np.pi) / 254016. + (378515. * np.pi * eta) / 1512.- (74045. * np.pi * eta**2) / 756. + (-25150083775. / 3048192. + (10566655595. * eta) / 762048. - (1042165. * eta**2) / 3024. + (5345. * eta**3) / 36.
         + (14585. / 8. - 7270. * eta + 80. * eta**2) * chi_a**2) * chi_s + (14585. / 24. - (475. * eta) / 6. + (100. * eta**2) / 3.) * chi_s**3 + delta * ((-25150083775. / 3048192.
         + (26804935. * eta) / 6048. - (1985. * eta**2) / 48.) * chi_a + (14585. / 24. - 2380. * eta) * chi_a**3 + (14585. / 8. - (215. * eta) / 2.) * chi_a * chi_s**2)))

    def pn_sum(v):
        """PN phase series at velocity v, including the logarithmic terms at 2.5PN and 3PN."""
        p5L = (38645. * np.pi / 756. - 65. * np.pi * eta / 9. - gamma) * 3 * np.log(v)
        p6L = - (6848. / 21.) * np.log(v)
        return (3. / (128. * v**5 * eta))*(p0 + v * p1 + v**2 * p2 + v**3 * p3+ v**4 * p4 + v**5 * (p5 + p5L) + v**6 * (p6 + p6L) + v**7 * p7)

    phase = 2*freqs*np.pi*time - phi - np.pi/4. + pn_sum(v)
    if f_ref != 0.:
        # Subtract the phase at the reference frequency
        v_ref = (np.pi*M_sec*f_ref)**(1./3.)
        phase -= pn_sum(v_ref)

    # Restricted amplitude: only the leading order is kept.
    # The full 3.5PN amplitude would be
    # A * freqs**(-7/6) * (a0 + v*a1 + v**2*a2 + v**3*a3 + v**4*a4 + v**5*a5 + v**6*a6 + v**7*a7)
    amp = A * freqs**(-7. /6.)

    hp = 0.5 * (1. + (np.cos(iota))**2) * amp * np.exp(-1j * phase)
    hc = -1j * np.cos(iota) * amp * np.exp(-1j * phase)

    return hp, hc
