"""
TaylorF2 waveform with the tidal heating phase correction of a braneworld-charged binary.
"""

import numpy as np

from scipy.signal import get_window
from isco import isco_cutoff
from TaylorF2 import TaylorF2
from tidal_heating_correction import delta_psi_tidal


def TaylorF2_charged(q_BH = 0.0, th_on = True, **kwargs):
    """
    TaylorF2 waveform with the phase correction due to tidal heating in the presence of a
    braneworld charge q_BH, h(f) = A(f) exp(-i [psi(f) + delta_psi(f)]).

    The point-particle baseline is the TaylorF2 of TaylorF2.py (phase from Appendix A of
    arXiv:2311.17554) rather than LALSimulation. The amplitude is kept at the restricted
    (Newtonian) order, as in LALSimulation, so that the two differ only in the phase, by the
    cubic-in-spin terms at 3.5 PN, which LAL does not include.

    The waveform is zeroed outside [f_low, f_cut], with f_cut = min(f_high, 2 * f_ISCO) of the
    remnant (see isco.py), and smoothly tapered at the band edges with a Tukey window.

    Parameters
    ----------
    q_BH     : float (with sign), optional, default=0.0 - braneworld charge in range [-1, 1]
    th_on    : bool, optional, default=True - if False, no tidal heating correction is applied
    **kwargs : dict - frequency-domain waveform parameters.
             Required:
             - mass_1, mass_2 : float [M_sun] - component masses, mass_1 >= mass_2
             - f_low          : float [Hz]    - starting frequency of the waveform
             - f_high         : float [Hz]    - highest frequency of the grid (Nyquist frequency);
                                                the actual cutoff is min(f_high, f_cut), where
                                                f_cut is computed with isco_cutoff
             Optional:
             - chi_1, chi_2   : float, default = 0.0       - dimensionless (anti)aligned spins
             - delta_f        : float [Hz], default = 1/32 - frequency resolution
             - distance       : float [Mpc], default = 500 - luminosity distance
             - inclination    : float [rad], default = 0.5 - inclination angle
             - phase          : float [rad], default = 0.0 - reference phase
             - time           : float [s], default = 0.0   - reference time
             - f_ref          : float [Hz], default = 0.0  - reference frequency; if 0, the
                                                             reference point is coalescence
                                                             (as in LALSimulation)
             - taper_hz       : float [Hz], default = 2.0  - width of the Tukey roll-off at
                                                             each band edge (0 = hard cut)

    Returns
    -------
    hp : np.ndarray (complex) - plus (+) polarization, zero outside [f_low, f_cut]
    hc : np.ndarray (complex) - cross (x) polarization, zero outside [f_low, f_cut]
    """

    required_params = ['mass_1', 'mass_2', 'f_low', 'f_high']
    missing_params = [
            param for param in required_params
            if param not in kwargs or kwargs[param] is None
        ]
    if missing_params:
        raise ValueError(
            f"TaylorF2_charged could not be generated, the following parameters are missing: "
            f"{missing_params}."
        )

    m_1 = kwargs['mass_1']                                     # required
    m_2 = kwargs['mass_2']                                     # required
    chi_1 = kwargs.get('chi_1', 0.0)
    chi_2 = kwargs.get('chi_2', 0.0)
    distance = kwargs.get('distance', 500)
    inclination = kwargs.get('inclination', 0.5)
    time = kwargs.get('time', 0.0)
    phase = kwargs.get('phase', 0.0)
    delta_f = kwargs.get('delta_f', 1.0 / 32.0)
    f_low = kwargs['f_low']                                    # required
    f_high = kwargs['f_high']                                  # required
    f_ref = kwargs.get('f_ref', 0.0)

    if m_1 < m_2:
        raise ValueError(
            f"TaylorF2_charged requires mass_1 >= mass_2. Input masses were: mass_1={m_1}, mass_2={m_2}."
        )

    q = m_1 / m_2
    f_cut, _ , _ = isco_cutoff(m_1 + m_2, q, chi_1, chi_2)
    f_cut = min(f_high, f_cut) # if f_cut is above the Nyquist frequency (f_high), the latter is the cutoff

    num_points = int(np.floor(f_high / delta_f)) + 1
    freqs = np.arange(num_points) * delta_f

    # Frequencies between f_low and f_cut
    mask = (freqs >= f_low) & (freqs <= f_cut)

    # Point-particle baseline
    hp_band, hc_band = TaylorF2(
        freqs[mask], m_1, m_2, chi_1, chi_2,
        distance=distance, time=time, phi=phase, iota=inclination, f_ref = f_ref,
    )

    hp = np.zeros(num_points, dtype=complex)
    hc = np.zeros(num_points, dtype=complex)
    hp[mask] = hp_band
    hc[mask] = hc_band

    # Tidal heating correction, referenced to f_ref like the point-particle phase
    if th_on:
        delta_psi = np.zeros(num_points, dtype=complex)
        delta_psi[mask] = delta_psi_tidal(freqs[mask], m_1, m_2, chi_1, chi_2, q_BH)
        if f_ref != 0.:
            delta_psi[mask] -= delta_psi_tidal(f_ref, m_1, m_2, chi_1, chi_2, q_BH)
        phase_correction = np.exp(-1j * delta_psi)
        hp[mask] *= phase_correction[mask]
        hc[mask] *= phase_correction[mask]

    # Smooth roll-off at the band edges, to avoid the discontinuity of a hard cut
    n = int(mask.sum())
    taper_hz = kwargs.get('taper_hz', 2.0)
    if taper_hz > 0 and n > 4 * taper_hz / delta_f:
        alpha = min(1.0, 2.0 * (taper_hz / delta_f) / n) # fraction of the samples over which the Tukey window tapers
        window = get_window(('tukey', alpha), n)
        hp[mask] *= window
        hc[mask] *= window

    hp[~mask] = 0.0
    hc[~mask] = 0.0

    return hp, hc
