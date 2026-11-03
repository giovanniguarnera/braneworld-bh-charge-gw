"""
Bilby-compatible frequency-domain source model wrapping TaylorF2_charged (TaylorF2
point-particle baseline + tidal heating phase correction with braneworld charge q_BH).

Note: the argument names mass_1, mass_2, chi_1, chi_2, luminosity_distance, iota, phase
are fixed by bilby (they must match the prior keys), so the standard bilby/LVK naming is
used here.
"""

import sys
import os
BILBY_DIR = os.path.dirname(os.path.abspath(__file__))   # .../bilby_inference
PROJECT_DIR = os.path.dirname(BILBY_DIR)                 # repository root
sys.path.insert(0, PROJECT_DIR)

import numpy as np

from config.params import F_LOW, F_REF
from TaylorF2_charged import TaylorF2_charged


def TaylorF2_charged_source_model(frequency_array, mass_1, mass_2, chi_1, chi_2, q_BH,
                                  luminosity_distance, iota, phase, **kwargs):
    """
    Parameters
    ----------
    frequency_array     : np.ndarray [Hz]   - frequency grid of bilby
    mass_1, mass_2      : float [M_sun]     - component masses
    chi_1, chi_2        : float (with sign) - dimensionless (anti)aligned spins
    q_BH                : float (with sign) - braneworld charge in range [-1, 1]
    luminosity_distance : float [Mpc]       - luminosity distance
    iota                : float [rad]       - inclination angle
    phase               : float [rad]       - reference phase
    **kwargs            : dict              - waveform_arguments of the WaveformGenerator
                                              ('minimum_frequency', 'reference_frequency')

    Returns
    -------
    strain  : dict - keys 'plus' and 'cross', np.ndarray (complex) of the same length
                     as frequency_array
    """
    delta_f = frequency_array[1] - frequency_array[0]
    f_low = kwargs.get('minimum_frequency', F_LOW)
    f_ref = kwargs.get('reference_frequency', F_REF)
    f_high = frequency_array[-1]

    hp, hc = TaylorF2_charged(
        q_BH=q_BH, mass_1=mass_1, mass_2=mass_2, chi_1=chi_1, chi_2=chi_2,
        distance=luminosity_distance, inclination=iota, phase=phase,
        delta_f=delta_f, f_low=f_low, f_high=f_high, f_ref=f_ref,
    )

    # Truncate (or zero-pad) the arrays returned by TaylorF2_charged to the length bilby expects
    num_points = len(frequency_array)
    hp_out = np.zeros(num_points, dtype=complex)
    hc_out = np.zeros(num_points, dtype=complex)
    num_available = min(num_points, len(hp))
    hp_out[:num_available] = hp[:num_available]
    hc_out[:num_available] = hc[:num_available]

    return {'plus': hp_out, 'cross': hc_out}
