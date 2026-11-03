"""
Mismatch between the charged-BH and Kerr-BH TaylorF2 waveforms, as a
function of the braneworld charge q^BH, for four binary configurations at fixed total mass
M = 32 Msun, with f_low = 20 Hz and the O5c design PSD.

Requires pycbc (used only here, for the match).

Usage (from the repository root):
    python scripts/make_fig1.py
The figure is saved to img/fig1_mismatch_vs_charge.png.
"""

import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pycbc import types
from pycbc.filter import match as pycbc_match

from TaylorF2_charged import TaylorF2_charged
from isco import isco_cutoff
from psd import get_psd_O5
from plotting import plot_mismatch_vs_qBH

M_TOTAL = 32.0           # Msun
F_LOW = 20.0             # Hz
F_HIGH = 1024.0          # Hz
DELTA_F = 1.0 / 32.0     # Hz
PAD_FACTOR = 16          # zero-padding of the frequency series: time resolution 1 / (2 * F_HIGH * PAD_FACTOR)


# Binary configurations: (q = m_1/m_2, chi_1, chi_2, label, color)

# Configurations of Fig. 1 of arXiv:2402.15336
CONFIGS = [
    (1.0, 0.7, 0.6, r"q=1, $\chi_1=0.7$, $\chi_2=0.6$", "tab:blue"),
    (3.0, 0.7, 0.6, r"q=3, $\chi_1=0.7$, $\chi_2=0.6$", "tab:orange"),
    (1.0, 0.7, 0.1, r"q=1, $\chi_1=0.7$, $\chi_2=0.1$", "tab:green"),
    (3.0, 0.7, 0.1, r"q=3, $\chi_1=0.7$, $\chi_2=0.1$", "tab:red"),
]

# Alternative: fixed mass ratio, varying spins
# CONFIGS = [
#     (3.0, 0.7, 0.6, r"q=3, $\chi_1=0.7$, $\chi_2=0.6$", "tab:blue"),
#     (3.0, 0.7, 0.1, r"q=3, $\chi_1=0.7$, $\chi_2=0.1$", "tab:orange"),
#     (3.0, 0.5, 0.3, r"q=3, $\chi_1=0.5$, $\chi_2=0.3$", "tab:green"),
#     (3.0, 0.2, 0.1, r"q=3, $\chi_1=0.2$, $\chi_2=0.1$", "tab:red"),
# ]

# Alternative: fixed spins, varying mass ratio
# CONFIGS = [
#     (1.0, 0.7, 0.6, r"q=1, $\chi_1=0.7$, $\chi_2=0.6$", "tab:blue"),
#     (2.0, 0.7, 0.6, r"q=2, $\chi_1=0.7$, $\chi_2=0.6$", "tab:orange"),
#     (3.0, 0.7, 0.6, r"q=3, $\chi_1=0.7$, $\chi_2=0.6$", "tab:green"),
#     (4.0, 0.7, 0.6, r"q=4, $\chi_1=0.7$, $\chi_2=0.6$", "tab:red"),
# ]

# q_BH = 0 is excluded (it would compare GR with itself)
Q_BH_GRID = np.concatenate([
    np.linspace(-1.0, -0.005, 250),
    np.linspace(0.005, 1.0, 250),
])


def masses_from_q(q, M_total):
    """
    Same as bilby_inference.inference_utils.masses_from_q, duplicated here so that this
    script does not need bilby.

    Parameters
    ----------
    q       : float         - mass ratio, q = m_1/m_2 >= 1
    M_total : float [M_sun] - total mass

    Returns
    -------
    m_1, m_2 : float [M_sun] - component masses
    """
    m_2 = M_total / (1.0 + q)
    m_1 = M_total - m_2
    return m_1, m_2


def compute_mismatch(h_1, h_2, psd, delta_f, f_low, f_high):
    """
    Mismatch 1 - match, with the match maximized over t_c and phi_c (pycbc.filter.match).
    The series are zero-padded by PAD_FACTOR to refine the time-shift maximization.

    Parameters
    ----------
    h_1, h_2 : np.ndarray (complex)          - frequency-domain waveforms
    psd      : np.ndarray or FrequencySeries - PSD on the same frequency grid
    delta_f  : float [Hz]                    - frequency resolution
    f_low    : float [Hz]                    - lower limit of the overlap integral
    f_high   : float [Hz]                    - upper limit of the overlap integral

    Returns
    -------
    mismatch : float
    """
    s_1 = types.FrequencySeries(np.asarray(h_1, dtype=complex), delta_f=delta_f)
    s_2 = types.FrequencySeries(np.asarray(h_2, dtype=complex), delta_f=delta_f)
    psd_series = psd if isinstance(psd, types.FrequencySeries) else types.FrequencySeries(np.asarray(psd), delta_f=delta_f)
    n = len(s_1)
    tlen = PAD_FACTOR * (n - 1) + 1
    s_1.resize(tlen)
    s_2.resize(tlen)
    psd_padded = types.FrequencySeries(np.ones(tlen), delta_f=delta_f)
    psd_padded[:n] = psd_series[:n]          # PSD beyond F_HIGH is irrelevant (outside the cutoff)
    m, _ = pycbc_match(s_1, s_2, psd=psd_padded, low_frequency_cutoff=f_low,
                       high_frequency_cutoff=f_high, subsample_interpolation=True)
    return 1.0 - m


def horizon_exists(m_1, m_2, chi_1, chi_2, q_BH):
    """
    Check that 1 - chi_i^2 + q_BH (M/m_i)^2 >= 0 for i = 1, 2.

    Parameters
    ----------
    m_1, m_2     : float [M_sun]     - component masses
    chi_1, chi_2 : float (with sign) - dimensionless (anti)aligned spins
    q_BH         : float (with sign) - braneworld charge

    Returns
    -------
    valid : bool - True if both horizons exist
    """
    M = m_1 + m_2
    for m_i, chi_i in ((m_1, chi_1), (m_2, chi_2)):
        x_i = (M / m_i) ** 2
        if 1.0 - chi_i**2 + q_BH * x_i < 0.0:
            return False
    return True


def mismatch_curve(q, chi_1, chi_2):
    """
    Mismatch between the charged and the uncharged (q_BH = 0) waveform for every q_BH in Q_BH_GRID.

    Parameters
    ----------
    q            : float             - mass ratio, q = m_1/m_2 >= 1
    chi_1, chi_2 : float (with sign) - dimensionless (anti)aligned spins

    Returns
    -------
    mismatches  : np.ndarray        - same length as Q_BH_GRID
    has_horizon : np.ndarray (bool) - True where both horizons exist
    """
    m_1, m_2 = masses_from_q(q, M_TOTAL)

    # f_cut does not depend on q_BH: no q_BH-dependent ISCO fit is currently
    # available, so the standard Kerr fit is used for every q_BH
    f_cut, _, _ = isco_cutoff(M_TOTAL, q, chi_1, chi_2)
    f_cut = min(F_HIGH, f_cut)

    hp_gr, _ = TaylorF2_charged(
        q_BH=0.0, mass_1=m_1, mass_2=m_2, chi_1=chi_1, chi_2=chi_2,
        f_low=F_LOW, f_high=F_HIGH, delta_f=DELTA_F,
    )
    num_points = len(hp_gr)
    psd = get_psd_O5(num_points, DELTA_F, F_LOW)

    mismatches = np.empty_like(Q_BH_GRID)
    has_horizon = np.zeros_like(Q_BH_GRID, dtype=bool)
    for i, qbh in enumerate(Q_BH_GRID):
        has_horizon[i] = horizon_exists(m_1, m_2, chi_1, chi_2, qbh)
        hp_q, _ = TaylorF2_charged(q_BH=qbh, mass_1=m_1, mass_2=m_2, chi_1=chi_1, chi_2=chi_2,
                                   f_low=F_LOW, f_high=F_HIGH, delta_f=DELTA_F)
        mismatches[i] = compute_mismatch(hp_q, hp_gr, psd, DELTA_F, F_LOW, f_cut)
    return mismatches, has_horizon


def main():
    """
    Compute the mismatch curves for all CONFIGS and save the figure to img/.

    Returns
    -------
    results : list of dict - input of plotting.plot_mismatch_vs_qBH
    """
    results = []
    for q, chi_1, chi_2, label, color in CONFIGS:
        print(f"Computing mismatch curve: {label} ...")
        mismatches, has_horizon = mismatch_curve(q, chi_1, chi_2)
        results.append({
            "q_BH": Q_BH_GRID,
            "mismatch": mismatches,
            "has_horizon": has_horizon,
            "label": label,
            "color": color,
        })

    img_dir = os.path.join(PROJECT_DIR, "img")
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(img_dir, "fig1_mismatch_vs_charge.png")
    plot_mismatch_vs_qBH(results, save_path=out_path)
    print(f"Saved figure to {out_path}")
    return results


if __name__ == "__main__":
    main()
