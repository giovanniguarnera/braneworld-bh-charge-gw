"""
Zero-noise injection with the Einstein Telescope 2L network (two L-shaped 15 km detectors,
Sardinia + Meuse-Rhine, arms at 45 deg): q^BH posterior for M = 32 Msun, mass ratio q = 3,
spins (0.7, 0.6), injected q^BH = Q_BH_INJECTED, network SNR = 800.

Requires the ET sensitivity curves in ET_sensitivity/ (see psd.py and the README).

Usage (from the repository root):
    python bilby_inference/scripts/make_fig_ET.py
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BILBY_DIR = os.path.dirname(SCRIPT_DIR)      # .../bilby_inference
PROJECT_DIR = os.path.dirname(BILBY_DIR)     # repository root
sys.path.insert(0, PROJECT_DIR)

from config.params import M_TOTAL_SIM, Q_MASS_RATIO, CHI_1, CHI_2
from bilby_inference.inference_utils import masses_from_q, run_injection, plot_posteriors

# Edit here
FREE_MASSES_AND_SPINS = True    # False -> fast 1D validation run (only q_BH free)
HORIZON_CONSTRAINT = False      # True  -> restrict to 1 - chi_i^2 + q_BH (M/m_i)^2 >= 0
Q_BH_INJECTED = 0.5             # injected charge
TARGET_SNR = 800.0              # network SNR
TIME_MARGINALIZATION = False
PHASE_MARGINALIZATION = False


def main():
    outdir = os.path.join(BILBY_DIR, "zero_noise", "outdir_fig_ET")
    os.makedirs(outdir, exist_ok=True)

    suffix = "5D" if FREE_MASSES_AND_SPINS else "1D"
    tag = f"{suffix}_q{Q_BH_INJECTED:+.2f}" + ("_hc" if HORIZON_CONSTRAINT else "")

    m_1, m_2 = masses_from_q(Q_MASS_RATIO, M_TOTAL_SIM)

    label = f"fig_ET_{tag}_chi_{CHI_1}_{CHI_2}_snr{int(TARGET_SNR)}"
    print(f"Running {label} ...")
    result = run_injection(
        m_1, m_2, CHI_1, CHI_2, q_BH_injected=Q_BH_INJECTED,
        target_snr=TARGET_SNR, label=label, outdir=outdir,
        free_masses_and_spins=FREE_MASSES_AND_SPINS,
        horizon_constraint=HORIZON_CONSTRAINT,
        time_marginalization=TIME_MARGINALIZATION, phase_marginalization=PHASE_MARGINALIZATION,
        einstein_telescope=True,
    )
    print(f"Injected luminosity distance: {result.injection_parameters['luminosity_distance']:.1f} Mpc")

    out_path = os.path.join(outdir, f"fig_ET_{tag}.png")
    plot_posteriors({f"$\\chi_1={CHI_1}, \\chi_2={CHI_2}$": result}, out_path,
                    injected_values=[Q_BH_INJECTED])
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
