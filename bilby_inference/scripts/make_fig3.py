"""
Fig. 3: posterior distributions of the braneworld charge q^BH for four
mass ratios, at fixed total mass M = 32 Msun, spins (chi_1, chi_2) = (0.7, 0.6),
network SNR = 120, injected q^BH = Q_BH_INJECTED.

Two noise settings:
    zero noise (INJECTION_INDEX = None) : O5c design PSD, zero-noise strain (paper reproduction)
                                          outdir -> bilby_inference/zero_noise/outdir_fig3
    real O4b noise (INJECTION_INDEX = n): PSD measured and noise taken from O4b data at the n-th
                                          time of segments/injection_times.json
                                          outdir -> bilby_inference/with_noise/outdir_fig3_O4b_<n>

Usage (from the repository root):
    python bilby_inference/scripts/make_fig3.py              # uses INJECTION_INDEX below
    python bilby_inference/scripts/make_fig3.py --index 3    # real O4b noise, realization 03
    python bilby_inference/scripts/make_fig3.py --zero-noise # zero noise
"""

import sys
import os
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BILBY_DIR = os.path.dirname(SCRIPT_DIR)      # .../bilby_inference
PROJECT_DIR = os.path.dirname(BILBY_DIR)     # repository root
sys.path.insert(0, PROJECT_DIR)

from config.params import M_TOTAL_SIM, Q_MASS_RATIO_CONFIGS_FIG3, CHI_1, CHI_2, TARGET_SNR
from bilby_inference.inference_utils import masses_from_q, run_injection, plot_posteriors

# Edit here
FREE_MASSES_AND_SPINS = True    # False -> fast 1D validation run (only q_BH free)
HORIZON_CONSTRAINT = False      # True  -> restrict to 1 - chi_i^2 + q_BH (M/m_i)^2 >= 0
Q_BH_INJECTED = 0.              # injected charge
TEST_RUN = False                # True  -> only the first configuration
INJECTION_INDEX = None          # None -> zero noise (O5 PSD); int -> index in segments/injection_times.json (real O4b noise)
TIME_MARGINALIZATION = False
PHASE_MARGINALIZATION = False
UNCHARGED_MODEL = False         # True -> inference with q_BH = 0 (the signal is still injected with Q_BH_INJECTED)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, default=None,
                       help="noise realization (index in injection_times.json); overrides INJECTION_INDEX")
    group.add_argument("--zero-noise", action="store_true", help="zero noise; overrides INJECTION_INDEX")
    args = parser.parse_args()
    injection_index = None if args.zero_noise else (args.index if args.index is not None else INJECTION_INDEX)

    if injection_index is None:
        injection_time, noise_tag = None, ""
        outdir = os.path.join(BILBY_DIR, "zero_noise", "outdir_fig3")
        print("Zero noise, O5 design PSD")
    else:
        with open(os.path.join(BILBY_DIR, "segments", "injection_times.json")) as f:
            injection_time = json.load(f)["geocent_times"][injection_index]
        noise_tag = f"_O4b{injection_index:02d}"
        outdir = os.path.join(BILBY_DIR, "with_noise", f"outdir_fig3_O4b_{injection_index:02d}")
        print(f"Real O4b noise, realization {injection_index:02d}, injection time {injection_time}")
    os.makedirs(outdir, exist_ok=True)

    suffix = "5D" if FREE_MASSES_AND_SPINS else "1D"
    tag = (f"{suffix}_q{Q_BH_INJECTED:+.2f}" + ("_hc" if HORIZON_CONSTRAINT else "")
           + ("_uncharged" if UNCHARGED_MODEL else "") + noise_tag)

    mass_ratios = Q_MASS_RATIO_CONFIGS_FIG3[:1] if TEST_RUN else Q_MASS_RATIO_CONFIGS_FIG3

    results = {}
    for q in mass_ratios:
        m_1, m_2 = masses_from_q(q, M_TOTAL_SIM)
        label = f"fig3_{tag}_q_{q:g}"
        print(f"Running {label} ...")
        results[f"$q = {q:g}$"] = run_injection(
            m_1, m_2, CHI_1, CHI_2, q_BH_injected=Q_BH_INJECTED,
            target_snr=float(TARGET_SNR), label=label, outdir=outdir,
            free_masses_and_spins=FREE_MASSES_AND_SPINS,
            horizon_constraint=HORIZON_CONSTRAINT,
            time_marginalization=TIME_MARGINALIZATION, phase_marginalization=PHASE_MARGINALIZATION,
            injection_time=injection_time, uncharged_model=UNCHARGED_MODEL,
        )

    # With the uncharged model q_BH is not sampled: there is no q_BH posterior to plot
    if not UNCHARGED_MODEL:
        out_path = os.path.join(outdir, f"fig3_{tag}.png")
        plot_posteriors(results, out_path, injected_values=[Q_BH_INJECTED])
        print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
