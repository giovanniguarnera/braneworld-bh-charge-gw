"""
Corner plots and credible intervals for every inference result in an output directory.

Reads all `*_result.json` files, works out which parameters were sampled in each
run, and marks the injected values. Handles both the 1D runs (only q_BH free) and
the 5D ones (masses and spins free as well) without being told which is which.

Usage (from the repository root):
    python bilby_inference/scripts/diagnostic_corner.py                    # uses OUTDIR below
    python bilby_inference/scripts/diagnostic_corner.py path/to/outdir
"""

import glob
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BILBY_DIR = os.path.dirname(SCRIPT_DIR)      # .../bilby_inference
PROJECT_DIR = os.path.dirname(BILBY_DIR)     # repository root
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, BILBY_DIR)                # lets bilby re-import source_model when reading results

import bilby

# Edit here
OUTDIR = "zero_noise/outdir_fig2"    # directory holding the *_result.json files, relative to bilby_inference/

DEFAULT_OUTDIR = os.path.join(BILBY_DIR, OUTDIR)

LATEX = {
    "q_BH": r"$q^{BH}$",
    "mass_1": r"$M_1$",
    "mass_2": r"$M_2$",
    "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$",
    "time_jitter": r"$t_j$",
}


def credible_interval(samples, level=68.0):
    """Median and equal-tailed credible interval of a set of posterior samples."""
    tail = 0.5 * (100.0 - level)
    low, median, high = np.percentile(samples, [tail, 50.0, 100.0 - tail])
    return median, low, high


def summarise(result, label):
    """Print the injected value, median and 68% interval of every sampled parameter."""
    print(f"\n{label}")
    print(f"  {'parameter':<10} {'injected':>10} {'median':>10} "
          f"{'68% low':>10} {'68% high':>10} {'width':>9}")
    for name in result.search_parameter_keys:
        samples = np.asarray(result.posterior[name])
        median, low, high = credible_interval(samples)
        injected = result.injection_parameters.get(name, float("nan"))
        print(f"  {name:<10} {injected:10.4f} {median:10.4f} "
              f"{low:10.4f} {high:10.4f} {high - low:9.4f}")


def process(path):
    """Summarise one result file and save its corner plot next to it."""
    result = bilby.result.read_in_result(filename=path)
    label = os.path.basename(path).replace("_result.json", "")

    summarise(result, label)

    sampled = result.search_parameter_keys
    if len(sampled) < 2:
        # A corner plot of a single parameter carries no more information than
        # the marginal already produced by the make_fig scripts, so skip it.
        print("  (single sampled parameter: corner plot skipped)")
        return

    truths = {name: result.injection_parameters[name]
              for name in sampled if name in result.injection_parameters}

    for name in sampled:
        if name in LATEX:
            result.parameter_labels_with_unit[sampled.index(name)] = LATEX[name]

    figure_path = os.path.join(os.path.dirname(path), f"{label}_corner.png")
    result.plot_corner(parameters=truths, filename=figure_path, dpi=150,
                       quantiles=[0.16, 0.84])
    print(f"  corner plot -> {figure_path}")


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTDIR
    paths = sorted(glob.glob(os.path.join(outdir, "*_result.json")))

    if not paths:
        raise SystemExit(f"No result files found in {outdir}")

    print(f"Found {len(paths)} result file(s) in {outdir}")
    for path in paths:
        process(path)


if __name__ == "__main__":
    main()
