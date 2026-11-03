"""
Overlay the posteriors of two sets of inference runs, one shared corner plot per configuration,
using bilby's own corner styling. Works for any figure (fig2, fig3, ...) and any pair of run
folders: every result in DIR_A is paired with the result in DIR_B that has the same label once
the noise tag `_O4bNN` and the `_uncharged` tag are removed.
Runs that differ only in f_ref (same labels in two folders) are paired the same way.

Median and 68% credible interval of each marginal are printed as a table.

Usage (from the repository root):
    python bilby_inference/scripts/shared_corner_plots.py
    python bilby_inference/scripts/shared_corner_plots.py --dir-a zero_noise/outdir_fig3 --dir-b with_noise/outdir_fig3_O4b_00
    python bilby_inference/scripts/shared_corner_plots.py --select 5D_q+0.00      # only labels containing this
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BILBY_DIR = os.path.dirname(SCRIPT_DIR)      # .../bilby_inference
PROJECT_DIR = os.path.dirname(BILBY_DIR)     # repository root
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, BILBY_DIR)                # lets bilby re-import source_model when reading results

import bilby

DEFAULT_OUTDIR = os.path.join(BILBY_DIR, "shared_corner_plots")

# Edit here (folders are relative to bilby_inference/; the command-line options override them)

# Comparison of two runs with different f_ref (e.g. 20 Hz and 200 Hz):
DIR_A = "zero_noise/outdir_fig5_fref20"   # directory of run A
DIR_B = "zero_noise/outdir_fig5_fref200"  # directory of run B
LABEL_A = r"$f_{ref} = 20$ Hz"            # legend entry for run A
LABEL_B = r"$f_{ref} = 200$ Hz"           # legend entry for run B

# Comparison of the models with and without charge (UNCHARGED_MODEL in the make_fig scripts):
# DIR_A = "zero_noise/outdir_fig5_q05"                  # directory of run A
# DIR_B = "zero_noise/outdir_fig5_q05_uncharged"        # directory of run B
# LABEL_A = r"model with charge ($q^{BH}$ free)"        # legend entry for run A
# LABEL_B = r"model without charge ($q^{BH} = 0$)"      # legend entry for run B

SELECT = ""           # optional substring of the labels (e.g. "5D_q+0.00")
SHOW_TITLES = False   # True: median and 68% CI of each run above the diagonal panels
LEGEND_FONTSIZE = 22  # size of the legend in the top-right corner
LABEL_FONTSIZE = 20   # size of the axis labels
TICK_FONTSIZE = 14    # size of the numbers on the axes

COLOURS = ["#1f77b4", "#d62728"]
REMOVE_TAGS = re.compile(r"_O4b\d{2}|_uncharged")  # tags ignored when pairing: noise realization, uncharged model

LATEX = {
    "q_BH": r"$q^{BH}$",
    "mass_1": r"$M_1$ ($M_\odot$)",
    "mass_2": r"$M_2$ ($M_\odot$)",
    "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$",
    "time_jitter": r"$t_j$",
}


def results_by_key(directory, select):
    """{label without noise and uncharged tags: path} for every *_result.json in `directory` containing `select`."""
    path = directory if os.path.isabs(directory) else os.path.join(BILBY_DIR, directory)
    if not os.path.isdir(path):
        raise SystemExit(f"No such directory: {path}")
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*_result.json"))):
        label = os.path.basename(f)[:-len("_result.json")]
        if select in label:
            out[REMOVE_TAGS.sub("", label)] = f
    return out


def credible_interval(samples, level=68.0):
    """Median and equal-tailed credible interval of a set of posterior samples."""
    tail = 0.5 * (100.0 - level)
    low, median, high = np.percentile(samples, [tail, 50.0, 100.0 - tail])
    return median, low, high


def prepare(result, parameters, label):
    """Restrict a result to `parameters` and apply LaTeX labels."""
    result.search_parameter_keys = list(parameters)
    result.parameter_labels_with_unit = [LATEX.get(p, p) for p in parameters]
    result.label = label
    return result


def shared_corner(file_a, file_b, label_a, label_b, skip, out_path, show_titles=SHOW_TITLES):
    """One overlaid corner plot for a pair of results, plus a table of the marginals."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import corner

    res_a = bilby.result.read_in_result(filename=file_a)
    res_b = bilby.result.read_in_result(filename=file_b)

    # Parameters sampled in both runs, in a stable order
    common = [k for k in res_a.search_parameter_keys
              if k in res_b.search_parameter_keys and k not in skip]
    if len(common) < 2:
        print(f"  skipped: fewer than two shared sampled parameters ({common})")
        return
    truths = [res_a.injection_parameters.get(k) for k in common]

    prepare(res_a, common, label_a)
    prepare(res_b, common, label_b)

    bilby.core.result.plot_multiple(
        [res_a, res_b],
        filename=out_path,
        save=False,
        labels=[label_a, label_b],
        colours=COLOURS,
        parameters=common,
        quantiles=None,
        smooth=0.9,
        bins=40,
        plot_datapoints=False,
        fill_contours=True,
        show_titles=False,
        label_kwargs=dict(fontsize=LABEL_FONTSIZE),
        dpi=150,
    )
    fig = plt.gcf()
    # Injected values (plot_multiple ignores `truths`): black dashed lines on every panel
    corner.overplot_lines(fig, truths, color="k", lw=1.0, ls="--")

    n = len(common)
    axes = np.array(fig.axes).reshape(n, n)

    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # Legend: bilby draws a small one on the top-right (empty) panel; replace it with a larger one
    legend_ax = axes[0, n - 1]
    old = legend_ax.get_legend()
    if old is not None:
        old.remove()
    handles = [Line2D([0], [0], color=c, lw=3) for c in COLOURS]
    legend_ax.legend(handles, [label_a, label_b], loc="upper right", frameon=False,
                     fontsize=LEGEND_FONTSIZE, handlelength=1.5)

    # Titles: one line per run, coloured to match the contours
    if show_titles:
        for i, k in enumerate(common):
            ax = axes[i, i]
            for j, (res, colour) in enumerate(((res_a, COLOURS[0]), (res_b, COLOURS[1]))):
                med, lo, hi = credible_interval(np.asarray(res.posterior[k]))
                ax.text(0.5, 1.01 + 0.13 * (1 - j),
                        f"${med:.3f}^{{+{hi - med:.3f}}}_{{-{med - lo:.3f}}}$",
                        transform=ax.transAxes, ha="center", va="bottom",
                        color=colour, fontsize=10)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    # Numerical comparison of the marginals (median and 68% interval)
    print(f"  {'parameter':<10} {'injected':>10} {'run A':>24} {'run B':>24}")
    for k, inj in zip(common, truths):
        cells = []
        for res in (res_a, res_b):
            med, lo, hi = credible_interval(np.asarray(res.posterior[k]))
            cells.append(f"{med:.3f} [{lo:.3f}, {hi:.3f}]")
        inj = float("nan") if inj is None else inj
        print(f"  {k:<10} {inj:10.4f} {cells[0]:>24} {cells[1]:>24}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir-a", default=DIR_A)
    p.add_argument("--dir-b", default=DIR_B)
    p.add_argument("--label-a", default=LABEL_A)
    p.add_argument("--label-b", default=LABEL_B)
    p.add_argument("--select", default=SELECT,
                   help="only compare labels containing this substring")
    p.add_argument("--skip", nargs="*", default=["time_jitter"],
                   help="parameters to leave out of the plot")
    p.add_argument("--titles", action="store_true", default=SHOW_TITLES,
                   help="show median and 68%% CI above the diagonal panels")
    args = p.parse_args()

    files_a = results_by_key(args.dir_a, args.select)
    files_b = results_by_key(args.dir_b, args.select)
    pairs = [key for key in files_a if key in files_b]
    print(f"{len(pairs)} pairs")
    only_a, only_b = sorted(set(files_a) - set(files_b)), sorted(set(files_b) - set(files_a))
    if only_a:
        print(f"  only in A (skipped): {only_a}")
    if only_b:
        print(f"  only in B (skipped): {only_b}")

    outdir = os.path.join(DEFAULT_OUTDIR, f"{os.path.basename(os.path.normpath(args.dir_a))}"
                                          f"_vs_{os.path.basename(os.path.normpath(args.dir_b))}")
    os.makedirs(outdir, exist_ok=True)

    for key in pairs:
        print(f"\n{key}")
        print(f"  A: {os.path.basename(files_a[key])}\n  B: {os.path.basename(files_b[key])}")
        shared_corner(files_a[key], files_b[key], args.label_a, args.label_b, args.skip,
                      os.path.join(outdir, f"{key}.png"), show_titles=args.titles)


if __name__ == "__main__":
    main()
