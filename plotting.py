"""
Plotting utilities for the mismatch-vs-charge figure
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_mismatch_vs_qBH(results, save_path=None):
    """
    Mismatch (1 - match) vs braneworld charge q^BH, for given binary configurations.
    Solid lines where both horizons exist, dashed lines where at least one does not.

    Parameters
    ----------
    results   : list of dict, each with keys:
                'q_BH'       : np.ndarray        - braneworld charge in range [-1, 1]
                'mismatch'   : np.ndarray        - mismatch between the two waveforms
                'has_horizon': np.ndarray (bool) - True when both horizons exist,
                                                   1 - chi_i^2 + q_BH (M/m_i)^2 >= 0
                'label'      : str               - legend label
                'color'      : str               - line color
    save_path : str, optional - if given, the figure is saved to this path

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for r in results:
        q_bh, mm, horizon_check = r["q_BH"], r["mismatch"], r["has_horizon"]
        solid = np.where(horizon_check, mm, np.nan)

        # extend the dashed segment by one point so that it joins the solid one
        dashed_mask = ~horizon_check | np.roll(~horizon_check, 1) | np.roll(~horizon_check, -1)
        dashed = np.where(dashed_mask, mm, np.nan)
        ax.plot(q_bh, solid, color=r["color"], lw=1.8, label=r["label"])
        ax.plot(q_bh, dashed, color=r["color"], lw=1.8, ls="--")

    ax.set_yscale("log")
    ax.tick_params(axis="both", which="both", labelsize=18)
    ax.set_xlabel(r"$q^{\mathrm{BH}}$", fontsize=16)
    ax.set_ylabel(r"$1 - \mathcal{M}$", fontsize=20)
    ax.set_xlim(-1.05, 1.05)
    ax.legend(loc="lower right", frameon=False, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig, ax
