"""Plot helpers for pion two-point benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.plot_settings import color_ls, default_plot, errorb_circle, fs_p, fs_small_p
from core.plot_settings import meff_label as default_meff_label


def momentum_label(momentum_mode: list[int] | tuple[int, int, int]) -> str:
    """Format a lattice momentum triplet for legends."""

    return f"({momentum_mode[0]}, {momentum_mode[1]}, {momentum_mode[2]})"


def save_effective_mass_plot(metrics: dict[str, object], output_path: Path) -> Path:
    """Save a meff summary plot for one benchmark result."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(metrics["effective_mass_times"], dtype=int)
    effective_mass = np.asarray(metrics["effective_mass"], dtype=float)
    effective_mass_stderr = np.asarray(metrics["effective_mass_stderr"], dtype=float)
    momentum_modes = [
        tuple(int(component) for component in mode)
        for mode in metrics["momentum_modes"]
    ]

    fig, ax = default_plot()
    plotted_any = False
    for idx, momentum_mode in enumerate(momentum_modes):
        valid = np.isfinite(effective_mass[idx])
        if not np.any(valid):
            continue
        plotted_any = True
        ax.errorbar(
            times[valid],
            effective_mass[idx, valid],
            yerr=effective_mass_stderr[idx, valid],
            color=color_ls[idx % len(color_ls)],
            label=momentum_label(momentum_mode),
            **errorb_circle,
        )

    ax.set_xlabel(r"$t_{\mathrm{sep}}$", **fs_p)
    ax.set_ylabel(default_meff_label, **fs_p)
    if plotted_any:
        ax.legend(ncol=2, loc="best", **fs_small_p)
    else:
        ax.text(
            0.5,
            0.5,
            "No valid periodic $m_{\\mathrm{eff}}$ points",
            ha="center",
            va="center",
            transform=ax.transAxes,
            **fs_small_p,
        )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
