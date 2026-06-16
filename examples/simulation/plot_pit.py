"""
Generate PIT histogram figures for Designs 1/2/3.

Usage (from cpit package root):
  python -m examples.simulation.plot_pit
  python -m examples.simulation.plot_pit --designs 3 --output-dir docs/figures --seed 42

Requires matplotlib (optional dependency):
  pip install matplotlib
"""

import argparse
from pathlib import Path

import numpy as np

from cpit.bc import (
    fit_global_affine,
    fit_x_dependent_affine_gam,
    apply_affine,
    apply_affine_x,
)
from cpit import fit_conformal_calibrator, randomized_pit
from cpit.data_splitter import split_four_way, apply_split
from cpit.evaluation.pit_histogram import pit_histogram_counts

from .dgp import DESIGN1, DESIGN2, DESIGN3, DGP

DESIGNS = {1: DESIGN1, 2: DESIGN2, 3: DESIGN3}
DESIGN_LABELS = {
    1: "Design 1: global location-scale",
    2: "Design 2: x-dependent location-scale",
    3: "Design 3: shape misspecification",
}

METHOD_NAMES = ["Raw", "Bias-corrected (Global)", "Bias-corrected (GAM)", "CPIT (Global)", "CPIT (GAM)"]
N_BINS = 10
SEED = 42
M = 100
PIT_SEED = 43


def compute_pit_values(dgp: DGP, seed: int) -> dict[str, np.ndarray]:
    """Run one replicate and return PIT values for all 5 full-CDF methods."""
    n_total = 7000  # 0 + 1000 + 1000 + 5000
    rng_data = np.random.default_rng(seed)
    X_all = rng_data.uniform(0, 1, size=(n_total, 1))
    Y_all = dgp.generate_y(X_all, rng_data)

    split = split_four_way(
        n_total,
        train_frac=0.0,
        bias_frac=1000 / n_total,
        cal_frac=1000 / n_total,
        test_frac=5000 / n_total,
        seed=seed,
    )
    (_, _), (X_bias, y_bias), (X_cal, y_cal), (X_test, y_test) = apply_split(X_all, Y_all, split)

    rng_gen = np.random.default_rng(seed + 1)
    bias_s = np.array([dgp.sample_generator(X_bias[i], M, rng_gen) for i in range(len(X_bias))])
    cal_s = np.array([dgp.sample_generator(X_cal[i], M, rng_gen) for i in range(len(X_cal))])
    test_s = np.array([dgp.sample_generator(X_test[i], M, rng_gen) for i in range(len(X_test))])

    n_test = len(y_test)

    # Global BC
    params_g = fit_global_affine(y_bias, bias_s)
    cal_adj_g = np.array([apply_affine(cal_s[i], params_g) for i in range(len(X_cal))])
    test_adj_g = np.array([apply_affine(test_s[i], params_g) for i in range(n_test)])
    _, c_hat_g = fit_conformal_calibrator(cal_adj_g, y_cal, pit_seed=PIT_SEED)

    # GAM BC
    params_x = fit_x_dependent_affine_gam(X_bias, y_bias, bias_s, b_min=None, b_max=None)
    cal_adj_x = np.array([apply_affine_x(cal_s[i], X_cal[i], params_x) for i in range(len(X_cal))])
    test_adj_x = np.array([apply_affine_x(test_s[i], X_test[i], params_x) for i in range(n_test)])
    _, c_hat_x = fit_conformal_calibrator(cal_adj_x, y_cal, pit_seed=PIT_SEED + 10)

    # PIT values
    rng0 = np.random.default_rng(PIT_SEED)
    u_raw = np.array([randomized_pit(test_s[i], y_test[i], rng=rng0) for i in range(n_test)])

    rng1 = np.random.default_rng(PIT_SEED + 1)
    u_bc_g = np.array([randomized_pit(test_adj_g[i], y_test[i], rng=rng1) for i in range(n_test)])

    rng2 = np.random.default_rng(PIT_SEED + 4)
    u_bc_x = np.array([randomized_pit(test_adj_x[i], y_test[i], rng=rng2) for i in range(n_test)])

    rng3 = np.random.default_rng(PIT_SEED + 2)
    u_adj_g = np.array([randomized_pit(test_adj_g[i], y_test[i], rng=rng3) for i in range(n_test)])
    u_cpit_g = c_hat_g(u_adj_g)

    rng4 = np.random.default_rng(PIT_SEED + 3)
    u_adj_x = np.array([randomized_pit(test_adj_x[i], y_test[i], rng=rng4) for i in range(n_test)])
    u_cpit_x = c_hat_x(u_adj_x)

    return {
        "Raw": u_raw,
        "Bias-corrected (Global)": u_bc_g,
        "Bias-corrected (GAM)": u_bc_x,
        "CPIT (Global)": u_cpit_g,
        "CPIT (GAM)": u_cpit_x,
    }


def plot_pit_histograms(pit_dict: dict[str, np.ndarray], title: str, out_path: Path) -> None:
    """Plot 5 PIT histograms side by side (paper-ready format)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib") from e

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": False,
    })

    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.6), sharey=True)
    uniform_level = 1.0 / N_BINS

    for ax, name in zip(axes, METHOD_NAMES):
        u = pit_dict[name]
        counts = pit_histogram_counts(u, n_bins=N_BINS)
        freq = counts / len(u)
        ax.bar(
            np.arange(N_BINS) / N_BINS,
            freq,
            width=1.0 / N_BINS,
            edgecolor="0.3",
            linewidth=0.5,
            alpha=0.8,
            align="edge",
            color="#6699CC",
        )
        ax.axhline(uniform_level, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("PIT")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(length=2, pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    axes[0].set_ylabel("Relative freq.")
    fig.suptitle(title, fontsize=11, y=1.05)
    fig.tight_layout(w_pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PIT histogram figures.")
    parser.add_argument("--designs", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-dir", type=str, default="docs/figures")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    for d in args.designs:
        dgp = DESIGNS[d]
        label = DESIGN_LABELS[d]
        print(f"Design {d}: computing PIT values (seed={args.seed})...")
        pit_dict = compute_pit_values(dgp, args.seed)
        out_path = out_dir / f"design{d}_pit_histograms.png"
        plot_pit_histograms(pit_dict, label, out_path)

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == "__main__":
    main()
