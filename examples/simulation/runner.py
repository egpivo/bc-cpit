"""
Shared pipeline for running one simulation replicate.

All three designs use the same evaluation pipeline; only the DGP differs.
Also provides aggregate_replicates() for multi-rep summary.
"""

import numpy as np
import pandas as pd

from cpit.data_splitter import split_four_way, apply_split
from cpit.bc import fit_global_affine, fit_x_dependent_affine_gam, apply_affine, apply_affine_x
from cpit import fit_conformal_calibrator, randomized_pit, get_weighted_samples_at_x, quantile_from_weighted_samples_batched
from cpit.evaluation import (
    cramer_von_mises,
    crps_from_weighted_samples,
    evaluate_interval_methods,
    intervals_from_sample_matrix,
    quantile_calibration_error_from_sample_matrix,
    quantile_calibration_error_from_weighted_matrix,
)
from cpit.baselines.quantile_baselines import cqr_intervals_batch
from cpit.baselines.score_baselines import (
    sr_split_conformal_intervals_batch,
    crps_split_conformal_intervals_batch,
)
from cpit.evaluation.local_diagnostics import local_diagnostics

from .dgp import DGP

N_TRAIN = 0
N_BIAS = 1000
N_CAL = 1000
N_TEST = 5000
M = 100
PIT_SEED = 43
ALPHAS = [0.02, 0.05, 0.10, 0.20]
TAU_GRID = np.arange(0.05, 1.0, 0.05)


def run_replicate(
    dgp: DGP,
    seed: int,
    *,
    n_train: int = N_TRAIN,
    n_bias: int = N_BIAS,
    n_cal: int = N_CAL,
    n_test: int = N_TEST,
    m: int = M,
    pit_seed: int = PIT_SEED,
    alphas: list[float] | None = None,
    tau_grid: np.ndarray | None = None,
    tau_grid_extra: np.ndarray | None = None,
    gam_n_splines: int | None = None,
    mode: str = "all",
) -> dict:
    """Run one replicate for any DGP; return dict of metrics.

    tau_grid_extra: additional τ levels reported separately (e.g. upper-tail
    for Design 3). QErr on ``tau_grid`` and ``tau_grid_extra`` are returned
    as separate keys so they are not mixed.
    """
    alphas = alphas or ALPHAS
    tau_grid = tau_grid if tau_grid is not None else TAU_GRID
    n_total = n_train + n_bias + n_cal + n_test

    # ── Data generation ──
    rng_data = np.random.default_rng(seed)
    X_all = rng_data.uniform(0, 1, size=(n_total, 1))
    Y_all = dgp.generate_y(X_all, rng_data)

    split = split_four_way(
        n_total,
        train_frac=n_train / n_total,
        bias_frac=n_bias / n_total,
        cal_frac=n_cal / n_total,
        test_frac=n_test / n_total,
        seed=seed,
    )
    (_, _), (X_bias, y_bias), (X_cal, y_cal), (X_test, y_test) = apply_split(X_all, Y_all, split)

    # ── Generator samples ──
    rng_gen = np.random.default_rng(seed + 1)
    bias_samples = np.array([dgp.sample_generator(X_bias[i], m, rng_gen) for i in range(len(X_bias))])
    cal_samples = np.array([dgp.sample_generator(X_cal[i], m, rng_gen) for i in range(len(X_cal))])
    test_samples = np.array([dgp.sample_generator(X_test[i], m, rng_gen) for i in range(len(X_test))])

    # ── Bias correction ──
    params_global = fit_global_affine(y_bias, bias_samples)
    cal_samples_adj = np.array([apply_affine(cal_samples[i], params_global) for i in range(len(X_cal))])
    _, c_hat_fn = fit_conformal_calibrator(cal_samples_adj, y_cal, pit_seed=pit_seed)

    params_gam = fit_x_dependent_affine_gam(
        X_bias, y_bias, bias_samples, gam_n_splines=gam_n_splines,
        b_min=None, b_max=None,
    )
    cal_samples_adj_gam = np.array([apply_affine_x(cal_samples[i], X_cal[i], params_gam) for i in range(len(X_cal))])
    _, c_hat_fn_gam = fit_conformal_calibrator(cal_samples_adj_gam, y_cal, pit_seed=pit_seed + 10)

    # ── PIT values ──
    n_test = len(y_test)
    rng_pit = np.random.default_rng(pit_seed)
    u_test_raw = np.array([randomized_pit(test_samples[i], y_test[i], rng=rng_pit) for i in range(n_test)])

    test_samples_adj = np.array([apply_affine(test_samples[i], params_global) for i in range(n_test)])
    rng_pit_bc = np.random.default_rng(pit_seed + 1)
    u_test_bc = np.array([randomized_pit(test_samples_adj[i], y_test[i], rng=rng_pit_bc) for i in range(n_test)])

    test_samples_adj_gam = np.array([apply_affine_x(test_samples[i], X_test[i], params_gam) for i in range(n_test)])
    rng_pit_bc_gam = np.random.default_rng(pit_seed + 4)
    u_test_bc_gam = np.array([randomized_pit(test_samples_adj_gam[i], y_test[i], rng=rng_pit_bc_gam) for i in range(n_test)])

    rng_pit_adj = np.random.default_rng(pit_seed + 2)
    u_adj_test = np.array([randomized_pit(test_samples_adj[i], y_test[i], rng=rng_pit_adj) for i in range(n_test)])
    u_test_cpit = c_hat_fn(u_adj_test)

    rng_pit_adj_gam = np.random.default_rng(pit_seed + 3)
    u_adj_test_gam = np.array([randomized_pit(test_samples_adj_gam[i], y_test[i], rng=rng_pit_adj_gam) for i in range(n_test)])
    u_test_cpit_gam = c_hat_fn_gam(u_adj_test_gam)

    # ── Weighted samples ──
    Yw_cpit = np.zeros((n_test, m))
    Ww_cpit = np.zeros((n_test, m))
    Yw_cpit_gam = np.zeros((n_test, m))
    Ww_cpit_gam = np.zeros((n_test, m))
    for i in range(n_test):
        Yw_cpit[i], Ww_cpit[i] = get_weighted_samples_at_x(test_samples[i].ravel(), params_global, c_hat_fn)
        yw_g, ww_g = get_weighted_samples_at_x(test_samples[i].ravel(), params_gam, c_hat_fn_gam, x=X_test[i])
        Yw_cpit_gam[i], Ww_cpit_gam[i] = yw_g, ww_g

    # ── Metrics ──
    qerr_bc_gam = quantile_calibration_error_from_sample_matrix(test_samples_adj_gam, y_test, tau_grid)
    qerr_cpit_gam = quantile_calibration_error_from_weighted_matrix(Yw_cpit_gam, Ww_cpit_gam, y_test, tau_grid)
    cvm_bc_gam = cramer_von_mises(u_test_bc_gam)
    cvm_cpit_gam = cramer_von_mises(u_test_cpit_gam)

    extra_qerr = {}
    if tau_grid_extra is not None and len(tau_grid_extra) > 0:
        extra_qerr["qerr_tail_bc_gam"] = quantile_calibration_error_from_sample_matrix(test_samples_adj_gam, y_test, tau_grid_extra)
        extra_qerr["qerr_tail_cpit_gam"] = quantile_calibration_error_from_weighted_matrix(Yw_cpit_gam, Ww_cpit_gam, y_test, tau_grid_extra)

    if mode == "all":
        qerr_raw = quantile_calibration_error_from_sample_matrix(test_samples, y_test, tau_grid)
        qerr_bc = quantile_calibration_error_from_sample_matrix(test_samples_adj, y_test, tau_grid)
        qerr_cpit = quantile_calibration_error_from_weighted_matrix(Yw_cpit, Ww_cpit, y_test, tau_grid)
        cvm_raw = cramer_von_mises(u_test_raw)
        cvm_bc = cramer_von_mises(u_test_bc)
        cvm_cpit = cramer_von_mises(u_test_cpit)
        if tau_grid_extra is not None and len(tau_grid_extra) > 0:
            extra_qerr["qerr_tail_raw"] = quantile_calibration_error_from_sample_matrix(test_samples, y_test, tau_grid_extra)
            extra_qerr["qerr_tail_bc"] = quantile_calibration_error_from_sample_matrix(test_samples_adj, y_test, tau_grid_extra)
            extra_qerr["qerr_tail_cpit"] = quantile_calibration_error_from_weighted_matrix(Yw_cpit, Ww_cpit, y_test, tau_grid_extra)

    # ── Intervals ──
    intervals_bc_gam = lambda alpha: intervals_from_sample_matrix(test_samples_adj_gam, alpha)
    if mode == "all":
        intervals_raw = lambda alpha: intervals_from_sample_matrix(test_samples, alpha)
        intervals_bc = lambda alpha: intervals_from_sample_matrix(test_samples_adj, alpha)

    def intervals_cpit(alpha: float):
        lo = quantile_from_weighted_samples_batched(Yw_cpit, Ww_cpit, alpha / 2)
        hi = quantile_from_weighted_samples_batched(Yw_cpit, Ww_cpit, 1.0 - alpha / 2)
        return np.stack([lo, hi], axis=1)

    def intervals_cpit_gam(alpha: float):
        lo = quantile_from_weighted_samples_batched(Yw_cpit_gam, Ww_cpit_gam, alpha / 2)
        hi = quantile_from_weighted_samples_batched(Yw_cpit_gam, Ww_cpit_gam, 1.0 - alpha / 2)
        return np.stack([lo, hi], axis=1)

    intervals_qr_xdep = lambda alpha: cqr_intervals_batch(cal_samples_adj_gam, y_cal, test_samples_adj_gam, alpha)[0]
    if mode == "all":
        intervals_qr_raw = lambda alpha: cqr_intervals_batch(cal_samples, y_cal, test_samples, alpha)[0]
        intervals_qr_global = lambda alpha: cqr_intervals_batch(cal_samples_adj, y_cal, test_samples_adj, alpha)[0]

    intervals_sr_raw = lambda alpha: sr_split_conformal_intervals_batch(cal_samples, y_cal, test_samples, alpha)
    intervals_sr_global = lambda alpha: sr_split_conformal_intervals_batch(cal_samples_adj, y_cal, test_samples_adj, alpha)
    intervals_sr_xdep = lambda alpha: sr_split_conformal_intervals_batch(cal_samples_adj_gam, y_cal, test_samples_adj_gam, alpha)
    intervals_crps_raw = lambda alpha: crps_split_conformal_intervals_batch(cal_samples, y_cal, test_samples, alpha)
    intervals_crps_global = lambda alpha: crps_split_conformal_intervals_batch(cal_samples_adj, y_cal, test_samples_adj, alpha)
    intervals_crps_xdep = lambda alpha: crps_split_conformal_intervals_batch(cal_samples_adj_gam, y_cal, test_samples_adj_gam, alpha)

    if mode == "all":
        method_list = [
            ("Raw", intervals_raw), ("Bias-correction (Global)", intervals_bc), ("Bias-correction (GAM)", intervals_bc_gam),
            ("CPIT (Global)", intervals_cpit), ("CPIT (GAM)", intervals_cpit_gam),
            ("Quantile residual (Raw)", intervals_qr_raw),
            ("Quantile residual (Global)", intervals_qr_global),
            ("Quantile residual (x-dependent)", intervals_qr_xdep),
            ("Scaled residual (Raw)", intervals_sr_raw),
            ("Scaled residual (Global)", intervals_sr_global),
            ("Scaled residual (x-dependent)", intervals_sr_xdep),
            ("eCRPS (Raw)", intervals_crps_raw),
            ("eCRPS (Global)", intervals_crps_global),
            ("eCRPS (x-dependent)", intervals_crps_xdep),
        ]
    else:
        method_list = [
            ("Bias-correction (GAM)", intervals_bc_gam),
            ("CPIT (GAM)", intervals_cpit_gam),
            ("Quantile residual (x-dependent)", intervals_qr_xdep),
            ("Scaled residual (x-dependent)", intervals_sr_xdep),
            ("eCRPS (x-dependent)", intervals_crps_xdep),
        ]
    results_cov, results_len = evaluate_interval_methods(method_list, y_test, alphas)

    # ── CRPS ──
    w_uni = np.ones((n_test, m)) / m
    crps_bc_gam_per = crps_from_weighted_samples(test_samples_adj_gam, w_uni, y_test)
    crps_cpit_gam_per = crps_from_weighted_samples(Yw_cpit_gam, Ww_cpit_gam, y_test)

    # ── Local diagnostics: 5 equal-mass bins of X ──
    u_local = {"BC(GAM)": u_test_bc_gam, "CPIT(GAM)": u_test_cpit_gam}
    iv_local = {"BC(GAM)": intervals_bc_gam, "CPIT(GAM)": intervals_cpit_gam}
    qerr_s_local = {"BC(GAM)": test_samples_adj_gam}
    qerr_w_local: dict[str, tuple[np.ndarray, np.ndarray]] = {"CPIT(GAM)": (Yw_cpit_gam, Ww_cpit_gam)}
    if mode == "all":
        u_local.update({"Raw": u_test_raw, "BC(Global)": u_test_bc, "CPIT(Global)": u_test_cpit})
        iv_local.update({"Raw": intervals_raw, "BC(Global)": intervals_bc, "CPIT(Global)": intervals_cpit})
        qerr_s_local.update({"Raw": test_samples, "BC(Global)": test_samples_adj})
        qerr_w_local["CPIT(Global)"] = (Yw_cpit, Ww_cpit)
    local_diag = local_diagnostics(
        X_test, y_test, u_local, iv_local, alpha_diag=0.10, n_bins=5,
        qerr_sample_by_method=qerr_s_local, qerr_weighted_by_method=qerr_w_local,
        tau_grid=tau_grid,
    )

    if mode == "all":
        crps_raw_per = crps_from_weighted_samples(test_samples, w_uni, y_test)
        crps_bc_per = crps_from_weighted_samples(test_samples_adj, w_uni, y_test)
        crps_cpit_per = crps_from_weighted_samples(Yw_cpit, Ww_cpit, y_test)

        result = {
            "cvm_raw": cvm_raw, "cvm_bc": cvm_bc, "cvm_bc_gam": cvm_bc_gam,
            "cvm_cpit": cvm_cpit, "cvm_cpit_gam": cvm_cpit_gam,
            "qerr_raw": qerr_raw, "qerr_bc": qerr_bc, "qerr_bc_gam": qerr_bc_gam,
            "qerr_cpit": qerr_cpit, "qerr_cpit_gam": qerr_cpit_gam,
            "crps_raw": float(np.mean(crps_raw_per)), "crps_bc": float(np.mean(crps_bc_per)),
            "crps_bc_gam": float(np.mean(crps_bc_gam_per)),
            "crps_cpit": float(np.mean(crps_cpit_per)), "crps_cpit_gam": float(np.mean(crps_cpit_gam_per)),
            "results_cov": results_cov,
            "results_len": results_len,
            "local_diag": local_diag,
        }
        result.update(extra_qerr)
        return result

    result = {
        "cvm_bc_gam": cvm_bc_gam, "cvm_cpit_gam": cvm_cpit_gam,
        "qerr_bc_gam": qerr_bc_gam, "qerr_cpit_gam": qerr_cpit_gam,
        "crps_bc_gam": float(np.mean(crps_bc_gam_per)),
        "crps_cpit_gam": float(np.mean(crps_cpit_gam_per)),
        "results_cov": results_cov,
        "results_len": results_len,
        "local_diag": local_diag,
    }
    result.update(extra_qerr)
    return result


# ── Aggregation ──────────────────────────────────────────────────────────────

ALL_METHOD_NAMES = [
    "Raw", "Bias-correction (Global)", "Bias-correction (GAM)",
    "CPIT (Global)", "CPIT (GAM)",
    "Quantile residual (Raw)", "Quantile residual (Global)", "Quantile residual (x-dependent)",
    "Scaled residual (Raw)", "Scaled residual (Global)", "Scaled residual (x-dependent)",
    "eCRPS (Raw)", "eCRPS (Global)", "eCRPS (x-dependent)",
]

GAM_ONLY_METHOD_NAMES = [
    "Bias-correction (GAM)", "CPIT (GAM)",
    "Quantile residual (x-dependent)", "Scaled residual (x-dependent)", "eCRPS (x-dependent)",
]


def aggregate_replicates(
    replicate_results: list[dict],
    *,
    mode: str = "all",
) -> pd.DataFrame:
    """Aggregate replicate dicts into summary DataFrame (mean ± SE)."""
    n_rep = len(replicate_results)

    if mode == "all":
        method_names = ALL_METHOD_NAMES
        cvm_keys = ["cvm_raw", "cvm_bc", "cvm_bc_gam", "cvm_cpit", "cvm_cpit_gam"]
        qerr_keys = ["qerr_raw", "qerr_bc", "qerr_bc_gam", "qerr_cpit", "qerr_cpit_gam"]
        crps_keys = ["crps_raw", "crps_bc", "crps_bc_gam", "crps_cpit", "crps_cpit_gam"]
    else:
        method_names = GAM_ONLY_METHOD_NAMES
        cvm_keys = ["cvm_bc_gam", "cvm_cpit_gam"]
        qerr_keys = ["qerr_bc_gam", "qerr_cpit_gam"]
        crps_keys = ["crps_bc_gam", "crps_cpit_gam"]

    cvm_arr = np.array([[rep[k] for k in cvm_keys] for rep in replicate_results])
    qerr_arr = np.array([[rep[k] for k in qerr_keys] for rep in replicate_results])
    crps_arr = np.array([[rep[k] for k in crps_keys] for rep in replicate_results])

    def mean_se_strings(arr_1d: np.ndarray) -> str:
        if n_rep <= 1:
            return f"{float(arr_1d[0]):.6f}"
        mean = float(np.nanmean(arr_1d))
        se = float(np.nanstd(arr_1d, ddof=1) / np.sqrt(n_rep))
        return f"{mean:.6f} ± {se:.6f}"

    cvm_vals, qerr_vals, crps_vals = [], [], []
    for j in range(len(cvm_keys)):
        cvm_vals.append(mean_se_strings(cvm_arr[:, j]))
        qerr_vals.append(mean_se_strings(qerr_arr[:, j]))
        crps_vals.append(mean_se_strings(crps_arr[:, j]))
    n_interval_only = len(method_names) - len(cvm_keys)
    cvm_vals += [float("nan")] * n_interval_only
    qerr_vals += [float("nan")] * n_interval_only
    crps_vals += [float("nan")] * n_interval_only

    cov_cols: dict[str, list[str]] = {}
    len_cols: dict[str, list[str]] = {}
    for a in ALPHAS:
        cov_per, len_per = [], []
        for name in method_names:
            per_rep_cov = np.array([rep["results_cov"][name][a] for rep in replicate_results], dtype=float)
            per_rep_len = np.array([rep["results_len"][name][a] for rep in replicate_results], dtype=float)
            if n_rep <= 1:
                cov_per.append(f"{float(per_rep_cov[0]):.6f}")
                len_per.append(f"{float(per_rep_len[0]):.6f}")
            else:
                cov_per.append(f"{np.mean(per_rep_cov):.6f} ± {np.std(per_rep_cov, ddof=1)/np.sqrt(n_rep):.6f}")
                len_per.append(f"{np.mean(per_rep_len):.6f} ± {np.std(per_rep_len, ddof=1)/np.sqrt(n_rep):.6f}")
        cov_cols[f"Cov(α={a})"] = cov_per
        len_cols[f"Len(α={a})"] = len_per

    summary = pd.DataFrame({"CvM": cvm_vals, "QErr": qerr_vals, "Mean CRPS": crps_vals}, index=method_names)
    result = pd.concat([summary, pd.DataFrame(cov_cols, index=method_names), pd.DataFrame(len_cols, index=method_names)], axis=1)

    # Append upper-tail QErr columns if present (Design 3).
    tail_key_to_method = {
        "qerr_tail_raw": "Raw",
        "qerr_tail_bc": "Bias-correction (Global)",
        "qerr_tail_bc_gam": "Bias-correction (GAM)",
        "qerr_tail_cpit": "CPIT (Global)",
        "qerr_tail_cpit_gam": "CPIT (GAM)",
    }
    tail_keys = [k for k in replicate_results[0] if k.startswith("qerr_tail_")]
    if tail_keys:
        col_vals = [float("nan")] * len(method_names)
        for k in tail_keys:
            method = tail_key_to_method.get(k)
            if method and method in method_names:
                vals = np.array([rep[k] for rep in replicate_results], dtype=float)
                idx = method_names.index(method)
                col_vals[idx] = mean_se_strings(vals)
        result["Tail QErr"] = col_vals

    return result


def aggregate_local_diagnostics(
    replicate_results: list[dict],
    n_bins: int = 5,
) -> pd.DataFrame:
    """Aggregate local diagnostics across replicates."""
    n_rep = len(replicate_results)
    if n_rep == 0 or "local_diag" not in replicate_results[0]:
        return pd.DataFrame()

    sample_bin = replicate_results[0]["local_diag"][0]
    _excluded_keys = {"bin", "n", "x_min", "x_max"}
    metric_keys = [k for k in sample_bin if k not in _excluded_keys]

    rows = []
    for b in range(n_bins):
        row = {"bin": b + 1}
        n_vals = np.array(
            [rep["local_diag"][b].get("n", float("nan")) for rep in replicate_results
             if len(rep["local_diag"]) > b], dtype=float,
        )
        if len(n_vals) > 0 and not np.all(np.isnan(n_vals)):
            row["n_mean"] = int(round(np.nanmean(n_vals)))
            row["n_min"] = int(np.nanmin(n_vals))
            row["n_max"] = int(np.nanmax(n_vals))
        xmin_vals = np.array(
            [rep["local_diag"][b].get("x_min", float("nan")) for rep in replicate_results
             if len(rep["local_diag"]) > b], dtype=float,
        )
        xmax_vals = np.array(
            [rep["local_diag"][b].get("x_max", float("nan")) for rep in replicate_results
             if len(rep["local_diag"]) > b], dtype=float,
        )
        if not np.all(np.isnan(xmin_vals)):
            row["x_min"] = float(np.nanmean(xmin_vals))
            row["x_max"] = float(np.nanmean(xmax_vals))
        for k in metric_keys:
            vals = np.array(
                [rep["local_diag"][b][k] for rep in replicate_results
                 if len(rep["local_diag"]) > b], dtype=float,
            )
            if n_rep <= 1:
                row[k] = f"{float(vals[0]):.4f}"
            else:
                row[k] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1)/np.sqrt(len(vals)):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)
