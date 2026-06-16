# cpit

Bias correction → conformal PIT calibration → calibrated predictive distributions, with baselines and evaluation. Implements the method in the accompanying paper.

## Method overview

The pipeline has three stages:

1. **Bias correction (§2.1):** Fit an affine transform $\hat{y}_{\text{adj}} = \hat{a}(x) + \hat{b}(x)\, \hat{y}$ that matches the generator mean and spread to the truth. Two variants:
   - *Global* (`AffineParams`): scalar $(\hat{a}_0, \hat{b}_0)$ via quasi-log-likelihood (Eq. 3).
   - *x-dependent GAM* (`XAffineParamsGAM`): alternating LinearGAM + GammaGAM (Eq. 6–10).

2. **Conformal PIT calibration (§2.2):** Compute randomized PITs $u_i$ on a held-out calibration split from the bias-corrected samples; build $\hat{C}$ as the empirical CDF of $u_i$ (Eq. 13–14).

3. **Calibrated inference (§2.3–2.5):** The calibrated CDF is $\tilde{F}(y|x) = \hat{C}(F_{\text{adj}}(y|x))$. Weighted samples $w_j = \hat{C}(j/m) - \hat{C}((j-1)/m)$ (Eq. 17) support quantile, central interval, HDR, and calibrated resampling.

## Install

```bash
pip install cpit
```

For development (editable install):

```bash
git clone https://github.com/egpivo/cpit.git
cd cpit
pip install -e ".[dev]"
# or
make install
```

Optional environments:

```bash
make venv          # create .venv, then: .venv/bin/pip install -e .[dev]
make conda-env     # create conda env 'cpit', then: conda activate cpit
```

## Quick start

```python
import numpy as np
from cpit.bc import fit_global_affine, apply_affine
from cpit import fit_conformal_calibrator, randomized_pit, get_weighted_samples_at_x
from cpit import quantile_from_weighted_samples, central_interval_from_weighted_samples

# 1. Bias correction on held-out bias split
params = fit_global_affine(y_bias_observed, y_bias_samples)   # -> AffineParams(a, b)

# 2. Calibration: PIT on bias-corrected calibration split
cal_samples_adj = [apply_affine(s, params) for s in cal_samples]
u_cal, c_hat = fit_conformal_calibrator(cal_samples_adj, y_cal)  # -> callable Ĉ(t)

# 3. Inference at test point
y_pts, weights = get_weighted_samples_at_x(y_samples_raw, params, c_hat)
q80 = quantile_from_weighted_samples(y_pts, weights, 0.80)
lo, hi = central_interval_from_weighted_samples(y_pts, weights, alpha=0.10)
```

For x-dependent correction:

```python
from cpit.bc import fit_x_dependent_affine_gam, apply_affine_x

params_gam = fit_x_dependent_affine_gam(x_bias, y_bias_observed, y_bias_samples)
y_pts, weights = get_weighted_samples_at_x(y_samples_raw, params_gam, c_hat, x=x_test)
```

Or use the high-level pipeline helper:

```python
from cpit.pipeline import run_pipeline, predict_interval

state = run_pipeline(X, y, generator_fn)
lo, hi = predict_interval(state, X_test, y_samples_test, alpha=0.10)
```

## Reproduce paper results

```bash
make test          # unit tests
make run-sim       # Designs 1/2/3 simulations (§5), 100 replicates each
make pit-figures   # PIT histogram figures for all designs
make run-wb2       # WB2 application: Taiwan + Europe × mean + p95 (§6)
```

## Package layout

```
cpit/              Python package (pip install cpit)
  bc/              Bias correction: params, fit, apply
  baselines/       Competing methods: quantile regression, CRPS, score residual
  evaluation/      Metrics, PIT histograms, local diagnostics
  calibrator.py    Conformal calibrator Ĉ
  pit.py           Randomized PIT
  weighted_samples.py
  inference.py     Quantile, central interval, HDR, resampling
  pipeline.py      High-level run_pipeline / predict_interval
  data_splitter.py

examples/          Orchestration scripts (call cpit, not part of the package)
  simulation/      §5: Designs 1/2/3, dgp, runner, CLI run_design*.py
  real_data/       §6: WB2 Taiwan + Europe download + experiment scripts

tests/             Unit + integration tests
envs/              Conda and pip environment files  →  envs/README.md
```

## Paper → code map

| Paper | Code |
|-------|------|
| Eq. (3): global affine loss | `cpit/bc/fit.py` → `fit_global_affine` |
| Eq. (4): apply global correction | `cpit/bc/apply.py` → `apply_affine` |
| Eq. (6)–(10): x-dep GAM alternating | `cpit/bc/fit.py` → `fit_x_dependent_affine_gam` |
| Eq. (5), (10): apply x-dep correction | `cpit/bc/apply.py` → `apply_affine_x` |
| Eq. (13)–(14): randomized PIT + Ĉ | `cpit/pit.py`, `cpit/calibrator.py` |
| Eq. (17): weighted samples | `cpit/weighted_samples.py` → `get_weighted_samples_at_x` |
| Eq. (29)–(30): smooth CDF / quantiles | `cpit/inference.py` → `quantile_from_weighted_samples(smooth=True)` |

## Dependencies

Core: `numpy`, `scikit-learn`, `pygam`
Evaluation: `matplotlib`, `pandas` (optional, for plotting)
Dev: `pytest`

See `envs/conda/environment_cpit.yml` or `pyproject.toml` for the full list.

## License

MIT — see [LICENSE](LICENSE).
