# bc-cpit

[![CI](https://github.com/egpivo/cpit/actions/workflows/ci.yml/badge.svg)](https://github.com/egpivo/cpit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/egpivo/bc-cpit/graph/badge.svg?token=SNoFZfquk5)](https://codecov.io/gh/egpivo/bc-cpit)
[![PyPI](https://img.shields.io/pypi/v/bc-cpit)](https://pypi.org/project/bc-cpit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Conditional generators (diffusion, ensembles, simulators) often return samples
`Y(1),…,Y(m) ~ P̂(·|x)` with no tractable likelihood. Those sample clouds can be
biased and poorly calibrated, and standard conformal tools mainly return a
fixed-level set—not a predictive CDF you can query for arbitrary quantiles,
exceedance probabilities, or tail losses.
**bc-cpit** is a split-sample post-processing pipeline:
1. **Bias correction** — fit an affine location–scale map on a held-out bias
   split (global, or x-dependent via GAM).
2. **Conformal PIT calibration** — calibrate randomized PITs on a calibration
   split; represent the corrected law as a **weighted empirical distribution**
   on the generator order statistics.
From those weights you get threshold-coherent probabilities, quantiles, central /
highest-density intervals, and calibrated resamples. An optional PIT-centrality
wrapper adds nested intervals with finite-sample marginal coverage under
exchangeability.

This package does **not** retrain the generator; it recalibrates whatever samples
you already have.

## Install

```bash
pip install bc-cpit
```

Development:

```bash
git clone https://github.com/egpivo/bc-cpit.git
cd cpit
pip install -e ".[dev]"
```

## Quick start

```python
from cpit.bc import fit_global_affine, apply_affine
from cpit import fit_conformal_calibrator, get_weighted_samples_at_x
from cpit import quantile_from_weighted_samples, central_interval_from_weighted_samples

# 1. Bias correction
params = fit_global_affine(y_bias, y_bias_samples)

# 2. Conformal calibration
cal_samples_adj = [apply_affine(s, params) for s in cal_samples]
_, c_hat = fit_conformal_calibrator(cal_samples_adj, y_cal)

# 3. Inference
y_pts, weights = get_weighted_samples_at_x(y_test_samples, params, c_hat)
q80 = quantile_from_weighted_samples(y_pts, weights, 0.80)
lo, hi = central_interval_from_weighted_samples(y_pts, weights, alpha=0.10)
```

x-dependent (GAM) correction:

```python
from cpit.bc import fit_x_dependent_affine_gam

params_gam = fit_x_dependent_affine_gam(x_bias, y_bias, y_bias_samples)
y_pts, weights = get_weighted_samples_at_x(y_test_samples, params_gam, c_hat, x=x_test)
```

High-level pipeline:

```python
from cpit.pipeline import run_pipeline, predict_interval

state = run_pipeline(X, y, generator_fn)
lo, hi = predict_interval(state, X_test, y_samples_test, alpha=0.10)
```

## Reproduce paper results

```bash
make test          # unit tests
make run-sim       # Designs 1/2/3 simulations (§5)
make pit-figures   # PIT histogram figures
make run-wb2       # WB2 application: Taiwan + Europe (§6)
```


## Citation

The paper is currently being submitted to arXiv; the entry below will be updated with the final arXiv ID once it is live.

If you use this package, please cite:

> Wang, W.-T., Tzeng, S., Fan, Y.-T., & Huang, H.-C. (2026). Calibrated Predictive Distributions from Sample-Based Generators. *arXiv preprint arXiv:XXXX.XXXXX*.

```bibtex
@article{wang2026calibrated,
  title   = {Calibrated Predictive Distributions from Sample-Based Generators},
  author  = {Wang, Wen-Ting and Tzeng, ShengLi and Fan, Yu-Ting and Huang, Hsin-Cheng},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
