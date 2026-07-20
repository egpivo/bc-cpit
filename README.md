# bc-cpit

[![CI](https://github.com/egpivo/cpit/actions/workflows/ci.yml/badge.svg)](https://github.com/egpivo/cpit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/egpivo/cpit/graph/badge.svg?token=SNoFZfquk5)](https://codecov.io/gh/egpivo/cpit)
[![PyPI](https://img.shields.io/pypi/v/bc-cpit)](https://pypi.org/project/bc-cpit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Bias-corrected conformal PIT calibration for sample-based predictive distributions.

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

## Layout

```
cpit/          Python package
  bc/          bias correction (fit, apply)
  baselines/   competing methods
  evaluation/  metrics, PIT histograms, diagnostics
  calibrator.py / pit.py / inference.py / weighted_samples.py / pipeline.py

examples/      orchestration scripts (call cpit)
  simulation/  §5 designs 1–3
  real_data/   §6 WB2 Taiwan + Europe
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
