import numpy as np

from cpit.baselines.score_baselines import (
    crps_split_conformal_interval,
    sr_split_conformal_interval,
)


def _toy_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_cal, m = 80, 200
    mu = rng.normal(0.0, 1.0, size=n_cal)
    y_cal_s = mu[:, None] + rng.normal(0.0, 1.0, size=(n_cal, m))
    y_cal_o = mu + rng.normal(0.0, 1.0, size=n_cal)
    y_test_s = rng.normal(0.5, 1.2, size=m)
    return y_cal_s, y_cal_o, y_test_s


def test_sr_interval_shape_and_order():
    y_cal_s, y_cal_o, y_test_s = _toy_data()
    low, high = sr_split_conformal_interval(y_cal_s, y_cal_o, y_test_s, alpha=0.1)
    assert np.isfinite(low)
    assert np.isfinite(high)
    assert low <= high


def test_crps_interval_shape_and_order():
    y_cal_s, y_cal_o, y_test_s = _toy_data()
    low, high = crps_split_conformal_interval(
        y_cal_s, y_cal_o, y_test_s, alpha=0.1, n_grid=300
    )
    assert np.isfinite(low)
    assert np.isfinite(high)
    assert low <= high
