import numpy as np

from cpit.data_splitter import apply_split, split_four_way


def test_split_four_way_reproducible():
    s1 = split_four_way(1000, seed=42)
    s2 = split_four_way(1000, seed=42)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a, b)


def test_split_four_way_disjoint():
    s = split_four_way(1000, seed=0)
    all_idx = np.concatenate([s.train, s.bias, s.calibration, s.test])
    np.testing.assert_array_equal(np.sort(all_idx), np.arange(1000))


def test_split_four_way_fractions():
    s = split_four_way(
        1000, train_frac=0.4, bias_frac=0.2, cal_frac=0.2, test_frac=0.2, seed=0
    )
    assert len(s.train) == 400
    assert len(s.bias) == 200
    assert len(s.calibration) == 200
    assert len(s.test) == 200


def test_apply_split():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    s = split_four_way(
        10, train_frac=0.3, bias_frac=0.2, cal_frac=0.2, test_frac=0.3, seed=0
    )
    (Xt, yt), (Xb, yb), (Xc, yc), (Xte, yte) = apply_split(X, y, s)
    assert Xt.shape[0] == yt.shape[0]
    assert len(Xte) + len(Xt) + len(Xb) + len(Xc) == 10
