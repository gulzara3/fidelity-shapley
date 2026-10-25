import numpy as np
import pandas as pd

from fidelity_shapley.config import CMAPSS_SENSORS, SKAB_CHANNELS
from fidelity_shapley.data.pipeline import cmapss_windows, redundancy_groups, safe_std, skab_windows


def test_union_find_groups_are_connected_components():
    rho = np.eye(5)
    rho[0, 1] = rho[1, 0] = 0.95
    rho[1, 2] = rho[2, 1] = 0.93       # 0–1–2 chained
    rho[3, 4] = rho[4, 3] = 0.5        # below threshold
    assert redundancy_groups(rho, 0.9) == {0: [0, 1, 2]}


def test_safe_std_maps_numerically_constant_channels_to_one():
    df = pd.DataFrame({"const": [14.62] * 1000, "var": np.arange(1000.0)})
    sd = safe_std(df, 1e-6)
    assert sd["const"] == 1.0
    assert sd["var"] > 1
    z = (df - df.mean()) / sd
    assert np.allclose(z["const"], 0.0, atol=1e-9)   # never ±1 from rounding noise


def test_skab_windows_shapes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 8)), columns=SKAB_CHANNELS)
    df["anomaly"] = 0.0
    X, T, y = skab_windows(df, df[SKAB_CHANNELS].mean(), df[SKAB_CHANNELS].std(), 32)
    assert X.shape == (100 - 32 - 1, 32, 8) and T.shape == (67, 8) and y.shape == (67,)
    np.testing.assert_allclose(T[0], X[1, -1])        # target is the next step


def test_cmapss_windows_train_and_test():
    rows = []
    for unit, n in ((1, 40), (2, 20)):
        for c in range(n):
            rows.append([unit, c + 1] + list(np.full(21, float(c))))
    df = pd.DataFrame(rows, columns=["unit", "cycle"] + CMAPSS_SENSORS)
    mu, sd = df[CMAPSS_SENSORS].mean(), safe_std(df[CMAPSS_SENSORS], 1e-6)
    X, y = cmapss_windows(df, mu, sd, 30, 125.0, True)
    assert X.shape == (10, 30, 21) and np.isclose(y[0], 10 / 125.0)
    Xt, _ = cmapss_windows(df, mu, sd, 30, 125.0, False)
    assert Xt.shape == (2, 30, 21) and np.all(Xt[1, :10] == 0)   # short engine zero-padded
