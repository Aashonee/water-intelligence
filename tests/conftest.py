import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_plant(days=30, coc=2.0, coc_after=None, change_day=15, flow=3000.0, flow_after=None,
               noise=0.0, seed=0, start="2024-01-01", **extra):
    """Hourly synthetic plant with continuous blowdown at a steady CoC.

    Chemistry is chosen so skin LSI stays moderate. ``coc_after``/``flow_after`` apply from
    ``change_day`` (0-based) onward.
    """
    rng = np.random.default_rng(seed)
    n = days * 24
    t = pd.date_range(start, periods=n, freq="h")
    day = np.arange(n) // 24
    c = np.where((coc_after is not None) & (day >= change_day), coc_after or coc, coc).astype(float)
    f = np.where((flow_after is not None) & (day >= change_day), flow_after or flow, flow).astype(float)
    def eps():
        return 1 + noise * rng.standard_normal(n)

    tds_m = 200.0
    df = pd.DataFrame({
        "date": t,
        "TDS_circ": tds_m * c * eps(),
        "TDS_makeup": tds_m,
        "pH": 7.6 + 0.5 * noise * rng.standard_normal(n),
        "temp_C": 32.0 + 0.5 * np.sin(np.arange(n) / 24 * 2 * np.pi),
        "flow_rate": f * eps(),
        "calcium_hardness": 60.0 * c * eps(),
        "alkalinity": 50.0 * c * eps(),
    })
    for k, v in extra.items():
        df[k] = v
    return df


@pytest.fixture
def sample_csv():
    return pd.read_csv(ROOT / "sample_cooling_tower_data.csv")
