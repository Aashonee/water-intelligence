"""Independent checks: simulate a tower with an explicit, mass-conserving solids balance,
feed its sensor log to the engine, and compare against the true makeup. These don't
reuse any engine formula, so they catch modelling errors the unit tests can't."""

import numpy as np
import pandas as pd
import pytest

from water_intelligence import chemistry as ch
from water_intelligence import data, pipeline, savings

V, E, W, CM = 600.0, 25.0, 3.0, 200.0  # basin m³, evaporation m³/h, drift m³/h, makeup TDS


def _log(conc):
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(conc), freq="h"),
        "TDS_circ": conc, "TDS_makeup": CM, "pH": 7.8, "temp_C": 32.0, "flow_rate": 1.0,
        "calcium_hardness": 150.0, "alkalinity": 120.0,
    })


def _engine_makeup(conc):
    df, _ = data.clean(_log(conc))
    hb = pd.DataFrame({"evap_m3h": E, "drift_m3h": W, "heat_kw": 0.0}, index=df.index)
    d = savings.daily_rollup(df, hb, 3.0)
    return d, d["modelled_makeup_m3"].sum(), d["hours"].sum()


def test_batch_blowdown_matches_mass_balance_simulation():
    c, conc, true = CM * 1.4, [], 0.0
    for _ in range(24 * 40):
        mu = E + W
        c = (c * V + mu * CM - W * c) / V
        if c >= 3.0 * CM:  # dump, then refill with makeup, back to CoC 1.4
            d = V * (c - 1.4 * CM) / (c - CM)
            mu += d
            c = ((V - d) * c + d * CM) / V
        true += mu
        conc.append(c)
    daily, model, hours = _engine_makeup(conc)
    assert daily.attrs["blowdown_mode"].startswith("batch")
    assert model == pytest.approx(true * hours / len(conc), rel=0.02)


def test_continuous_bleed_matches_mass_balance_simulation():
    c, conc, true, bleeding = 3.0 * CM, [], 0.0, False
    for _ in range(24 * 40):
        bleeding = c > 3.05 * CM or (bleeding and c > 2.95 * CM)
        b = 20.0 if bleeding else 0.0
        mu = E + W + b
        c = (c * V + mu * CM - (W + b) * c) / V
        true += mu
        conc.append(c)
    daily, model, hours = _engine_makeup(conc)
    assert daily.attrs["blowdown_mode"].startswith("continuous")
    assert model == pytest.approx(true * hours / len(conc), rel=0.01)


def test_envelope_matches_brute_force_search(sample_csv):
    a = pipeline.run(sample_csv, pipeline.Settings(envelope=ch.EnvelopeSettings(lsi_limit=1.8, compliance=0.9)))
    df = a.df[a.df["coc"].notna()]

    def share_ok(r):
        lsi = ch.lsi(df.pH + np.log10(r), df.TDS_circ * r, 50.0, df.calcium_hardness * r, df.alkalinity * r)
        return np.average(lsi <= 1.8, weights=df.dt_h)

    grid = np.linspace(1.0, 3.0, 4001)
    r_best = max(r for r in grid if share_ok(r) >= 0.9)
    assert a.env.limits["LSI at heat-exchanger skin"] == pytest.approx(a.current_coc * r_best, rel=1e-3)
