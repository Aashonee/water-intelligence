"""Properties, LSI, envelope and water balance — includes regressions for the review's
section A/B findings."""

import numpy as np
import pandas as pd
import pytest

from water_intelligence import chemistry as ch
from water_intelligence import data
from water_intelligence import water_balance as wb
from water_intelligence.properties import latent_heat_kj_per_kg, water_density_kg_per_m3


# --- properties ---------------------------------------------------------------
def test_latent_heat_interpolates_not_nearest():
    assert latent_heat_kj_per_kg(35) == pytest.approx(2417.9)
    mid = latent_heat_kj_per_kg(32.5)
    assert 2417.9 < mid < 2429.8 and mid == pytest.approx((2417.9 + 2429.8) / 2)
    assert latent_heat_kj_per_kg(60) == pytest.approx(2357.7)  # no 20-50 °C clamp


def test_density_depends_on_temperature():
    assert water_density_kg_per_m3(4) == pytest.approx(999.97, abs=0.01)
    assert water_density_kg_per_m3(35) == pytest.approx(994.0, abs=0.1)


# --- LSI ----------------------------------------------------------------------
def test_lsi_uses_273_15_and_rejects_nonpositive():
    v = ch.lsi(7.8, 350, 34.5, 200, 150)
    b = -13.12 * np.log10(34.5 + 273.15) + 34.55
    phs = 9.3 + (np.log10(350) - 1) / 10 + b - (np.log10(200) - 0.4) - np.log10(150)
    assert v == pytest.approx(7.8 - phs)
    with pytest.raises(ValueError):
        ch.lsi(7.8, 0, 30, 200, 150)


@pytest.mark.parametrize("model,dph", [("fixed", 0.0), ("calibrated-equilibrium", 1.0)])
def test_concentration_slope_matches_direct_recalculation(model, dph):
    """Concentrating Ca, alkalinity and TDS by r (and pH by dph·log r) must match the
    analytic slope used by the envelope solve."""
    pH, tds, t, ca, alk, r = 7.8, 350.0, 50.0, 200.0, 150.0, 1.7
    l0 = ch.lsi(pH, tds, t, ca, alk)
    direct = ch.lsi(pH + dph * np.log10(r), tds * r, t, ca * r, alk * r)
    assert ch.lsi_after_concentrating(l0, r, model) == pytest.approx(direct)
    r_back = ch.ratio_to_reach_lsi(l0, float(direct), model)
    assert r_back == pytest.approx(r)


def test_reviewer_reference_max_coc_with_scaled_chemistry(sample_csv):
    """Review A.2: scaling Ca/alk/TDS with pH held, LSI 0.5 at basin → CoC ≈ 1.71 < 1.74."""
    d = sample_csv
    coc = d.TDS_circ.mean() / d.TDS_makeup.mean()
    l0 = ch.lsi(d.pH.mean(), d.TDS_circ.mean(), d.temp_C.mean(), d.calcium_hardness.mean(), d.alkalinity.mean())
    max_coc = coc * ch.ratio_to_reach_lsi(l0, 0.5, "fixed")
    assert max_coc == pytest.approx(1.71, abs=0.01)
    assert max_coc < coc


def test_max_safe_coc_is_not_the_current_coc(sample_csv):
    """Review A.1: v0.1 returned max safe CoC == current CoC (to 1e-14)."""
    df, _ = data.clean(sample_csv)
    lsi_df = ch.hourly_lsi(df, 50.0)
    for limit in (0.5, 1.8, 2.5):
        env = ch.envelope(df, lsi_df, ch.EnvelopeSettings(lsi_limit=limit))
        assert abs(env.max_safe_coc - df["coc"].mean()) > 0.05


def test_skin_lsi_higher_than_basin(sample_csv):
    df, _ = data.clean(sample_csv)
    l = ch.hourly_lsi(df, 50.0)
    assert l["lsi_skin"].mean() == pytest.approx(0.80, abs=0.02)
    assert l["lsi_basin"].mean() == pytest.approx(0.52, abs=0.02)
    assert (l["lsi_basin"] > 0.5).sum() in range(360, 368)  # review: 363 hours


def test_envelope_reports_binding_and_unchecked_constraints():
    df, _ = data.clean(pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=48, freq="h"),
        "TDS_circ": 400.0, "TDS_makeup": 200.0, "pH": 7.5, "temp_C": 32.0, "flow_rate": 3000.0,
        "calcium_hardness": 120.0, "alkalinity": 100.0, "silica_makeup": 50.0,
    }))
    env = ch.envelope(df, ch.hourly_lsi(df, 50), ch.EnvelopeSettings(lsi_limit=2.5, silica_limit=150))
    assert env.limits["Silica"] == pytest.approx(3.0)
    assert env.binding == "Silica" and env.max_safe_coc == pytest.approx(3.0)
    assert any("chloride" in s for s in env.not_checked)
    assert not any(s.startswith("silica") for s in env.not_checked)


def test_lsi_status_bands_follow_limit():
    assert ch.lsi_status(0.6, 0.5)[1] == "error"
    assert ch.lsi_status(0.6, 1.8)[1] == "success"
    assert ch.lsi_status(-0.8, 1.8)[0] == "Corrosive"


# --- water balance ------------------------------------------------------------
def test_drift_is_not_evaporation():
    """Review A.3: B = E/(CoC-1) - W. Reviewer numbers: 28.5 vs old 51.7 (their units)."""
    flow = 3300.0
    e = 0.00866 * flow           # 0.866 % of circulation, as on the sample data
    w = 0.003 * flow             # v0.1's 0.3 % drift
    coc = 1.743345
    new = wb.blowdown_m3h(e, w, coc)
    old = (e + w) / (coc - 1)
    assert new == pytest.approx(28.5, abs=0.3)
    assert old == pytest.approx(51.8, abs=0.3)
    m = wb.makeup_m3h(e, w, coc)
    assert m == pytest.approx(e * coc / (coc - 1))
    assert m == pytest.approx(e + new + w)


def test_makeup_when_drift_exceeds_blowdown_requirement():
    e, w = 10.0, 3.0
    cap = wb.drift_limited_coc(e, w)
    assert cap == pytest.approx(1 + 10 / 3)
    assert wb.blowdown_m3h(e, w, cap + 1) < 0
    assert wb.makeup_m3h(e, w, cap + 1) == pytest.approx(e + w)


def test_coc_must_exceed_one():
    with pytest.raises(ValueError):
        wb.makeup_m3h(10, 1, 1.0)


def test_evaporation_about_0_86_percent_of_circulation(sample_csv):
    df, _ = data.clean(sample_csv)
    hb = wb.hourly_balance(df, wb.BalanceSettings(delta_t_c=5.0))
    assert (hb["evap_m3h"] / df["flow_m3h"]).mean() == pytest.approx(0.0087, abs=0.0003)
    assert hb["heat_kw"].mean() > 10_000  # an industrial tower, not 19 kW


def test_measured_delta_t_is_used_per_row():
    raw = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4, freq="h"),
        "TDS_circ": 400.0, "TDS_makeup": 200.0, "pH": 7.5, "temp_C": 30.0, "flow_rate": 1000.0,
        "calcium_hardness": 120.0, "alkalinity": 100.0,
        "T_hot": [35.0, 36.0, 37.0, 38.0], "T_cold": 30.0,
    })
    df, _ = data.clean(raw)
    hb = wb.hourly_balance(df, wb.BalanceSettings(delta_t_c=99))
    assert list(hb["delta_t"]) == [5, 6, 7, 8]
    assert hb["delta_t_source"].iat[0] == "measured"
