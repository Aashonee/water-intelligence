"""Data-quality layer, M&V and units — review sections A (savings, baseline, units) and C.4."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_plant

from water_intelligence import chemistry as ch
from water_intelligence import data, pipeline, savings
from water_intelligence import water_balance as wb


# --- data quality -------------------------------------------------------------
def test_missing_columns_raise_clear_error():
    with pytest.raises(data.DataError, match="Missing required columns"):
        data.clean(pd.DataFrame({"date": ["2024-01-01"]}))


def test_nan_text_unsorted_duplicates_and_bad_ranges_are_handled():
    raw = make_plant(days=3).astype({"pH": object})
    raw.loc[5, "pH"] = "n/a"
    raw.loc[6, "TDS_circ"] = np.nan
    raw.loc[7, "pH"] = 14.2
    raw = pd.concat([raw, raw.iloc[[10]]])          # duplicate timestamp
    raw = raw.sample(frac=1, random_state=1)        # shuffle
    df, rep = data.clean(raw)
    assert df["date"].is_monotonic_increasing
    assert rep.duplicates == 1
    assert rep.dropped["missing / non-numeric required value"] == 2
    assert rep.dropped["value outside physical range"] == 1
    assert rep.rows_used == 72 - 3


def test_gaps_are_not_counted_as_hours():
    raw = make_plant(days=2)
    raw = raw.drop(index=range(10, 20))  # 10-hour gap
    df, rep = data.clean(raw)
    assert rep.gaps == 1 and rep.longest_gap_h == pytest.approx(11)
    assert rep.hours_covered == pytest.approx(38)


def test_conductivity_preferred_as_tracer():
    raw = make_plant(days=2, cond_circ=900.0, cond_makeup=300.0)
    df, rep = data.clean(raw)
    assert rep.tracer == "conductivity"
    assert df["coc"].median() == pytest.approx(3.0)


def test_sensor_spike_excluded_from_coc():
    raw = make_plant(days=2, coc=2.0)
    raw.loc[20, "TDS_circ"] = 4000.0
    df, rep = data.clean(raw)
    assert rep.coc_outliers == 1 and np.isnan(df.loc[20, "coc"])


def test_flow_units_and_heat_load_warning():
    raw = make_plant(days=2, flow=3300.0)
    df, _ = data.clean(raw, "kg/h (≈ L/h)")
    hb = wb.hourly_balance(df, wb.BalanceSettings())
    assert data.heat_load_check(hb["heat_kw"].median()) is not None  # ~19 kW → wrong unit
    df2, _ = data.clean(raw, "m³/h")
    assert data.heat_load_check(wb.hourly_balance(df2, wb.BalanceSettings())["heat_kw"].median()) is None


# --- M&V ----------------------------------------------------------------------
def _run(raw, **kw):
    s = pipeline.Settings(envelope=ch.EnvelopeSettings(lsi_limit=2.5), **kw)
    return pipeline.run(raw, s)


def test_baseline_end_date_is_inclusive(sample_csv):
    df, _ = data.clean(sample_csv)
    d = df["date"].dt.date
    assert ((d >= date(2024, 1, 1)) & (d <= date(2024, 1, 15))).sum() == 360  # review: v0.1 had 337
    a = _run(sample_csv, baseline_start=date(2024, 1, 1), baseline_end=date(2024, 1, 15))
    assert a.mv.baseline_days == 15 and a.mv.reporting_start == date(2024, 1, 16)


def test_step_change_in_coc_gives_exact_daily_savings_in_m3_per_day():
    """CoC 2 → 3 after day 15 at constant load. Saving per day = E_day·(2/1 − 3/2)."""
    raw = make_plant(days=30, coc=2.0, coc_after=3.0, change_day=15)
    a = _run(raw, baseline_start=date(2024, 1, 1), baseline_end=date(2024, 1, 15), coc_window_days=1)
    e_day = a.daily["evap_m3"].iloc[20]
    w_day = a.daily["drift_m3"].iloc[20]
    expected = wb.makeup_m3h(e_day, w_day, 2.0) * 1 - wb.makeup_m3h(e_day, w_day, 3.0)
    # intensity method: k_b·E − M; k_b includes drift, so compare against the same construction
    k_b = wb.makeup_m3h(e_day, w_day, 2.0) / e_day
    assert a.mv.daily["saving_m3"].iloc[5] == pytest.approx(k_b * e_day - wb.makeup_m3h(e_day, w_day, 3.0), rel=1e-6)
    assert a.mv.daily["saving_m3"].iloc[5] == pytest.approx(expected, rel=1e-6)
    assert a.mv.daily["saving_m3"].iloc[5] > 100  # m³/day order, not a 24× understatement
    assert a.mv.total_savings_m3 == pytest.approx(a.mv.daily["saving_m3"].sum())
    assert np.isfinite(a.mv.annual_m3)  # 15 reporting days ≥ 14


def test_load_change_is_not_booked_as_savings():
    raw = make_plant(days=30, coc=2.5, flow=3000.0, flow_after=4500.0, change_day=15)
    a = _run(raw, baseline_start=date(2024, 1, 1), baseline_end=date(2024, 1, 15), coc_window_days=1)
    assert abs(a.mv.total_savings_m3) < 1e-6 * a.daily["makeup_m3"].sum()


def test_worse_operation_shows_negative_savings_unclamped():
    raw = make_plant(days=30, coc=3.0, coc_after=1.6, change_day=15)
    a = _run(raw, baseline_start=date(2024, 1, 1), baseline_end=date(2024, 1, 15), coc_window_days=1)
    assert a.mv.total_savings_m3 < 0
    assert a.mv.annual_m3 < 0


def test_metered_makeup_is_used_when_present():
    raw = make_plant(days=20, coc=2.0, makeup_flow=40.0)
    a = _run(raw)
    assert a.mv.basis == "metered makeup"
    assert a.daily["makeup_m3"].iloc[0] == pytest.approx(40.0 * 24)


def test_no_projection_without_enough_reporting_days():
    raw = make_plant(days=20, coc=2.0)
    a = _run(raw, baseline_start=date(2024, 1, 1), baseline_end=date(2024, 1, 15))
    assert a.mv.reporting_days == 5 and np.isnan(a.mv.annual_m3)
    assert any("≥ 14" in m for m in a.mv.messages)


def test_potential_per_30_days_scales_by_hours_covered():
    raw = make_plant(days=10, coc=2.0)
    a = _run(raw)
    assert a.potential.hours_basis == pytest.approx(240)
    total = a.daily["makeup_m3"].sum()
    assert a.potential.current_m3_per_30d == pytest.approx(total * 3)


def test_infeasible_envelope_is_reported_not_crashed(sample_csv):
    s = pipeline.Settings(envelope=ch.EnvelopeSettings(lsi_limit=0.5, pH_model="fixed"))
    a = pipeline.run(sample_csv, s)
    assert a.coc_status == "infeasible"
    assert not a.potential.feasible


def test_baseline_order_validated():
    d = savings.daily_rollup(*_frames())
    with pytest.raises(ValueError):
        savings.measure_and_verify(d, date(2024, 1, 10), date(2024, 1, 1))


def _frames():
    df, _ = data.clean(make_plant(days=5))
    hb = wb.hourly_balance(df, wb.BalanceSettings())
    return df, hb, 3.0


def test_rolling_coc_does_not_leak_across_baseline_boundary():
    raw = make_plant(days=30, coc=2.0, coc_after=3.0, change_day=15)
    a = _run(raw, baseline_start=date(2024, 1, 1), baseline_end=date(2024, 1, 15), coc_window_days=7)
    assert a.daily["coc"].iloc[14] == pytest.approx(2.0)   # last baseline day
    assert a.daily["coc"].iloc[15] == pytest.approx(3.0)   # first reporting day


def test_potential_compares_model_with_model_even_when_metered():
    """A meter reading 20 % above the model must not be booked as extra savings."""
    raw = make_plant(days=10, coc=2.0)
    base = _run(raw).potential
    df_m = raw.copy()
    hb_rate = _run(raw).daily["modelled_makeup_m3"].iloc[0] / 24
    df_m["makeup_flow"] = hb_rate * 1.2
    metered = _run(df_m)
    assert metered.potential.savings_m3_per_30d == pytest.approx(base.savings_m3_per_30d)
    assert metered.potential.meter_vs_model == pytest.approx(1.2, rel=0.01)
    assert any("Metered makeup" in w for w in metered.warnings)


def test_block_bootstrap_is_wider_for_autocorrelated_savings():
    rng = np.random.default_rng(0)
    x = np.repeat(rng.normal(size=10), 7)  # strongly autocorrelated: 7-day plateaus
    iid = savings.block_bootstrap_mean(x, 1)
    blk = savings.block_bootstrap_mean(x, 7)
    assert np.std(blk) > 1.5 * np.std(iid)


def test_unnormalised_load_is_disclosed():
    a = _run(make_plant(days=20), baseline_start=date(2024, 1, 1), baseline_end=date(2024, 1, 10))
    assert any("NOT normalised" in m for m in a.mv.messages)
