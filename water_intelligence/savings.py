"""Daily water balance, baseline-vs-reporting M&V and annualisation.

Design choices (fixing v0.1):
* Everything is rolled up per calendar day in m³/day, so units are consistent
  (v0.1 summed an m³/h rate as m³/day and understated savings 24×).
* Savings are normalised for heat load: the baseline *water intensity*
  k = makeup / evaporation is applied to each reporting day's evaporation, so a
  change in load is not booked as savings or loss.
* If a makeup meter is logged (``makeup_flow``) the savings are metered. Otherwise
  they are labelled as modelled from CoC — never "verified".
* The baseline end date is inclusive of the whole day.
* The annual projection is the mean daily saving × 365 with a bootstrap confidence
  interval. No synthetic training data, no clamping: negative savings are shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .water_balance import makeup_m3h

MIN_DAY_COVERAGE_H = 12.0   # a day needs ≥ 12 h of data to count
MIN_REPORTING_DAYS = 14     # for an annual projection


def daily_rollup(df: pd.DataFrame, hb: pd.DataFrame, max_safe_coc: float,
                 coc_window_days: int = 7) -> pd.DataFrame:
    """Per-day volumes. The CoC used for the modelled balance is a centred rolling
    time-weighted mean over ``coc_window_days``: M = E·C/(C-1) is only valid on average
    over whole blowdown cycles, and applying it to a single day of a batch-blowdown
    sawtooth (CoC ≈ 1 right after a dump) would wildly overstate makeup."""
    w = df["dt_h"]
    day = df["date"].dt.floor("D")
    frame = pd.DataFrame({
        "day": day,
        "hours": w,
        "evap_m3": hb["evap_m3h"] * w,
        "drift_m3": hb["drift_m3h"] * w,
        "heat_mwh": hb["heat_kw"] * w / 1000,
        "coc_w": df["coc"] * w,
        "coc_hours": w.where(df["coc"].notna(), 0.0),
    })
    has_meter = "makeup_flow" in df.columns and df["makeup_flow"].notna().mean() > 0.8
    if has_meter:
        frame["metered_m3"] = df["makeup_flow"] * w
    g = frame.groupby("day").sum(numeric_only=True)
    g["coc_day"] = g["coc_w"] / g["coc_hours"].replace(0, np.nan)
    win = max(int(coc_window_days), 1)
    roll = lambda s: s.rolling(win, center=True, min_periods=1).sum()  # noqa: E731
    g["coc"] = roll(g["coc_w"]) / roll(g["coc_hours"]).replace(0, np.nan)
    g = g.drop(columns=["coc_w", "coc_hours"])
    g = g[(g["hours"] >= MIN_DAY_COVERAGE_H) & g["coc"].gt(1)]
    # Volumes over a day are rates × hours, so the same balance applies to them.
    g["modelled_makeup_m3"] = makeup_m3h(g["evap_m3"], g["drift_m3"], g["coc"])
    if max_safe_coc > 1:
        g["optimised_makeup_m3"] = makeup_m3h(g["evap_m3"], g["drift_m3"], max_safe_coc)
    else:
        g["optimised_makeup_m3"] = np.nan
    g["makeup_m3"] = g["metered_m3"] if has_meter else g["modelled_makeup_m3"]
    g.attrs["basis"] = "metered makeup" if has_meter else "modelled from CoC (no makeup meter)"
    return g


@dataclass
class Potential:
    current_m3_per_30d: float
    optimised_m3_per_30d: float
    savings_m3_per_30d: float
    hours_basis: float
    feasible: bool  # False when the envelope is at or below 1


def potential(daily: pd.DataFrame) -> Potential:
    hours = float(daily["hours"].sum())
    scale = 720.0 / hours if hours else np.nan
    cur = float(daily["makeup_m3"].sum()) * scale
    # NaN when the envelope is infeasible (no safe CoC above 1) — never a silent 0.
    opt = float(daily["optimised_makeup_m3"].sum(min_count=len(daily))) * scale
    feasible = bool(np.isfinite(opt))
    return Potential(cur, opt, cur - opt if feasible else np.nan, hours, feasible)


@dataclass
class MandV:
    basis: str
    baseline_days: int
    reporting_days: int
    baseline_intensity: float = np.nan
    reporting_intensity: float = np.nan
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_savings_m3: float = np.nan
    reporting_start: date | None = None
    reporting_end: date | None = None
    annual_m3: float = np.nan
    annual_ci_m3: tuple = (np.nan, np.nan)
    messages: list = field(default_factory=list)


def measure_and_verify(daily: pd.DataFrame, baseline_start: date, baseline_end: date,
                       seed: int = 0) -> MandV:
    if baseline_end < baseline_start:
        raise ValueError("Baseline end must be on or after baseline start.")
    days = daily.index.date
    base = daily[(days >= baseline_start) & (days <= baseline_end)]
    rep = daily[days > baseline_end]
    mv = MandV(basis=daily.attrs.get("basis", ""), baseline_days=len(base), reporting_days=len(rep))
    if len(base) < 3:
        mv.messages.append("Baseline needs at least 3 full days of data.")
        return mv
    if len(rep) < 1:
        mv.messages.append("No data after the baseline period yet — nothing to verify.")
        return mv

    k_b = base["makeup_m3"].sum() / base["evap_m3"].sum()
    mv.baseline_intensity = float(k_b)
    mv.reporting_intensity = float(rep["makeup_m3"].sum() / rep["evap_m3"].sum())
    out = pd.DataFrame(index=rep.index)
    out["adjusted_baseline_m3"] = k_b * rep["evap_m3"]  # what the baseline would have used at this load
    out["actual_m3"] = rep["makeup_m3"]
    out["saving_m3"] = out["adjusted_baseline_m3"] - out["actual_m3"]
    out["cumulative_m3"] = out["saving_m3"].cumsum()
    mv.daily = out
    mv.total_savings_m3 = float(out["saving_m3"].sum())
    mv.reporting_start, mv.reporting_end = rep.index.min().date(), rep.index.max().date()

    if len(rep) >= MIN_REPORTING_DAYS:
        s = out["saving_m3"].to_numpy()
        rng = np.random.default_rng(seed)
        boots = rng.choice(s, size=(2000, len(s)), replace=True).mean(axis=1) * 365
        mv.annual_m3 = float(s.mean() * 365)
        mv.annual_ci_m3 = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
        mv.messages.append("Annual figure = mean daily saving × 365. It carries no seasonality; "
                           "a full year of data is needed for that.")
    else:
        mv.messages.append(f"Annual projection needs ≥ {MIN_REPORTING_DAYS} reporting days "
                           f"(have {len(rep)}).")
    return mv
