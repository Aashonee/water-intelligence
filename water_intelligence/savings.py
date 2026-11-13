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
* The annual projection is the mean daily saving × 365 with a moving-block bootstrap
  confidence interval (days are autocorrelated). No synthetic training data, no clamping: negative savings are shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .water_balance import makeup_m3h

MIN_DAY_COVERAGE_H = 12.0   # a day needs ≥ 12 h of data to count
MIN_REPORTING_DAYS = 14     # for an annual projection


DUMP_DROP = 0.15  # a ≥15 % fall in CoC between consecutive readings = batch blowdown


def detect_dumps(df: pd.DataFrame, drop: float = DUMP_DROP) -> np.ndarray:
    """Row positions where a batch blowdown happened (CoC falls sharply in one step)."""
    c = df["coc"].to_numpy()
    out, last = [], np.nan
    for i, v in enumerate(c):
        if np.isnan(v):
            continue
        if np.isfinite(last) and v < (1 - drop) * last:
            out.append(i)
        last = v
    return np.asarray(out, dtype=int)


def batch_makeup_rows(df: pd.DataFrame, hb: pd.DataFrame, dumps: np.ndarray) -> pd.Series:
    """Per-row makeup volume (m³) for a tower blown down in batches.

    Over one cycle between dumps (all quantities in volumes and CoC units):
        water:   M = (E + W) + D
        solids:  M = W·Σ(CoC·dt) + D·CoC_peak     (drift leaves at basin CoC, the dump at the peak)
    so  D = [(E + W) − W·Σ(CoC·dt)] / (CoC_peak − 1).
    The dump volume D is booked on the dump row (that is when the makeup refills it).
    Partial cycles at the ends use the average makeup/evaporation ratio of whole cycles.
    The steady-state M = E·CoC/(CoC−1) on the mean CoC overstates makeup here
    (~30 % in a simulated sawtooth) because it assumes solids leave at the mean CoC.
    """
    dt = df["dt_h"].to_numpy()
    e = hb["evap_m3h"].to_numpy() * dt
    w = hb["drift_m3h"].to_numpy() * dt
    coc = df["coc"].interpolate(limit_direction="both").to_numpy()
    vol = e + w
    ratios = []
    for a, b in zip(dumps[:-1], dumps[1:], strict=False):
        seg = slice(a, b)  # from one dump (inclusive) up to the reading before the next
        peak = coc[b - 1]
        if peak <= 1.05:
            continue
        d = (e[seg].sum() + w[seg].sum() - (w[seg] * coc[seg]).sum()) / (peak - 1)
        d = max(d, 0.0)
        vol[b] += d
        ratios.append((e[seg].sum() + w[seg].sum() + d) / e[seg].sum())
    if ratios:
        k = float(np.mean(ratios))
        for seg in (slice(0, dumps[0]), slice(dumps[-1], len(df))):
            vol[seg] = e[seg] * k
    return pd.Series(vol, index=df.index)


def daily_rollup(df: pd.DataFrame, hb: pd.DataFrame, max_safe_coc: float,
                 coc_window_days: int = 7, split_after: date | None = None) -> pd.DataFrame:
    """Per-day volumes. The CoC used for the modelled balance is a centred rolling
    time-weighted mean over ``coc_window_days``: M = E·C/(C-1) is only valid on average
    over whole blowdown cycles, and applying it to a single day of a batch-blowdown
    sawtooth (CoC ≈ 1 right after a dump) would wildly overstate makeup.

    ``split_after`` (the baseline end) stops the window from mixing baseline and
    reporting days, which would smear a real step change across the boundary."""
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
    dumps = detect_dumps(df)
    batch = len(dumps) >= 3  # at least two whole cycles
    if batch:
        frame["batch_makeup_m3"] = batch_makeup_rows(df, hb, dumps)
    g = frame.groupby("day").sum(numeric_only=True)
    g["coc_day"] = g["coc_w"] / g["coc_hours"].replace(0, np.nan)
    win = max(int(coc_window_days), 1)
    segment = (g.index.date > split_after) if split_after is not None else np.zeros(len(g), bool)

    def roll(col):
        return col.groupby(segment).transform(lambda x: x.rolling(win, center=True, min_periods=1).sum())

    g["coc"] = roll(g["coc_w"]) / roll(g["coc_hours"]).replace(0, np.nan)
    g = g.drop(columns=["coc_w", "coc_hours"])
    g = g[(g["hours"] >= MIN_DAY_COVERAGE_H) & g["coc"].gt(1)]
    # Volumes over a day are rates × hours, so the same balance applies to them.
    if batch:
        g["modelled_makeup_m3"] = g["batch_makeup_m3"]
    else:
        g["modelled_makeup_m3"] = makeup_m3h(g["evap_m3"], g["drift_m3"], g["coc"])
    if max_safe_coc > 1:
        g["optimised_makeup_m3"] = makeup_m3h(g["evap_m3"], g["drift_m3"], max_safe_coc)
    else:
        g["optimised_makeup_m3"] = np.nan
    g["makeup_m3"] = g["metered_m3"] if has_meter else g["modelled_makeup_m3"]
    g.attrs["basis"] = "metered makeup" if has_meter else "modelled from CoC (no makeup meter)"
    g.attrs["coc_window_days"] = win
    g.attrs["blowdown_mode"] = (f"batch ({len(dumps)} dumps detected; solids balance per cycle)"
                                if batch else "continuous (steady-state balance)")
    if batch:
        cycle_days = float(df["dt_h"].sum() / 24 / max(len(dumps), 1))
        g.attrs["coc_window_days"] = max(win, int(round(cycle_days)))  # bootstrap block ≥ one cycle
    return g


@dataclass
class Potential:
    current_m3_per_30d: float
    optimised_m3_per_30d: float
    savings_m3_per_30d: float
    hours_basis: float
    feasible: bool  # False when the envelope is at or below 1
    meter_vs_model: float = np.nan  # metered / modelled makeup, when a meter is logged


def potential(daily: pd.DataFrame) -> Potential:
    """Current vs optimised makeup, both from the same model so model error cancels.
    (Comparing metered current use with modelled optimised use would book any model
    bias as savings.) The meter, if present, is reported as a model check."""
    hours = float(daily["hours"].sum())
    scale = 720.0 / hours if hours else np.nan
    cur = float(daily["modelled_makeup_m3"].sum()) * scale
    # NaN when the envelope is infeasible (no safe CoC above 1) — never a silent 0.
    opt = float(daily["optimised_makeup_m3"].sum(min_count=len(daily))) * scale
    feasible = bool(np.isfinite(opt))
    ratio = (float(daily["metered_m3"].sum() / daily["modelled_makeup_m3"].sum())
             if "metered_m3" in daily.columns else np.nan)
    return Potential(cur, opt, cur - opt if feasible else np.nan, hours, feasible, ratio)


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


def block_bootstrap_mean(x: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Moving-block bootstrap of the mean. Daily savings are autocorrelated (shared
    rolling CoC window, weather, operator habits); an i.i.d. bootstrap would give a
    falsely narrow interval."""
    n = len(x)
    block = int(min(max(block, 1), n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return x[idx].mean(axis=1)


def measure_and_verify(daily: pd.DataFrame, baseline_start: date, baseline_end: date,
                       seed: int = 0, load_normalised: bool = True) -> MandV:
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
    if not load_normalised:
        mv.messages.append("T_hot/T_cold not logged: evaporation is scaled by flow only, so changes in "
                           "heat load (ΔT) between baseline and reporting are NOT normalised.")

    if len(rep) >= MIN_REPORTING_DAYS:
        s = out["saving_m3"].to_numpy()
        block = max(int(daily.attrs.get("coc_window_days", 1)), 3)
        boots = block_bootstrap_mean(s, block, seed=seed) * 365
        mv.annual_m3 = float(s.mean() * 365)
        mv.annual_ci_m3 = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
        mv.messages.append("Annual figure = mean daily saving × 365. It carries no seasonality; "
                           "a full year of data is needed for that.")
    else:
        mv.messages.append(f"Annual projection needs ≥ {MIN_REPORTING_DAYS} reporting days "
                           f"(have {len(rep)}).")
    return mv
