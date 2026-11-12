"""One call that runs the whole analysis. The app and the tests both use this."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import anomaly, chemistry, data, savings, water_balance


@dataclass
class Settings:
    flow_unit: str = "m³/h"
    balance: water_balance.BalanceSettings = field(default_factory=water_balance.BalanceSettings)
    envelope: chemistry.EnvelopeSettings = field(default_factory=chemistry.EnvelopeSettings)
    baseline_start: date | None = None
    baseline_end: date | None = None   # inclusive
    coc_window_days: int = 7
    water_cost: float = 50.0           # ₹/m³
    fee_share: float = 0.20            # share of savings billed under pay-as-you-save


@dataclass
class Analysis:
    settings: Settings
    df: pd.DataFrame
    quality: data.QualityReport
    hourly: pd.DataFrame
    lsi: pd.DataFrame
    env: chemistry.Envelope
    daily: pd.DataFrame
    potential: savings.Potential
    mv: savings.MandV
    anomalies: anomaly.AnomalyResult
    current_coc: float
    baseline: tuple  # (start, end) inclusive dates actually used
    warnings: list

    @property
    def period(self) -> tuple[date, date]:
        return self.df["date"].min().date(), self.df["date"].max().date()

    @property
    def lsi_skin_hours_above(self) -> float:
        mask = self.lsi["lsi_skin"] > self.settings.envelope.lsi_limit
        return float(self.df.loc[mask, "dt_h"].sum())

    @property
    def coc_status(self) -> str:
        c, m = self.current_coc, self.env.max_safe_coc
        if m <= 1:
            return "infeasible"
        if c > m * (1 + 1e-6):
            return "above"
        return "below"


def default_baseline(df: pd.DataFrame) -> tuple[date, date]:
    start = df["date"].min().date()
    span = (df["date"].max().date() - start).days + 1
    return start, start + timedelta(days=max(span // 2, 3) - 1)


def run(raw: pd.DataFrame, s: Settings) -> Analysis:
    df, rep = data.clean(raw, s.flow_unit)
    warnings: list[str] = []

    hb = water_balance.hourly_balance(df, s.balance)
    msg = data.heat_load_check(float(hb["heat_kw"].median()))
    if msg:
        warnings.append(msg)

    lsi_df = chemistry.hourly_lsi(df, s.envelope.skin_temp_c)
    drift_cap = water_balance.drift_limited_coc(float(hb["evap_m3h"].mean()), float(hb["drift_m3h"].mean()))
    env = chemistry.envelope(df, lsi_df, s.envelope, drift_cap)

    w = df["dt_h"].where(df["coc"].notna())
    current = float(np.nansum(df["coc"] * w) / np.nansum(w))  # time-weighted mean of per-row ratios

    daily = savings.daily_rollup(df, hb, env.max_safe_coc, s.coc_window_days)
    pot = savings.potential(daily)

    if s.baseline_start is None or s.baseline_end is None:
        bs, be = default_baseline(df)
    else:
        bs, be = s.baseline_start, s.baseline_end
    mv = savings.measure_and_verify(daily, bs, be)

    d = df["date"].dt.date
    train = (d >= bs) & (d <= be)
    anomalies = anomaly.detect(df, lsi_df, env.max_safe_coc, s.envelope.lsi_limit, train)

    return Analysis(s, df, rep, hb, lsi_df, env, daily, pot, mv, anomalies, current, (bs, be), warnings)


@dataclass
class PointResult:
    coc: float
    lsi_basin: float
    lsi_skin: float
    max_safe_coc: float
    heat_kw: float
    evap_m3h: float
    drift_m3h: float
    blowdown_now_m3h: float
    makeup_now_m3h: float
    makeup_opt_m3h: float | None


def single_point(tds_circ, tds_makeup, pH, temp_c, flow_m3h, calcium, alkalinity,
                 bal: water_balance.BalanceSettings, env: chemistry.EnvelopeSettings) -> PointResult:
    """Quick calculator for one set of readings. CoC is derived from the TDS ratio, so
    it can never disagree with the chemistry inputs."""
    if tds_makeup <= 0 or tds_circ <= tds_makeup:
        raise ValueError("Circulating TDS must be greater than makeup TDS (CoC > 1).")
    coc = tds_circ / tds_makeup
    frame = pd.DataFrame({"date": [pd.Timestamp("2000-01-01")], "temp_C": [temp_c], "flow_m3h": [flow_m3h]})
    hb = water_balance.hourly_balance(frame, bal).iloc[0]
    lb = float(chemistry.lsi(pH, tds_circ, temp_c, calcium, alkalinity))
    ls = float(chemistry.lsi(pH, tds_circ, env.skin_temp_c, calcium, alkalinity))
    caps = [coc * float(chemistry.ratio_to_reach_lsi(ls, env.lsi_limit, env.pH_model)),
            water_balance.drift_limited_coc(hb.evap_m3h, hb.drift_m3h)]
    if env.vendor_max_coc:
        caps.append(env.vendor_max_coc)
    max_coc = min(caps)
    return PointResult(
        coc=coc, lsi_basin=lb, lsi_skin=ls, max_safe_coc=max_coc, heat_kw=float(hb.heat_kw),
        evap_m3h=float(hb.evap_m3h), drift_m3h=float(hb.drift_m3h),
        blowdown_now_m3h=float(water_balance.blowdown_m3h(hb.evap_m3h, hb.drift_m3h, coc)),
        makeup_now_m3h=float(water_balance.makeup_m3h(hb.evap_m3h, hb.drift_m3h, coc)),
        makeup_opt_m3h=float(water_balance.makeup_m3h(hb.evap_m3h, hb.drift_m3h, max_coc)) if max_coc > 1 else None,
    )
