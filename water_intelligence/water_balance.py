"""Cooling-tower water balance.

    Makeup     M = E + B + W
    Solids     M·C_m = (B + W)·C_c   ->   B + W = E/(CoC-1)

so  B = E/(CoC-1) - W   and   M = E·CoC/(CoC-1).

Drift (windage) carries dissolved solids out of the tower exactly like blowdown, so
it is *not* added to evaporation. v0.1 used B = (E+W)/(CoC-1), which overstated
blowdown by ~80% and makeup by ~35% on the sample data.

Evaporation follows the heat rejected (Q = m·Cp·ΔT), so it is computed per row from
the measured ΔT when T_hot/T_cold are logged, otherwise from a site-constant ΔT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .properties import CP_WATER, latent_heat_kj_per_kg, water_density_kg_per_m3


@dataclass
class BalanceSettings:
    delta_t_c: float = 5.0        # range, used only when T_hot/T_cold are not logged
    evap_fraction: float = 0.85   # share of heat rejected by evaporation (rest is sensible).
                                  # 0.85 reproduces the standard E = 0.00153·ΔT(°C)·R rule;
                                  # literature range ~0.65 (humid) to 1.0 (dry/hot).
    drift_pct: float = 0.10       # % of recirculation flow lost as drift


def hourly_balance(df: pd.DataFrame, s: BalanceSettings) -> pd.DataFrame:
    """Per-row heat load, evaporation and drift (rates in m³/h)."""
    if not 0 < s.evap_fraction <= 1:
        raise ValueError("evap_fraction must be in (0, 1].")
    if s.drift_pct < 0:
        raise ValueError("drift_pct must be ≥ 0.")
    out = pd.DataFrame(index=df.index)
    if {"T_hot", "T_cold"} <= set(df.columns) and df[["T_hot", "T_cold"]].notna().all(axis=1).mean() > 0.8:
        dT = (df["T_hot"] - df["T_cold"]).fillna(s.delta_t_c)
        t_mean = ((df["T_hot"] + df["T_cold"]) / 2).fillna(df["temp_C"] + s.delta_t_c / 2)
        out["delta_t_source"] = "measured"
    else:
        if s.delta_t_c <= 0:
            raise ValueError("ΔT must be > 0.")
        dT = pd.Series(s.delta_t_c, index=df.index)
        t_mean = df["temp_C"] + s.delta_t_c / 2  # basin is the cold side
        out["delta_t_source"] = "site constant"
    rho = water_density_kg_per_m3(t_mean)
    lam = latent_heat_kj_per_kg(t_mean)
    m_kg_s = df["flow_m3h"] * rho / 3600
    out["delta_t"] = dT
    out["heat_kw"] = m_kg_s * CP_WATER * dT
    out["evap_m3h"] = s.evap_fraction * out["heat_kw"] / lam * 3600 / rho
    out["drift_m3h"] = s.drift_pct / 100 * df["flow_m3h"]
    return out


def blowdown_m3h(evap, drift, coc):
    """Blowdown needed to hold ``coc``. Negative means drift alone exceeds it."""
    coc = np.asarray(coc, dtype=float)
    if np.any(coc <= 1):
        raise ValueError("CoC must be > 1 for a water balance.")
    return np.asarray(evap) / (coc - 1) - np.asarray(drift)


def makeup_m3h(evap, drift, coc):
    """Makeup = E + max(B, 0) + W. When drift exceeds the blowdown requirement the tower
    cannot actually reach ``coc``; makeup is then E + W."""
    b = blowdown_m3h(evap, drift, coc)
    return np.asarray(evap) + np.maximum(b, 0) + np.asarray(drift)


def drift_limited_coc(evap_m3h: float, drift_m3h: float) -> float:
    """Highest CoC reachable with blowdown closed: E/(CoC-1) = W -> CoC = 1 + E/W."""
    return float("inf") if drift_m3h <= 0 else 1 + evap_m3h / drift_m3h


def period_volume(rate_m3h, dt_h) -> float:
    """Integrate a rate (m³/h) over each row's duration (h)."""
    return float(np.nansum(np.asarray(rate_m3h) * np.asarray(dt_h)))
