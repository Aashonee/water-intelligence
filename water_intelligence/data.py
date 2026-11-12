"""Input schema, unit handling and the data-quality layer.

Column semantics (declared, so CoC-dependent maths is unambiguous):

Required
    date               timestamp of the reading
    TDS_circ           circulating (basin) water TDS, mg/L
    TDS_makeup         makeup water TDS, mg/L
    pH                 circulating water pH
    temp_C             circulating (basin) water temperature, °C
    flow_rate          recirculation flow; unit chosen in the app (default m³/h)
    calcium_hardness   circulating water calcium hardness, mg/L as CaCO3
    alkalinity         circulating water M-alkalinity, mg/L as CaCO3

Optional (used when present)
    cond_circ, cond_makeup          conductivity, µS/cm  -> preferred CoC tracer
    chloride_circ, chloride_makeup  chloride, mg/L       -> CoC tracer if no conductivity
    T_hot, T_cold                   tower inlet/outlet water temperature, °C (per-row heat load)
    makeup_flow                     metered makeup, m³/h (enables metered M&V)
    blowdown_flow                   metered blowdown, m³/h
    calcium_makeup                  makeup calcium hardness, mg/L as CaCO3 (calcium-cycling check)
    silica_makeup                   makeup silica, mg/L as SiO2
    magnesium_makeup                makeup magnesium hardness, mg/L as CaCO3
    sulfate_makeup                  makeup sulfate, mg/L as SO4
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "date", "TDS_circ", "TDS_makeup", "pH", "temp_C",
    "flow_rate", "calcium_hardness", "alkalinity",
]
OPTIONAL_COLUMNS = [
    "cond_circ", "cond_makeup", "chloride_circ", "chloride_makeup",
    "T_hot", "T_cold", "makeup_flow", "blowdown_flow",
    "calcium_makeup", "silica_makeup", "magnesium_makeup", "sulfate_makeup",
]

# Conversion of recirculation flow to m³/h.
FLOW_UNITS = {
    "m³/h": 1.0,
    "L/s": 3.6,
    "US gpm": 0.227124707,
    "kg/h (≈ L/h)": 1e-3,
}

# Physically plausible ranges for an open recirculating cooling tower. Readings
# outside these are treated as sensor faults and excluded.
VALID_RANGES = {
    "TDS_circ": (1.0, 50_000.0),
    "TDS_makeup": (1.0, 20_000.0),
    "pH": (4.0, 11.0),
    "temp_C": (0.0, 70.0),
    "flow_rate_m3h": (1e-3, 1e6),
    "calcium_hardness": (1.0, 5_000.0),
    "alkalinity": (1.0, 5_000.0),
    "T_hot": (0.0, 80.0),
    "T_cold": (0.0, 70.0),
    "makeup_flow": (0.0, 1e5),
    "blowdown_flow": (0.0, 1e5),
}

# A real industrial tower rejects more than ~100 kW. Below this the flow unit is
# probably wrong (the old code's "3,300 kg/h" gave ~19 kW).
MIN_PLAUSIBLE_HEAT_KW = 100.0


class DataError(ValueError):
    """Raised when the uploaded file cannot be used at all."""


@dataclass
class QualityReport:
    rows_in: int = 0
    rows_used: int = 0
    dropped: dict = field(default_factory=dict)
    duplicates: int = 0
    interval_h: float = float("nan")
    gaps: int = 0
    longest_gap_h: float = 0.0
    hours_covered: float = 0.0
    tracer: str = "TDS"
    coc_outliers: int = 0
    coc_below_one: int = 0
    warnings: list = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_used

    def summary_lines(self) -> list[str]:
        lines = [
            f"{self.rows_used} of {self.rows_in} rows used "
            f"({self.hours_covered:,.0f} h covered, logging interval {self.interval_h:g} h)",
            f"CoC tracer: {self.tracer}",
        ]
        for reason, n in self.dropped.items():
            if n:
                lines.append(f"{n} rows dropped: {reason}")
        if self.duplicates:
            lines.append(f"{self.duplicates} duplicate timestamps removed")
        if self.gaps:
            lines.append(f"{self.gaps} logging gaps (longest {self.longest_gap_h:g} h)")
        if self.coc_outliers:
            lines.append(f"{self.coc_outliers} CoC readings excluded as sensor spikes")
        if self.coc_below_one:
            lines.append(f"{self.coc_below_one} CoC readings ≤ 1 excluded (tracer noise after blowdown)")
        return lines + list(self.warnings)


def pick_tracer(df: pd.DataFrame) -> tuple[str, str, str]:
    """Choose the conservative tracer for CoC: conductivity > chloride > TDS."""
    for name, c, m in (
        ("conductivity", "cond_circ", "cond_makeup"),
        ("chloride", "chloride_circ", "chloride_makeup"),
    ):
        if c in df.columns and m in df.columns and df[[c, m]].notna().mean().min() > 0.8:
            return name, c, m
    return "TDS", "TDS_circ", "TDS_makeup"


def hampel_mask(x: pd.Series, window: int = 7, n_sigma: float = 4.0) -> pd.Series:
    """True where ``x`` is a spike relative to its rolling median (Hampel filter)."""
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    scale = 1.4826 * mad
    floor = 0.02 * med.abs()  # don't flag tiny wiggles when the series is very smooth
    return (x - med).abs() > n_sigma * np.maximum(scale, floor)


def clean(raw: pd.DataFrame, flow_unit: str = "m³/h") -> tuple[pd.DataFrame, QualityReport]:
    """Validate, coerce and clean raw sensor data.

    Returns a time-sorted frame with derived columns:
      flow_m3h   recirculation flow in m³/h
      dt_h       hours each row represents (handles gaps and non-hourly logging)
      coc        per-row cycles of concentration from the chosen tracer (NaN if excluded)
    """
    rep = QualityReport(rows_in=len(raw))
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}. Expected: {REQUIRED_COLUMNS}")
    if flow_unit not in FLOW_UNITS:
        raise DataError(f"Unknown flow unit {flow_unit!r}; choose one of {list(FLOW_UNITS)}")

    keep = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in raw.columns]
    df = raw[keep].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in keep:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_date = df["date"].isna()
    rep.dropped["unreadable timestamp"] = int(bad_date.sum())
    df = df[~bad_date]

    req_num = [c for c in REQUIRED_COLUMNS if c != "date"]
    nan_rows = df[req_num].isna().any(axis=1)
    rep.dropped["missing / non-numeric required value"] = int(nan_rows.sum())
    df = df[~nan_rows]

    df = df.sort_values("date", kind="stable")
    dup = df["date"].duplicated(keep="first")
    rep.duplicates = int(dup.sum())
    df = df[~dup]

    df["flow_m3h"] = df["flow_rate"] * FLOW_UNITS[flow_unit]

    out_of_range = pd.Series(False, index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            v = df[col]
            out_of_range |= v.notna() & ((v < lo) | (v > hi))
    if "T_hot" in df.columns and "T_cold" in df.columns:
        both = df["T_hot"].notna() & df["T_cold"].notna()
        out_of_range |= both & (df["T_hot"] <= df["T_cold"])
    rep.dropped["value outside physical range"] = int(out_of_range.sum())
    df = df[~out_of_range].reset_index(drop=True)

    if len(df) < 2:
        raise DataError("Fewer than 2 valid rows after cleaning — check the file.")

    # Time weighting: each row stands for the interval to the next reading, capped at
    # the typical interval so a logging gap isn't silently filled.
    diffs_h = df["date"].diff().dt.total_seconds().div(3600).shift(-1)
    interval = float(np.nanmedian(diffs_h))
    if not np.isfinite(interval) or interval <= 0:
        raise DataError("Could not infer a logging interval from the timestamps.")
    rep.interval_h = round(interval, 4)
    gap_mask = diffs_h > 1.5 * interval
    rep.gaps = int(gap_mask.sum())
    rep.longest_gap_h = float(diffs_h[gap_mask].max()) if rep.gaps else 0.0
    df["dt_h"] = diffs_h.clip(upper=interval).fillna(interval)
    rep.hours_covered = float(df["dt_h"].sum())

    tracer, c_col, m_col = pick_tracer(df)
    rep.tracer = tracer
    ratio = df[c_col] / df[m_col]
    spikes = hampel_mask(ratio)
    below_one = ratio <= 1.0
    rep.coc_outliers = int((spikes & ~below_one).sum())
    rep.coc_below_one = int(below_one.sum())
    df["coc"] = ratio.where(~spikes & ~below_one)

    if df["coc"].notna().sum() < 2:
        raise DataError("No usable cycles-of-concentration readings (tracer ratio ≤ 1 everywhere?).")

    rep.rows_used = len(df)
    if rep.interval_h > 6:
        rep.warnings.append("Logging interval > 6 h: hourly chemistry excursions will be missed.")
    return df, rep


def heat_load_check(heat_kw: float) -> str | None:
    if heat_kw < MIN_PLAUSIBLE_HEAT_KW:
        return (
            f"Heat rejected is only {heat_kw:,.0f} kW — too small for an industrial tower. "
            "The flow-rate unit is probably wrong (e.g. kg/h selected for m³/h data)."
        )
    return None
