"""Anomaly detection: physics rules for known failure modes + a learned detector.

v0.1 problems fixed here:
* ``contamination=0.05`` forced 5% of rows to be flagged on any plant. We now train on
  the baseline window, threshold on the training score distribution, and only raise
  an *event* when the deviation persists (≥ ``min_hours``). A healthy plant can report
  zero anomalies.
* The model now sees time context (CoC, its rate of change, deviation from the 24 h
  median) and the scaling-relevant chemistry (Ca, alkalinity, temperature, makeup TDS,
  skin LSI), not just TDS/pH/flow.
* Known failure modes are detected with explicit, explainable rules.
* The "explanation" is the readings that deviate most from the baseline (robust
  z-scores). It says *what* is unusual, not *why*. SHAP on an isolation forest
  explains isolation depth, not root cause, so it was removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_LABELS = {
    "coc": "CoC",
    "coc_rate": "CoC rate of change",
    "coc_dev24": "CoC vs 24 h median",
    "pH": "pH",
    "temp_C": "basin temperature",
    "flow_m3h": "recirculation flow",
    "calcium_hardness": "calcium hardness",
    "alkalinity": "alkalinity",
    "TDS_makeup": "makeup TDS",
    "lsi_skin": "skin LSI",
}


@dataclass
class AnomalyResult:
    rows: pd.DataFrame     # per-row score / flag
    events: pd.DataFrame   # start, end, hours, kind, detail
    trained_on: str

    @property
    def event_hours(self) -> float:
        return float(self.events["hours"].sum()) if len(self.events) else 0.0

    def dominant_signal(self) -> str:
        learned = self.events[self.events["kind"] == "Unusual pattern (learned)"]
        if not len(learned):
            return "none"
        return learned["detail"].str.extract(r"^([^(]+)")[0].str.strip().mode().iat[0]


def build_features(df: pd.DataFrame, lsi_df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    coc = df["coc"].interpolate(limit_direction="both")
    f["coc"] = coc
    f["coc_rate"] = coc.diff().div(df["dt_h"].shift(1)).fillna(0.0)
    window = max(int(round(24 / max(df["dt_h"].median(), 1e-6))), 3)
    f["coc_dev24"] = coc - coc.rolling(window, min_periods=1, center=True).median()
    for c in ("pH", "temp_C", "flow_m3h", "calcium_hardness", "alkalinity", "TDS_makeup"):
        f[c] = df[c]
    f["lsi_skin"] = lsi_df["lsi_skin"]
    return f


def runs(mask: pd.Series, dt_h: pd.Series, min_hours: float) -> list[tuple[int, int, float]]:
    """(start_pos, end_pos, hours) for consecutive True runs lasting ≥ min_hours."""
    m = mask.to_numpy(dtype=bool)
    dt = dt_h.to_numpy(dtype=float)
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            hrs = float(dt[i:j + 1].sum())
            if hrs >= min_hours:
                out.append((i, j, hrs))
            i = j + 1
        else:
            i += 1
    return out


def _flatline(series: pd.Series) -> pd.Series:
    return series.diff().abs().fillna(1.0).eq(0)


def detect(df: pd.DataFrame, lsi_df: pd.DataFrame, max_safe_coc: float, lsi_limit: float,
           train_mask: pd.Series | None = None, min_hours: float = 3.0,
           score_quantile: float = 0.01, z_limit: float = 6.0, seed: int = 0) -> AnomalyResult:
    """A row is flagged when the isolation-forest score is below the 1st percentile of
    baseline scores *or* any single reading is more than ``z_limit`` robust standard
    deviations from its baseline median (isolation scores saturate for one extreme
    feature among many). Only runs of flagged rows lasting ≥ ``min_hours`` become events."""
    feats = build_features(df, lsi_df)
    feats = feats.fillna(feats.median(numeric_only=True))

    if train_mask is None or train_mask.sum() < 48:
        train_mask = pd.Series(np.arange(len(df)) < len(df) // 2, index=df.index)
        trained_on = "first half of the data (baseline window too short)"
    else:
        trained_on = "baseline window"

    train = feats[train_mask]
    med = train.median()
    iqr = (train.quantile(0.75) - train.quantile(0.25)).replace(0, np.nan)
    scale = (1.4826 * (train - med).abs().median()).replace(0, np.nan).fillna(iqr / 1.349).fillna(1.0)
    z = (feats - med) / scale

    model = IsolationForest(n_estimators=200, random_state=seed)
    model.fit(z[train_mask])
    score = pd.Series(model.score_samples(z), index=df.index)
    threshold = float(np.quantile(score[train_mask], score_quantile))
    flagged = (score < threshold) | (z.abs().max(axis=1) > z_limit)

    rows = pd.DataFrame({"score": score, "flagged": flagged}, index=df.index)
    events = []

    def add(kind, mask, detail_fn, hours=min_hours):
        for s, e, hrs in runs(mask, df["dt_h"], hours):
            events.append({"start": df["date"].iat[s], "end": df["date"].iat[e],
                           "hours": hrs, "kind": kind, "detail": detail_fn(s, e)})

    def top_signals(s, e):
        zz = z.iloc[s:e + 1].abs().mean().sort_values(ascending=False)
        return ", ".join(f"{FEATURE_LABELS[k]} (|z|≈{v:.1f})" for k, v in zz.head(2).items())

    add("Unusual pattern (learned)", flagged, top_signals)

    if max_safe_coc > 1:
        add("CoC above safe envelope — check blowdown valve / controller",
            df["coc"].fillna(0) > max_safe_coc,
            lambda s, e: f"peak CoC {df['coc'].iloc[s:e + 1].max():.2f} vs limit {max_safe_coc:.2f}", 6.0)
    add("Skin LSI above programme limit — scaling risk",
        lsi_df["lsi_skin"] > lsi_limit,
        lambda s, e: f"peak skin LSI {lsi_df['lsi_skin'].iloc[s:e + 1].max():+.2f} vs limit {lsi_limit:+.2f}", 6.0)
    for col, label in (("TDS_circ", "TDS"), ("pH", "pH"), ("temp_C", "temperature")):
        add(f"Sensor flatline — {label} probe", _flatline(df[col]),
            lambda s, e, c=col: f"{c} stuck at {df[c].iat[s]:g}", 6.0)
    if "calcium_makeup" in df.columns and df["calcium_makeup"].notna().any():
        ca_coc = df["calcium_hardness"] / df["calcium_makeup"]
        deficit = (ca_coc / df["coc"]) < 0.85
        add("Calcium cycling below tracer CoC — calcium is precipitating",
            deficit.fillna(False),
            lambda s, e: f"Ca cycles {ca_coc.iloc[s:e + 1].mean():.2f} vs tracer "
                         f"{df['coc'].iloc[s:e + 1].mean():.2f}", 6.0)

    ev = pd.DataFrame(events, columns=["start", "end", "hours", "kind", "detail"])
    return AnomalyResult(rows=rows, events=ev.sort_values("start").reset_index(drop=True),
                         trained_on=trained_on)
