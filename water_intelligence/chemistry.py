"""Water chemistry: Langelier Saturation Index and the max-safe-CoC envelope.

Key corrections versus v0.1:
* The max-CoC solve now concentrates calcium, alkalinity and TDS together with CoC
  and lets pH rise (CO2 stripping) instead of holding them constant, and it targets
  the LSI *limit* instead of plugging the measured LSI back in (which returned the
  current CoC exactly).
* LSI is judged at the heat-exchanger skin temperature, hour by hour, not once at
  basin temperature on period-averaged inputs.
* LSI is one constraint among several (silica, Mg×SiO2, CaSO4, chloride, inhibitor
  limit, drift). The binding one sets the envelope, and unchecked ones are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LSI_VALID_MAX_TDS = 10_000.0  # mg/L; above this use Stiff-Davis instead of LSI

# Typical guideline limits. Every one of these should be confirmed with the plant's
# water-treatment vendor; they are inputs in the app, these are only defaults.
DEFAULT_SILICA_LIMIT = 150.0          # mg/L SiO2 in circulating water
DEFAULT_MG_SILICA_PRODUCT = 35_000.0  # Mg (as CaCO3) × SiO2
DEFAULT_CASO4_PRODUCT = 500_000.0     # Ca (as CaCO3) × SO4
CHLORIDE_LIMITS = {                   # mg/L Cl in circulating water, by wetted metallurgy
    "304 stainless steel": 200.0,
    "316 stainless steel": 1000.0,
    "Carbon steel / copper alloys": 500.0,
    "Titanium / duplex": 5000.0,
}

PH_MODELS = ("calibrated-equilibrium", "fixed")

# Indicative maximum LSI by treatment programme, applied at the heat-exchanger skin
# temperature. The real number comes from the plant's chemical vendor.
TREATMENT_PROGRAMMES = {
    "No scale inhibitor": 0.5,
    "Standard phosphonate inhibitor": 1.8,
    "High-performance polymer / phosphonate blend": 2.5,
}


def lsi(pH, tds, temp_c, calcium, alkalinity):
    """Langelier Saturation Index (vectorised). Calcium and alkalinity in mg/L as CaCO3."""
    tds, calcium, alkalinity = (np.asarray(v, dtype=float) for v in (tds, calcium, alkalinity))
    if np.any(tds <= 0) or np.any(calcium <= 0) or np.any(alkalinity <= 0):
        raise ValueError("TDS, calcium and alkalinity must be > 0 for LSI.")
    a = (np.log10(tds) - 1) / 10
    b = -13.12 * np.log10(np.asarray(temp_c, dtype=float) + 273.15) + 34.55
    c = np.log10(calcium) - 0.4
    d = np.log10(alkalinity)
    phs = 9.3 + a + b - c - d
    return np.asarray(pH, dtype=float) - phs


def lsi_slope_per_decade(pH_model: str) -> float:
    """dLSI / dlog10(concentration ratio r).

    Concentrating by r raises A by 0.1·log r and lowers C and D by log r each, so
    pHs falls by 1.9·log r. Under the calibrated-equilibrium pH model (Caplan-type
    relation pH = k + log10(M-alk) for a tower in CO2 equilibrium with air, with k
    fitted to the measured pH), pH rises by log r as well, giving 2.9. That
    equilibrium assumption is the conservative (upper-bound) case for pH rise.
    """
    if pH_model == "calibrated-equilibrium":
        return 2.9
    if pH_model == "fixed":
        return 1.9
    raise ValueError(f"pH_model must be one of {PH_MODELS}")


def lsi_after_concentrating(lsi_now, r, pH_model="calibrated-equilibrium"):
    return np.asarray(lsi_now, dtype=float) + lsi_slope_per_decade(pH_model) * np.log10(r)


def ratio_to_reach_lsi(lsi_now, lsi_limit, pH_model="calibrated-equilibrium"):
    """Concentration ratio r = CoC_new / CoC_now at which LSI reaches ``lsi_limit``."""
    return 10 ** ((lsi_limit - np.asarray(lsi_now, dtype=float)) / lsi_slope_per_decade(pH_model))


def lsi_status(value: float, limit: float) -> tuple[str, str]:
    """Status label and severity, with bands consistent with the plant's LSI limit."""
    if value > limit:
        return "Scaling risk (above programme limit)", "error"
    if value > 0:
        return "Mildly scale-forming (within programme limit)", "success"
    if value >= -0.5:
        return "Balanced", "success"
    return "Corrosive", "warning"


@dataclass
class EnvelopeSettings:
    lsi_limit: float = 0.5
    skin_temp_c: float = 50.0
    pH_model: str = "calibrated-equilibrium"
    compliance: float = 0.90  # fraction of hours that must stay within the LSI limit
    silica_limit: float = DEFAULT_SILICA_LIMIT
    mg_silica_product: float = DEFAULT_MG_SILICA_PRODUCT
    caso4_product: float = DEFAULT_CASO4_PRODUCT
    chloride_limit: float = CHLORIDE_LIMITS["304 stainless steel"]
    vendor_max_coc: float | None = None  # inhibitor / phosphate solubility limit from vendor


@dataclass
class Envelope:
    max_safe_coc: float
    binding: str
    limits: dict  # name -> max CoC (float) for every constraint that was evaluated
    not_checked: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def hourly_lsi(df: pd.DataFrame, skin_temp_c: float) -> pd.DataFrame:
    """LSI per row at basin temperature and at the heat-exchanger skin temperature."""
    out = pd.DataFrame(index=df.index)
    chem = dict(pH=df["pH"], tds=df["TDS_circ"], calcium=df["calcium_hardness"],
                alkalinity=df["alkalinity"])
    out["lsi_basin"] = lsi(temp_c=df["temp_C"], **chem)
    out["lsi_skin"] = lsi(temp_c=np.full(len(df), float(skin_temp_c)), **chem)
    return out


def envelope(df: pd.DataFrame, lsi_df: pd.DataFrame, s: EnvelopeSettings,
             drift_limited_coc: float | None = None) -> Envelope:
    limits: dict[str, float] = {}
    not_checked: list[str] = []
    notes: list[str] = []

    ok = df["coc"].notna()
    r = ratio_to_reach_lsi(lsi_df.loc[ok, "lsi_skin"], s.lsi_limit, s.pH_model)
    coc_max_rows = df.loc[ok, "coc"] * r
    limits["LSI at heat-exchanger skin"] = float(np.quantile(coc_max_rows, 1 - s.compliance))

    def med(col):
        return float(df[col].median()) if col in df.columns and df[col].notna().any() else None

    si, mg, ca_m, so4 = med("silica_makeup"), med("magnesium_makeup"), med("calcium_makeup"), med("sulfate_makeup")
    cl = med("chloride_makeup")

    if si:
        limits["Silica"] = s.silica_limit / si
    else:
        not_checked.append("silica (add silica_makeup)")
    if si and mg:
        limits["Mg × SiO2 product"] = float(np.sqrt(s.mg_silica_product / (mg * si)))
    else:
        not_checked.append("Mg × SiO2 (add magnesium_makeup + silica_makeup)")
    if ca_m and so4:
        limits["Calcium sulfate"] = float(np.sqrt(s.caso4_product / (ca_m * so4)))
    else:
        not_checked.append("calcium sulfate (add calcium_makeup + sulfate_makeup)")
    if cl:
        limits["Chloride (metallurgy)"] = s.chloride_limit / cl
    else:
        not_checked.append("chloride corrosion (add chloride_makeup)")
    if s.vendor_max_coc:
        limits["Inhibitor / phosphate (vendor)"] = float(s.vendor_max_coc)
    else:
        not_checked.append("inhibitor & phosphate solubility (enter vendor max CoC)")
    if drift_limited_coc is not None and np.isfinite(drift_limited_coc):
        limits["Drift (max achievable)"] = float(drift_limited_coc)

    binding = min(limits, key=limits.get)
    max_coc = limits[binding]

    tds_at_max = float(df.loc[ok, "TDS_circ"].median() * max_coc / df.loc[ok, "coc"].median())
    if tds_at_max > LSI_VALID_MAX_TDS:
        notes.append(f"Circulating TDS at the envelope (~{tds_at_max:,.0f} mg/L) is beyond LSI validity; "
                     "use Stiff-Davis.")
    if max_coc <= 1.0:
        notes.append("No CoC above 1 satisfies the constraints — chemistry inputs are inconsistent "
                     "or the tower cannot run without treatment changes.")
    return Envelope(max_safe_coc=max_coc, binding=binding, limits=limits,
                    not_checked=not_checked, notes=notes)
