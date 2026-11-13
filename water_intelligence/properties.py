"""Physical properties of liquid water, valid roughly 0-100 °C."""

from __future__ import annotations

import numpy as np

CP_WATER = 4.18  # kJ/(kg·K), specific heat of liquid water (varies <1% over 0-60 °C)

# Saturated-steam-table latent heat of vaporisation, kJ/kg (IAPWS-IF97 values).
_LATENT_T = np.array([0, 10, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100], dtype=float)
_LATENT_H = np.array(
    [2500.9, 2477.2, 2453.5, 2441.7, 2429.8, 2417.9, 2406.0, 2394.0, 2382.0,
     2357.7, 2333.0, 2308.0, 2282.5, 2256.4],
    dtype=float,
)


def latent_heat_kj_per_kg(temp_c):
    """Latent heat of vaporisation at ``temp_c`` (°C), linearly interpolated.

    Replaces the old nearest-neighbour lookup that was clamped to 20-50 °C.
    """
    t = np.asarray(temp_c, dtype=float)
    if np.any((t < 0) | (t > 100)):
        raise ValueError("Latent heat table is valid for 0-100 °C only.")
    out = np.interp(t, _LATENT_T, _LATENT_H)
    return float(out) if out.ndim == 0 else out


def water_density_kg_per_m3(temp_c):
    """Density of pure water (kg/m³), Tanaka et al. (2001) correlation, 0-40 °C ±0.001%,
    and within ~0.1% up to 100 °C. Replaces the hardcoded 1000 kg/m³."""
    t = np.asarray(temp_c, dtype=float)
    a1, a2, a3, a4, a5 = -3.983035, 301.797, 522528.9, 69.34881, 999.974950
    rho = a5 * (1 - (t + a1) ** 2 * (t + a2) / (a3 * (t + a4)))
    return float(rho) if rho.ndim == 0 else rho
