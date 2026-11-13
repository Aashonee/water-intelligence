# 💧 Water Intelligence: cooling-tower water optimisation

A physics-first tool for open recirculating cooling towers at Indian chemical and pharma plants. It reads plant data and works out:

- **Cycles of concentration (CoC)** from a conservative tracer (conductivity, then chloride, then TDS), per reading.
- **The safe operating envelope**: the highest CoC that keeps every checked constraint within limits. Constraints are the Langelier index at the heat-exchanger skin, silica, Mg×SiO₂, calcium sulfate, chloride vs metallurgy, the vendor's inhibitor limit, and drift. The binding constraint is named, and constraints with no data are listed as **not checked**.
- **Water balance** from a heat balance: evaporation, drift, blowdown and makeup, with B = E/(CoC−1) − W.
- **Savings vs a baseline, normalised for heat load** when `T_hot`/`T_cold` are logged (otherwise only for flow, and the app says so). These are *metered* if a makeup meter is logged and *modelled* if not, and the label always says which.
- **Anomaly events**: physics rules for known failure modes, plus a detector trained on the baseline. An event is raised only when a deviation persists.
- **A PDF report** and an optional plain-English explanation from Claude, which is rejected if it contains any number the engine didn't compute.

> **Status: pre-pilot.** This is batch analysis of uploaded CSVs, not a real-time control system. It has not yet been validated against a plant's metered makeup. See [ROADMAP.md](ROADMAP.md).
>
> v0.2 fixes the 40+ issues from an engineering review of v0.1. [FIXES.md](FIXES.md) maps every finding to its fix and test.

## Live demo
👉 https://water-intelligence.streamlit.app. Choose **Demo dataset** in the sidebar.

## Run locally
```bash
pip install -r requirements.txt
streamlit run cooling_tower_dashboard.py
```
The AI explanation is optional. To enable it, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `ANTHROPIC_API_KEY` and `AI_ACCESS_CODE`. The access code and a per-session call limit stop anyone with the public URL from spending your API budget.

## Modes
| Mode | Use it for |
|---|---|
| Demo dataset | The bundled synthetic 30-day file (results are labelled DEMO everywhere) |
| Upload CSV | Your plant's data |
| Quick calculator | One set of readings. CoC comes from the TDS ratio, so it can't contradict the chemistry |

## CSV format
Required columns (one row per reading, any regular interval):

| column | meaning |
|---|---|
| `date` | timestamp |
| `TDS_circ`, `TDS_makeup` | circulating / makeup TDS, mg/L |
| `pH`, `temp_C` | **circulating** water pH and basin temperature (°C) |
| `flow_rate` | recirculation flow; **choose the unit in the sidebar** (default m³/h) |
| `calcium_hardness`, `alkalinity` | **circulating** water, mg/L as CaCO₃ |

Optional columns that unlock more of the analysis:

| column | unlocks |
|---|---|
| `cond_circ`, `cond_makeup` (µS/cm) or `chloride_circ`, `chloride_makeup` | a better CoC tracer than TDS |
| `T_hot`, `T_cold` (°C) | per-reading heat load instead of a constant ΔT |
| `makeup_flow` (m³/h) | **metered** savings (billable) instead of modelled |
| `silica_makeup`, `magnesium_makeup`, `calcium_makeup`, `sulfate_makeup`, `chloride_makeup` | silica, Mg×SiO₂, CaSO₄, chloride limits; calcium-cycling scale check |

The app cleans every upload: it coerces types, sorts by time, removes duplicate timestamps, detects logging gaps, drops readings outside physical ranges and filters sensor spikes. It reports what was removed.

## How the numbers are made
- **LSI** is computed hour by hour at the skin temperature (default 50 °C) and at the basin temperature. The limit comes from the scale-inhibitor programme. The defaults are indicative only, and the real limit should come from your treatment vendor.
- **Max safe CoC.** Raising CoC by a factor *r* concentrates Ca, alkalinity and TDS by *r*. pH rises with alkalinity along a Kunz curve anchored to the measured pH (slope 1). You can switch to the steeper Puckorius slope (1.465, more conservative when raising CoC) or hold pH fixed (only valid with acid feed). So LSI(r) = LSI₀ + (1.9 + slope)·log₁₀ r. The envelope is the mean CoC × the largest scale-up that keeps 90% of hours (adjustable) within the limit, capped by every other constraint. A brute-force search confirms it.
- **Evaporation** E = f·m·Cp·ΔT / λ(T), with λ interpolated from steam tables and density from temperature. The default f = 0.85 reproduces the standard rule E ≈ 0.00153·ΔT(°C)·R. Raising f inflates savings.
- **Blowdown mode is detected automatically.** For continuous bleed, M = E·CoC/(CoC−1). For batch dumps (a sawtooth CoC), each cycle gets its own solids balance, with the dump leaving at the peak CoC. Both are checked against a mass-conserving tower simulation and agree within 1–2%. Applying the steady-state formula to a sawtooth overstated makeup by about 19%.
- **Savings.** Baseline water intensity k = makeup / evaporation. The daily saving in the reporting period is k·E_day − makeup_day. The annual figure is the mean daily saving × 365 with a moving-block bootstrap 95% interval (days are correlated), and it has no seasonality. The potential saving compares modelled with modelled, so model error cancels. A makeup meter, if present, is reported as a check on the model. Negative savings are shown, not clamped.

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q        # 54 tests: physics regressions, simulation cross-checks, M&V, data quality, anomaly, AI guard, PDF, app smoke
```
CI runs ruff and pytest on Python 3.12 and 3.13 for every push and pull request.

## Layout
```
water_intelligence/   engine (properties, data, chemistry, water_balance, savings, anomaly, ai, pipeline)
cooling_tower_dashboard.py   Streamlit UI (thin)
report_generator.py          PDF
tests/                       pytest suite
fonts/                       DejaVu Sans (Unicode PDF output; licence included)
```

## Tech stack
Python · Streamlit · pandas · NumPy · scikit-learn · fpdf2 · Anthropic Claude API
