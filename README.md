# 💧 Water Intelligence Dashboard

A real-time water optimisation platform for industrial cooling towers — built for Indian chemical and pharma manufacturers.

## 🔗 Live Demo
👉 https://water-intelligence.streamlit.app

Upload `sample_cooling_tower_data.csv` to see it in action.

## What It Does

**Layer 1 — Data Ingestion**
Accepts plant sensor data via CSV upload or manual sidebar inputs. Validates column structure on upload.

**Layer 2 — Engineering Formulas**
- Calculates Cycles of Concentration (CoC) from conductivity sensor data
- Derives maximum safe CoC from Langelier Saturation Index (LSI) — physics-based, not hardcoded
- Computes current vs optimised blowdown rate and makeup water consumption
- Verifies savings against a user-defined baseline period with cumulative savings chart

**Layer 3 — AI Intelligence**
- Plain-English operator recommendations via Claude API — explains what the formulas computed, never invents numbers
- Isolation Forest anomaly detection — learns each plant's normal pattern and flags deviations
- SHAP explainability — identifies which sensor (TDS, pH, or flow rate) caused each anomaly
- Scaling risk alerts when current CoC exceeds safe limit

**Layer 4 — Output Surfaces**
- Live operator dashboard with real-time metrics
- Automated monthly PDF report for the CFO — includes water balance summary, savings, LSI risk, anomaly summary with dominant cause, and verification statement
- AI-generated recommendation on demand

## Safety Architecture
AI output is always grounded in Layer 2 formula outputs. Claude acts as a translator, not a decision-maker. Every recommendation has a visible numerical basis. SHAP values explain the reason behind every anomaly alert.

## How To Run Locally
```
pip install -r requirements.txt
streamlit run cooling_tower_dashboard.py
```

## Required CSV Format
Columns: `date, TDS_circ, TDS_makeup, pH, temp_C, flow_rate, calcium_hardness, alkalinity`

## Tech Stack
Python · Streamlit · Pandas · NumPy · Scikit-learn · SHAP · fpdf2 · Anthropic Claude API
