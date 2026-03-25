# 💧 Water Intelligence Dashboard

A real-time water optimization platform for industrial cooling towers — built for Indian chemical and pharma manufacturers.

## 🔗 Live Demo
👉 https://water-intelligence.streamlit.app

Upload `sample_cooling_tower_data.csv` to see it in action.

## What It Does

**Layer 1 — Data ingestion**
Accepts plant sensor data via CSV upload or manual sidebar inputs.

**Layer 2 — Engineering formulas**
- Calculates Cycles of Concentration (CoC) from conductivity sensor data
- Derives maximum safe CoC from Langelier Saturation Index (LSI) — physics-based, not hardcoded
- Computes current vs optimised blowdown rate and makeup water consumption
- Verifies savings against a user-defined baseline period

**Layer 3 — AI intelligence**
- Plain-English operator recommendations via Claude API — explains what the formulas computed, never invents numbers
- Real-time blowdown anomaly detection

**Layer 4 — Output surfaces**
- Live operator dashboard
- Automated monthly PDF report for the CFO
- AI-generated recommendation on demand

## Safety Architecture
AI output is always grounded in Layer 2 formula outputs. Claude acts as a translator, not a decision-maker. Every recommendation has a visible numerical basis.

## How To Run Locally
pip install -r requirements.txt
streamlit run cooling_tower_dashboard.py

## Required CSV Format
Columns: date, TDS_circ, TDS_makeup, pH, temp_C, flow_rate, calcium_hardness, alkalinity

## Tech Stack
Python · Streamlit · Pandas · NumPy · fpdf2 · Anthropic Claude API

.env
__pycache__/
*.pyc
