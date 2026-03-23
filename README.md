# Cooling Tower Water Intelligence Dashboard

A real-time water optimization dashboard for industrial cooling towers — CoC monitoring, scaling risk (LSI), anomaly detection, and verified savings calculation

## Live Demo
👉 https://water-intelligence.streamlit.app/

Upload the sample CSV (`sample_cooling_tower_data.csv`) to see it in action.

## What it does
- Calculates Cycles of Concentration (CoC) from sensor data
- Monitors scaling risk using the Langelier Saturation Index (LSI)
- Detects blowdown anomalies
- Calculates verified water and cost savings vs baseline period

## How to run
pip install streamlit pandas numpy
streamlit run cooling_tower_dashboard.py

## Upload your plant data
CSV format required: date, TDS_circ, TDS_makeup, pH, temp_C, flow_rate, calcium_hardness, alkalinity
