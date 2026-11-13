# Roadmap

These items are capability gaps, not code bugs. They came out of the v0.1 engineering review. Each one needs site data, hardware or a design decision, so they aren't solved by editing code.

## Before a paid pilot (pay-as-you-save contract)
- [ ] **Makeup and blowdown flow meters on site.** Savings stay *modelled* until `makeup_flow` is logged. A fee based on a share of savings should be billed only on metered savings.
- [ ] **Agree the M&V plan with the customer in writing.** Cover the baseline window, how heat load is normalised, how non-routine adjustments are handled, and dispute resolution. Follow IPMVP Option B/C wording.
- [ ] **Get the chemistry limits from the plant's treatment vendor.** These are the LSI limit for their inhibitor, the max CoC for phosphate and inhibitor solubility, and the metallurgy (which sets the chloride limit). The app's defaults are indicative only.
- [ ] **Calibrate the pH-vs-CoC model with site data.** Step the CoC up in a controlled way, log pH, and fit the slope. The current default assumes CO₂ equilibrium, which is the conservative upper bound.
- [ ] **Validate the heat balance against the metered makeup** over at least 2 weeks. Fit the evaporation fraction (sensible vs latent heat) per site.

## Data ingestion (the "real-time" claim)
- [ ] Ingest from the historian or PLC: OPC-UA and Modbus TCP, or a Raspberry Pi gateway writing to TimescaleDB.
- [ ] Store data persistently, with per-plant baselines saved across sessions.
- [ ] Send alarms (SMS, WhatsApp or email) for sustained anomaly events, with acknowledgement and suppression.

## Analytics
- [ ] Compute LSI from the logged heat-exchanger outlet or skin temperature instead of a site constant.
- [ ] Add Stiff-Davis / Ryznar / Puckorius indices for high-TDS water (LSI isn't valid above about 10,000 mg/L).
- [ ] Detect heat-exchanger leaks and process contamination. This needs process-side signatures, e.g. TOC, ammonia or makeup-TDS drift.
- [ ] Add seasonal forecasting (Prophet or similar) only once there are at least 12 months of data. v0.1 fitted Prophet to a constant series and to synthetic data, so it was removed.
- [ ] Add a second sample dataset with the optional columns (meter, silica, conductivity) so the demo can show metered M&V.

## Product
- [ ] Boiler feedwater module.
- [ ] Add a Devanagari font to the PDF. Characters the bundled DejaVu font can't render currently print as "?" rather than crashing.
- [ ] Agent and tool-calling narration, with an evaluation harness that scores numeric faithfulness and compliance with the safety envelope. The number check in `water_intelligence/ai.py` is the first piece of this.
