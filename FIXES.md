# v0.1 → v0.2: engineering review fixes

This file maps every finding in the v0.1 review to what changed and the test that guards it. Items that need site data or hardware are in [ROADMAP.md](ROADMAP.md).

## A. Critical: the core numbers

| # | Finding | Fix | Test |
|---|---|---|---|
| A1 | Max safe CoC always equalled the current CoC, because the measured LSI was plugged back in where the target belonged | The envelope is solved against the LSI **limit** per hour: CoC_max = CoC·10^((limit − LSI)/slope) | `test_max_safe_coc_is_not_the_current_coc` |
| A1 | Float noise (±1e-14) triggered random scaling banners | Status comparisons use a tolerance. The two values are no longer identical anyway | `pipeline.Analysis.coc_status` |
| A1 | Sample mode mixed two inconsistent inputs (CoC 1.5 vs TDS ratio 2.5) | The separate CoC input is gone. The calculator derives CoC from TDS, and the demo uses the sample CSV | app smoke tests |
| A2 | Max-CoC solve held Ca, alkalinity and pH constant | Ca, alkalinity and TDS scale with CoC. pH rises with alkalinity (calibrated CO₂-equilibrium model) or is held fixed (user choice) | `test_concentration_slope_matches_direct_recalculation`, `test_reviewer_reference_max_coc_with_scaled_chemistry` (≈1.71) |
| A3 | Windage was treated as evaporation | B = E/(CoC−1) − W, M = E·CoC/(CoC−1). Drift is configurable (default 0.10%) | `test_drift_is_not_evaporation` (28.5 vs 51.8) |
| A4 | Savings understated 24× (an m³/h rate summed as m³/day). Fallback branch used other units | All M&V is per-day volumes in m³ | `test_step_change_in_coc_gives_exact_daily_savings_in_m3_per_day` |
| A5 | Flow unit was ambiguous (kg/h vs m³/h, a 1000× difference) | Unit chosen explicitly in the sidebar (default m³/h). Heat load below 100 kW raises a "check units" warning | `test_flow_units_and_heat_load_warning`, `test_evaporation_about_0_86_percent_of_circulation` |
| A6 | "Verified" savings were a model output with no normalisation | Savings are normalised to heat load via baseline water intensity. They are **metered** if `makeup_flow` is logged, otherwise labelled **modelled** in the UI and PDF. ΔT is taken per row from `T_hot`/`T_cold` when logged | `test_load_change_is_not_booked_as_savings`, `test_metered_makeup_is_used_when_present`, `test_measured_delta_t_is_used_per_row` |
| A7 | Prophet was fed a constant series. With under 14 days it trained on random numbers. Savings ratio clamped to [0.5, 0.99] | Prophet removed. Annual figure = mean daily saving × 365 with a bootstrap 95% interval, shown only with ≥ 14 reporting days. No synthetic data, no clamping (losses are shown) | `test_no_projection_without_enough_reporting_days`, `test_worse_operation_shows_negative_savings_unclamped` |
| A7 | Two Prophet fits on every rerun | The whole analysis is one `st.cache_data` call | — |
| A8 | Baseline end at midnight dropped 23 h of the last day | The baseline end date is inclusive and the split is by calendar day | `test_baseline_end_date_is_inclusive` (360 rows) |
| A9 | CoC from TDS, as a ratio of means. Bad readings skewed everything | Tracer priority: conductivity, then chloride, then TDS. Per-reading ratio, time-weighted mean. Hampel spike filter. CoC ≤ 1 readings excluded and counted | `test_conductivity_preferred_as_tracer`, `test_sensor_spike_excluded_from_coc` |

## B. LSI and water-chemistry validity

| Finding | Fix | Test |
|---|---|---|
| LSI only at basin temperature, on averaged inputs | Hourly LSI at both the skin temperature (default 50 °C) and basin temperature. Envelope set so a set share of hours (default 90%) complies. Report shows mean, P95, max and hours above the limit | `test_skin_lsi_higher_than_basin` |
| `+ 273` | `+ 273.15` | `test_lsi_uses_273_15_and_rejects_nonpositive` |
| LSI was the only envelope | Silica, Mg×SiO₂, CaSO₄, chloride by metallurgy, vendor inhibitor/phosphate max CoC and drift-limited CoC added. Binding constraint named. Unchecked constraints listed. Note added above 10,000 mg/L TDS | `test_envelope_reports_binding_and_unchecked_constraints` |
| Target hardcoded at 0.5 | Programme selector (none / phosphonate / high-performance / custom), labelled indicative | app smoke test |
| Status bands inconsistent with target | Bands now follow the limit | `test_lsi_status_bands_follow_limit` |
| Column semantics undefined | Declared in `water_intelligence/data.py` and the README: chemistry columns are circulating water | — |
| Heat balance used sidebar T1/T2, LSI used CSV temp | Heat balance uses the CSV `T_hot`/`T_cold` when present, otherwise basin temp + ΔT/2 | `test_measured_delta_t_is_used_per_row` |
| Latent heat was a nearest-neighbour lookup clamped to 20–50 °C. Density fixed at 1000. Month fixed at 720 h | Steam-table interpolation for 0–100 °C. Temperature-dependent density. Volumes integrate real hours covered and are scaled explicitly "per 30 days" | `test_latent_heat_interpolates_not_nearest`, `test_density_depends_on_temperature`, `test_potential_per_30_days_scales_by_hours_covered` |
| No guards for non-physical values (CoC ≤ 1, log of ≤ 0) | Range checks on load. `ValueError` for CoC ≤ 1 and non-positive chemistry. An infeasible envelope is shown as an error, not a division by zero | `test_coc_must_exceed_one`, `test_infeasible_envelope_is_reported_not_crashed`, `test_app_infeasible_programme_shows_error_not_crash` |

## C. ML and anomaly detection

| Finding | Fix | Test |
|---|---|---|
| `contamination=0.05` forced 5% of rows to be flagged | Detector is trained on the baseline window and thresholded on the baseline score distribution, plus a robust-z check. Only deviations lasting ≥ 3 h become events. A healthy plant reports zero | `test_healthy_plant_can_report_zero_anomalies` |
| No time context; the blowdown sawtooth looked anomalous | Features include CoC, its rate of change and its deviation from the 24 h median, with training on normal cycles | — |
| Only 3 features | Adds Ca, alkalinity, temperature, makeup TDS and skin LSI | `test_sustained_shift_is_learned_and_explained` |
| Not tied to failure modes | Rule-based events: CoC above the envelope (blowdown valve/controller), skin LSI above the limit, flatlined probes (TDS/pH/temperature), calcium cycling below the tracer (precipitation) | `test_flatlined_probe_is_detected`, `test_coc_above_envelope_rule` |
| SHAP over-claimed as a root cause | SHAP removed. Events list the most unusual readings (robust z), explicitly "not a root cause" | — |
| NaN crashed the model; no dtype, sort, duplicate or gap handling; "hours" assumed hourly rows | Data-quality layer in `data.clean`. Hours come from timestamps | `test_nan_text_unsorted_duplicates_and_bad_ranges_are_handled`, `test_gaps_are_not_counted_as_hours` |

## D. Reports and UI

| Finding | Fix |
|---|---|
| PDF's "Hours above LSI 0.5" row was filled with the anomaly count | Separate rows for hours with skin LSI above the limit and for anomaly events (test checks the text) |
| PDF and forecast only reachable in file mode; dead "N/A" code | Available in demo mode, with a DEMO watermark |
| Report period was the baseline window | The report header shows the data period, and M&V shows the baseline and reporting periods separately |
| "Using verified sensor data" was hardcoded | The statement depends on whether savings are metered or modelled, and on whether data-quality exclusions happened |
| Helvetica crashed on ₹ or non-Latin names; long names overflowed; `ln=True` was deprecated | Bundled DejaVu font. Unsupported glyphs are replaced, not raised. Names wrap. `new_x`/`new_y` API |
| Demo injected a synthetic fault and showed it as a finding; `buildup_rate` unused | Synthetic fault removed. The demo runs the real pipeline on the sample CSV and is labelled DEMO |
| Integer-only inputs | All numeric inputs are floats |
| "My Fees" / "Annual Fee to You"; unit-less metrics; duplicates | "Service fee at 20%". Every metric carries a unit. No duplicates |

## E. Engineering, deployment, security

| Finding | Fix |
|---|---|
| Claude calls had no error handling | `ai.narrate` never raises. It falls back to a deterministic summary and shows the error |
| Anyone with the URL could spend the API budget | Button needs `AI_ACCESS_CODE`, with a limit of 5 calls per session |
| No check that the reply uses the given numbers | Every number in the reply must match a computed fact (within rounding), or the reply is replaced. Tested |
| Envelope numbers sent to the model were wrong | The model receives the corrected envelope, binding constraint and unchecked constraints |
| Figures never closed; global `np.random.seed`; heavy reruns | `plt.close(fig)`. Local RNGs only. Cached analysis |
| "Real-time" claim | README says batch CSV analysis. Ingestion is on the ROADMAP |
| Unpinned requirements; no tests, CI or schema | Pinned `requirements.txt`. 54 pytest tests. GitHub Actions (ruff + pytest on 3.12/3.13). Schema documented |

## F. Self-audit after v0.2 (errors in my own first pass)

I re-checked v0.2 against the literature and against independent simulations. Five things were wrong or overstated.

| Issue | Effect | Fix | Test |
|---|---|---|---|
| Steady-state M = E·CoC/(CoC−1) applied to batch blowdown (sawtooth CoC) | Makeup overstated ~19% on a mass-conserving simulation | Dumps detected automatically. Per-cycle solids balance with the dump leaving at peak CoC | `test_batch_blowdown_matches_mass_balance_simulation` (±2%), `test_continuous_bleed_matches_mass_balance_simulation` (±1%) |
| Evaporation share defaulted to 1.0 and was described as "conservative" | It was the opposite: evaporation, and therefore savings, were ~15% too high compared with the standard 0.00153·ΔT·R rule | Default 0.85, help text corrected | `test_evaporation_matches_standard_rule_of_thumb` |
| pH-vs-alkalinity slope 1 was called a "conservative upper bound" | The Puckorius correlation has slope 1.465, so slope 1 isn't an upper bound | Named models: Kunz (1.0, default), Puckorius (1.465, conservative when raising CoC), fixed | `test_concentration_slope_matches_direct_recalculation` |
| Envelope took a quantile of per-hour max CoC | This mixes hours at different points in the sawtooth and doesn't answer "how far can the plant raise CoC" | Envelope = mean CoC × the weighted quantile of the per-hour allowed scale-up | `test_envelope_is_mean_coc_times_compliant_scale_up`, `test_envelope_matches_brute_force_search` |
| Rolling CoC window spanned the baseline/reporting boundary | Smeared a real step change into the wrong period | Window is split at the baseline end | `test_rolling_coc_does_not_leak_across_baseline_boundary` |
| Potential compared metered current makeup with modelled optimised makeup | Model bias would be booked as savings | Modelled vs modelled. The meter is reported as a model check (warning if off by >15%) | `test_potential_compares_model_with_model_even_when_metered` |
| i.i.d. bootstrap on autocorrelated daily savings | Confidence interval too narrow | Moving-block bootstrap (block ≥ CoC window / blowdown cycle) | `test_block_bootstrap_is_wider_for_autocorrelated_savings` |
| Load "normalisation" with a constant ΔT only normalises flow | Overstated what M&V controls for | Disclosed in UI and PDF when `T_hot`/`T_cold` aren't logged | `test_unnormalised_load_is_disclosed` |
| "CoC above envelope" rule fired on normal sawtooth peaks | False alarms | Compares the 24 h mean CoC (the envelope is a limit on the average) | `test_normal_blowdown_sawtooth_is_not_an_envelope_fault` |

Known approximation that remains: skin-temperature LSI uses the pH measured at basin temperature. This is the industry convention, but it slightly overstates LSI at the skin.
