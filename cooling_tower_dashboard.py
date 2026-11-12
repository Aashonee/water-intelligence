"""Water Intelligence — Streamlit UI.

All engineering lives in the ``water_intelligence`` package; this file only collects
inputs, calls ``pipeline.run`` (cached) and displays results.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from report_generator import generate_pdf_report  # noqa: E402
from water_intelligence import ai, chemistry, data, pipeline, water_balance  # noqa: E402

load_dotenv()
HERE = Path(__file__).parent
SAMPLE_CSV = HERE / "sample_cooling_tower_data.csv"
AI_CALLS_PER_SESSION = 5

st.set_page_config(page_title="Water Intelligence", page_icon="💧", layout="wide")


def secret(name: str) -> str | None:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


@st.cache_data(show_spinner="Running analysis…")
def run_analysis(csv_bytes: bytes, settings: pipeline.Settings) -> pipeline.Analysis:
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    return pipeline.run(raw, settings)


def fmt(x, spec=",.0f", none="—"):
    return none if x is None or not np.isfinite(x) else format(x, spec)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("💧 Water Intelligence")
mode = st.sidebar.radio("Mode", ["Demo dataset", "Upload CSV", "Quick calculator (one reading)"])

uploaded = None
if mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Plant data (CSV)", type="csv")
    with st.sidebar.expander("CSV format"):
        st.markdown(data.__doc__)

st.sidebar.header("Plant")
plant_name = st.sidebar.text_input("Plant name", value="Demo Cooling Tower Plant")
contact_name = st.sidebar.text_input("Report prepared for", value="Plant Manager")
contact_email = st.sidebar.text_input("Contact email", value="manager@plant.example")

st.sidebar.header("Site constants")
flow_unit = st.sidebar.selectbox("Recirculation flow unit (CSV column flow_rate)", list(data.FLOW_UNITS), index=0)
delta_t = st.sidebar.number_input("Tower range ΔT (°C) — used if T_hot/T_cold not logged",
                                  min_value=0.5, max_value=25.0, value=5.0, step=0.5)
evap_fraction = st.sidebar.number_input("Share of heat removed by evaporation", 0.5, 1.0, 1.0, 0.05,
                                        help="1.0 = all latent (conservative for savings). Typically 0.8–1.0.")
drift_pct = st.sidebar.number_input("Drift loss (% of recirculation)", 0.0, 0.5, 0.10, 0.01,
                                    help="Modern eliminators ≤0.005%; older towers 0.05–0.2%.")
water_cost = st.sidebar.number_input("Water cost (₹/m³)", 0.0, 10_000.0, 50.0, 5.0)
fee_share = st.sidebar.number_input("Pay-as-you-save fee share", 0.0, 1.0, 0.20, 0.05)

st.sidebar.header("Chemistry envelope")
prog_names = list(chemistry.TREATMENT_PROGRAMMES) + ["Custom"]
programme = st.sidebar.selectbox("Scale-inhibitor programme", prog_names, index=1,
                                 help="Indicative limits. Use the number your treatment vendor gives you.")
if programme == "Custom":
    lsi_limit = st.sidebar.number_input("Max LSI at skin temperature", -1.0, 3.5, 1.0, 0.1)
else:
    lsi_limit = chemistry.TREATMENT_PROGRAMMES[programme]
    st.sidebar.caption(f"LSI limit: {lsi_limit:+.1f} at skin temperature")
skin_temp = st.sidebar.number_input("Heat-exchanger skin temperature (°C)", 20.0, 90.0, 50.0, 1.0)
ph_model = st.sidebar.selectbox(
    "pH as CoC rises", chemistry.PH_MODELS, index=0,
    format_func={"calibrated-equilibrium": "Rises with alkalinity (CO₂ stripping, conservative)",
                 "fixed": "Held at measured value (optimistic)"}.get)
compliance = st.sidebar.slider("Hours that must stay within LSI limit", 0.5, 1.0, 0.90, 0.01)
metallurgy = st.sidebar.selectbox("Wetted metallurgy (chloride limit)", list(chemistry.CHLORIDE_LIMITS))
vendor_max = st.sidebar.number_input("Vendor max CoC for inhibitor / phosphate (0 = not set)", 0.0, 20.0, 0.0, 0.1)
silica_limit = st.sidebar.number_input("Silica limit (mg/L SiO₂)", 50.0, 300.0, 150.0, 10.0)

bal = water_balance.BalanceSettings(delta_t_c=delta_t, evap_fraction=evap_fraction, drift_pct=drift_pct)
envs = chemistry.EnvelopeSettings(
    lsi_limit=lsi_limit, skin_temp_c=skin_temp, pH_model=ph_model, compliance=compliance,
    silica_limit=silica_limit, chloride_limit=chemistry.CHLORIDE_LIMITS[metallurgy],
    vendor_max_coc=vendor_max or None,
)

tab_ct, tab_boiler = st.tabs(["🌊 Cooling Tower", "🔥 Boiler Feedwater"])
with tab_boiler:
    st.info("Boiler feedwater module — not built yet. See ROADMAP.md.")

# ---------------------------------------------------------------------------
# Quick calculator
# ---------------------------------------------------------------------------
if mode.startswith("Quick"):
    with tab_ct:
        st.title("Quick calculator — one set of readings")
        st.caption("Steady-state estimate from a single reading. CoC is derived from the TDS ratio. "
                   "Upload a CSV for hourly LSI, anomaly events and savings verification.")
        c1, c2, c3 = st.columns(3)
        tds_c = c1.number_input("Circulating TDS (mg/L)", 1.0, 50_000.0, 500.0, 10.0)
        tds_m = c1.number_input("Makeup TDS (mg/L)", 1.0, 20_000.0, 200.0, 10.0)
        ph = c2.number_input("Circulating pH", 4.0, 11.0, 7.5, 0.05)
        temp = c2.number_input("Basin temperature (°C)", 0.0, 70.0, 34.5, 0.5)
        ca = c3.number_input("Calcium hardness (mg/L as CaCO₃)", 1.0, 5000.0, 300.0, 5.0)
        alk = c3.number_input("M-alkalinity (mg/L as CaCO₃)", 1.0, 5000.0, 200.0, 5.0)
        flow = c1.number_input(f"Recirculation flow ({flow_unit})", 0.001, 1e6, 3300.0, 10.0)
        try:
            r = pipeline.single_point(tds_c, tds_m, ph, temp, flow * data.FLOW_UNITS[flow_unit], ca, alk, bal, envs)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        msg = data.heat_load_check(r.heat_kw)
        if msg:
            st.warning(msg)
        m = st.columns(4)
        m[0].metric("CoC", f"{r.coc:.2f}")
        m[1].metric("Max safe CoC", f"{r.max_safe_coc:.2f}" if r.max_safe_coc > 1 else "none")
        m[2].metric("Skin LSI", f"{r.lsi_skin:+.2f}", help=f"Basin LSI {r.lsi_basin:+.2f}")
        m[3].metric("Heat rejected", f"{r.heat_kw:,.0f} kW")
        m = st.columns(4)
        m[0].metric("Evaporation", f"{r.evap_m3h:,.2f} m³/h")
        m[1].metric("Blowdown now", f"{max(r.blowdown_now_m3h, 0):,.2f} m³/h")
        m[2].metric("Makeup now", f"{r.makeup_now_m3h * 720:,.0f} m³ / 30 d")
        if r.makeup_opt_m3h is not None:
            d = (r.makeup_now_m3h - r.makeup_opt_m3h) * 720
            m[3].metric("Change at max safe CoC", f"{d:,.0f} m³ / 30 d", help=f"₹{d * water_cost:,.0f} per 30 days")
        if r.coc > r.max_safe_coc * (1 + 1e-6):
            st.error(f"🚨 CoC {r.coc:.2f} is above the safe limit {r.max_safe_coc:.2f} — increase blowdown.")
        st.caption("Only LSI, drift and the vendor limit are checked here. Silica, chloride, Mg×SiO₂ and "
                   "CaSO₄ need makeup analyses — upload a CSV with those columns.")
    st.stop()

# ---------------------------------------------------------------------------
# Data modes
# ---------------------------------------------------------------------------
demo = mode == "Demo dataset"
if demo:
    csv_bytes = SAMPLE_CSV.read_bytes()
elif uploaded is None:
    with tab_ct:
        st.title("Upload plant data")
        st.info("Upload a CSV in the sidebar. Expand **CSV format** there for the columns.")
    st.stop()
else:
    csv_bytes = uploaded.getvalue()

# Baseline defaults come from the data's date span.
try:
    _dates = pd.to_datetime(pd.read_csv(io.BytesIO(csv_bytes), usecols=["date"])["date"], errors="coerce").dropna()
except ValueError:
    _dates = pd.Series(dtype="datetime64[ns]")
if _dates.empty:
    with tab_ct:
        st.error("The file has no readable `date` column.")
    st.stop()
_lo, _hi = _dates.min().date(), _dates.max().date()
_bs, _be = pipeline.default_baseline(pd.DataFrame({"date": _dates}))
st.sidebar.header("Savings baseline")
bs = st.sidebar.date_input("Baseline start", value=_bs, min_value=_lo, max_value=_hi)
be = st.sidebar.date_input("Baseline end (inclusive)", value=_be, min_value=_lo, max_value=_hi)
coc_window = st.sidebar.number_input("CoC averaging window for water balance (days)", 1, 30, 7)

settings = pipeline.Settings(
    flow_unit=flow_unit, balance=bal, envelope=envs, baseline_start=bs, baseline_end=be,
    coc_window_days=int(coc_window), water_cost=water_cost, fee_share=fee_share,
)

with tab_ct:
    try:
        A = run_analysis(csv_bytes, settings)
    except (data.DataError, ValueError) as e:
        st.error(f"❌ {e}")
        st.stop()

    st.title(plant_name)
    if demo:
        st.info("📊 **Demo dataset** (`sample_cooling_tower_data.csv`, synthetic). "
                "Figures are illustrative, not a plant result.")

    dq_title = f"Data quality — {A.quality.rows_used} of {A.quality.rows_in} rows used"
    with st.expander(dq_title, expanded=bool(A.warnings)):
        for line in A.quality.summary_lines():
            st.write("• " + line)
    for w in A.warnings:
        st.warning(w)

    # --- Envelope -----------------------------------------------------------
    env = A.env
    if A.coc_status == "infeasible":
        st.error("🚨 No CoC above 1 satisfies the chemistry limits with these settings. "
                 "Review the treatment programme / inputs with your water-treatment vendor.")
    elif A.coc_status == "above":
        st.error(f"🚨 SCALING RISK — current CoC {A.current_coc:.2f} is above the safe limit "
                 f"{env.max_safe_coc:.2f} ({env.binding}). Increase blowdown.")
    else:
        st.success(f"Current CoC {A.current_coc:.2f} is within the safe limit {env.max_safe_coc:.2f} "
                   f"(binding: {env.binding}).")

    pot = A.potential
    c = st.columns(4)
    c[0].metric("Current CoC (time-weighted)", f"{A.current_coc:.2f}")
    c[1].metric("Max safe CoC", f"{env.max_safe_coc:.2f}" if env.max_safe_coc > 1 else "none")
    c[2].metric("Makeup now (m³ / 30 days)", fmt(pot.current_m3_per_30d))
    c[3].metric("Makeup at max safe CoC (m³ / 30 days)", fmt(pot.optimised_m3_per_30d))
    c = st.columns(3)
    c[0].metric("Potential change (m³ / 30 days)", fmt(pot.savings_m3_per_30d),
                help="Model estimate. Negative = more water needed to be safe.")
    c[1].metric("Potential change (₹ / 30 days)", fmt(pot.savings_m3_per_30d * water_cost))
    c[2].metric(f"Service fee at {fee_share:.0%} (₹ / 30 days)",
                fmt(max(pot.savings_m3_per_30d, 0) * water_cost * fee_share) if pot.feasible else "—")

    with st.expander("Envelope: every constraint"):
        st.dataframe(pd.DataFrame({"Max CoC": env.limits}).sort_values("Max CoC").style.format("{:.2f}"))
        if env.not_checked:
            st.warning("Not checked (data missing): " + "; ".join(env.not_checked))
        for n in env.notes:
            st.warning(n)

    st.subheader("Cycles of concentration")
    trend = pd.DataFrame({"CoC": A.df["coc"].to_numpy(),
                          "Max safe CoC": env.max_safe_coc if env.max_safe_coc > 1 else np.nan},
                         index=A.df["date"])
    st.line_chart(trend)

    # --- LSI ----------------------------------------------------------------
    st.subheader("Scaling risk — Langelier index (hourly)")
    sk = A.lsi["lsi_skin"]
    label, sev = chemistry.lsi_status(float(sk.quantile(0.95)), lsi_limit)
    getattr(st, sev)(f"95th-percentile skin LSI {sk.quantile(0.95):+.2f}: {label}")
    c = st.columns(4)
    c[0].metric("Skin LSI mean", f"{sk.mean():+.2f}")
    c[1].metric("Skin LSI max", f"{sk.max():+.2f}")
    c[2].metric("Basin LSI mean", f"{A.lsi['lsi_basin'].mean():+.2f}")
    c[3].metric("Hours above limit", f"{A.lsi_skin_hours_above:,.0f}")
    st.line_chart(pd.DataFrame({"Skin LSI": sk.to_numpy(), "Basin LSI": A.lsi["lsi_basin"].to_numpy(),
                                "Limit": lsi_limit}, index=A.df["date"]))

    with st.expander("Water balance details"):
        hb = A.hourly
        cur_b = water_balance.blowdown_m3h(hb["evap_m3h"].mean(), hb["drift_m3h"].mean(), A.current_coc)
        c = st.columns(4)
        c[0].metric("Heat rejected (mean)", f"{hb['heat_kw'].mean():,.0f} kW")
        c[1].metric("Evaporation (mean)", f"{hb['evap_m3h'].mean():,.2f} m³/h")
        c[2].metric("Drift", f"{hb['drift_m3h'].mean():,.2f} m³/h")
        c[3].metric("Blowdown at current CoC", f"{max(cur_b, 0):,.2f} m³/h")
        st.caption(f"ΔT basis: {hb['delta_t_source'].iat[0]}. B = E/(CoC−1) − W, M = E·CoC/(CoC−1).")

    # --- M&V ----------------------------------------------------------------
    st.subheader("Savings vs baseline")
    mv = A.mv
    st.caption(f"Basis: **{mv.basis}**. Baseline {A.baseline[0]} → {A.baseline[1]} (inclusive), "
               "normalised to heat load via water intensity (m³ makeup per m³ evaporated).")
    for m in mv.messages:
        st.caption("• " + m)
    if len(mv.daily):
        st.line_chart(mv.daily[["cumulative_m3"]].rename(columns={"cumulative_m3": "Cumulative saving (m³)"}))
        c = st.columns(4)
        c[0].metric("Water intensity: baseline → reporting",
                    f"{mv.baseline_intensity:.3f} → {mv.reporting_intensity:.3f}")
        c[1].metric("Saving to date (m³)", fmt(mv.total_savings_m3))
        c[2].metric("Saving to date (₹)", fmt(mv.total_savings_m3 * water_cost))
        if np.isfinite(mv.annual_m3):
            lo, hi = mv.annual_ci_m3
            c[3].metric("Annualised (m³)", fmt(mv.annual_m3), help=f"95% interval {lo:,.0f} to {hi:,.0f} m³")
        if not mv.basis.startswith("metered"):
            st.warning("No makeup meter in the data: these savings are modelled, not measured. "
                       "Add a `makeup_flow` column (m³/h) for verifiable savings.")

    # --- Anomalies ----------------------------------------------------------
    st.subheader("Anomaly events")
    an = A.anomalies
    st.caption(f"Learned detector trained on the {an.trained_on}; only deviations lasting ≥ 3 h "
               "(rules: ≥ 6 h) become events. Signals listed are the most unusual readings, not a root cause.")
    if len(an.events):
        st.dataframe(an.events, hide_index=True, use_container_width=True)
    else:
        st.success("No sustained anomalies detected.")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(A.df["date"], A.df["coc"], ".", ms=3, alpha=0.6, label="CoC")
    for _, e in an.events.iterrows():
        ax.axvspan(e["start"], e["end"], color="red", alpha=0.15)
    if env.max_safe_coc > 1:
        ax.axhline(env.max_safe_coc, color="k", ls="--", lw=1, label="Max safe CoC")
    ax.set_ylabel("CoC")
    ax.legend(loc="upper left")
    st.pyplot(fig)
    plt.close(fig)

    # --- AI narration -------------------------------------------------------
    st.subheader("🤖 Plain-English explanation")
    facts = {
        "Current CoC": f"{A.current_coc:.2f}",
        "Max safe CoC": f"{env.max_safe_coc:.2f}" if env.max_safe_coc > 1 else "none feasible",
        "Binding constraint": env.binding,
        "Constraints not checked": "; ".join(env.not_checked) or "none",
        "Makeup now (m3 per 30 days)": fmt(pot.current_m3_per_30d),
        "Makeup at max safe CoC (m3 per 30 days)": fmt(pot.optimised_m3_per_30d),
        "Potential change (m3 per 30 days)": fmt(pot.savings_m3_per_30d),
        "Potential change (Rs per 30 days)": fmt(pot.savings_m3_per_30d * water_cost),
        "Skin LSI 95th percentile": f"{sk.quantile(0.95):+.2f}",
        "LSI limit": f"{lsi_limit:+.2f}",
        "Hours above LSI limit": fmt(A.lsi_skin_hours_above),
        "Anomaly events": str(len(an.events)),
        "Savings basis": mv.basis,
    }
    access_code = secret("AI_ACCESS_CODE")
    calls = st.session_state.setdefault("ai_calls", 0)
    if not access_code:
        st.caption("AI explanation is disabled: set `AI_ACCESS_CODE` and `ANTHROPIC_API_KEY` in secrets.")
    else:
        code = st.text_input("Access code", type="password")
        disabled = code != access_code or calls >= AI_CALLS_PER_SESSION
        if calls >= AI_CALLS_PER_SESSION:
            st.caption("Session limit reached.")
        if st.button("Explain these results", disabled=disabled):
            st.session_state["ai_calls"] = calls + 1
            with st.spinner("Asking Claude…"):
                n = ai.narrate(facts, secret("ANTHROPIC_API_KEY"))
            if n.error:
                st.warning(n.error + (f" Unsupported numbers: {n.unsupported}" if n.unsupported else ""))
            st.info(n.text)

    # --- PDF ----------------------------------------------------------------
    st.subheader("Report")
    try:
        pdf = generate_pdf_report(A, plant_name, contact_name, contact_email, demo=demo)
        p0, p1 = A.period
        st.download_button("📄 Download PDF report", data=pdf, file_name=f"water_report_{p0}_{p1}.pdf",
                           mime="application/pdf")
    except Exception as e:  # never take the dashboard down because of the PDF
        st.error(f"PDF could not be generated: {e}")
