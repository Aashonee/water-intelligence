"""PDF report for the plant / CFO.

Fixes versus v0.1: Unicode font (₹, accented names no longer crash), long names wrap,
the report period is the actual reporting window, the anomaly and LSI-exceedance rows
are separate and correctly labelled, the verification statement depends on whether
savings are metered or modelled, and the data-quality basis is printed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT_DIR = Path(__file__).parent / "fonts"
NAVY = (10, 36, 99)
GREY = (245, 245, 245)
BLACK = (0, 0, 0)
RED, AMBER, GREEN = (180, 30, 30), (200, 100, 0), (30, 130, 60)


def _fmt(x, spec=",.0f", none="n/a"):
    try:
        if x is None or not np.isfinite(x):
            return none
    except TypeError:
        return none
    return format(x, spec)


class _Report(FPDF):
    def __init__(self, footer_text: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.footer_text = footer_text
        self.unicode = (FONT_DIR / "DejaVuSans.ttf").exists()
        if self.unicode:
            self.add_font("Body", "", str(FONT_DIR / "DejaVuSans.ttf"))
            self.add_font("Body", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
            self.add_font("Body", "I", str(FONT_DIR / "DejaVuSans-Oblique.ttf"))
            self.family = "Body"
        else:  # pragma: no cover - fonts are shipped with the repo
            self.family = "Helvetica"
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=16)

    def safe(self, text) -> str:
        text = str(text)
        if self.unicode:
            # Drop characters the bundled font has no glyph for (e.g. Devanagari)
            # instead of raising, and say so.
            font = self.fonts.get("body") or next(iter(self.fonts.values()), None)
            cmap = getattr(font, "cmap", None)
            if cmap:
                cleaned = "".join(ch if (ord(ch) in cmap or ch in "\n\t") else "?" for ch in text)
                return cleaned
            return text
        return text.replace("₹", "Rs.").encode("latin-1", "replace").decode("latin-1")

    def footer(self):
        self.set_y(-12)
        self.set_font(self.family, "", 7)
        self.set_text_color(120, 120, 120)
        self.multi_cell(0, 4, self.safe(f"{self.footer_text}  |  page {self.page_no()}"), align="C")

    def heading(self, title):
        if self.get_y() > self.h - 40:
            self.add_page()
        self.ln(3)
        self.set_fill_color(*NAVY)
        self.set_text_color(255, 255, 255)
        self.set_font(self.family, "B", 10)
        self.cell(0, 7, self.safe(f"  {title}"), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def row(self, label, value, shade=False, color=BLACK):
        w = self.w - self.l_margin - self.r_margin
        y0 = self.get_y()
        self.set_font(self.family, "", 8.5)
        self.set_text_color(*BLACK)
        lines_l = self.multi_cell(w * 0.58, 5, self.safe(label), dry_run=True, output="LINES")
        self.set_font(self.family, "B", 8.5)
        lines_v = self.multi_cell(w * 0.42 - 2, 5, self.safe(value), dry_run=True, output="LINES")
        h = 5 * max(len(lines_l), len(lines_v)) + 1.5
        if y0 + h > self.page_break_trigger:
            self.add_page()
            y0 = self.get_y()
        if shade:
            self.set_fill_color(*GREY)
            self.rect(self.l_margin, y0, w, h, style="F")
        self.set_xy(self.l_margin + 2, y0 + 0.75)
        self.set_font(self.family, "", 8.5)
        self.multi_cell(w * 0.58 - 2, 5, self.safe(label), align="L")
        self.set_xy(self.l_margin + w * 0.58, y0 + 0.75)
        self.set_font(self.family, "B", 8.5)
        self.set_text_color(*color)
        self.multi_cell(w * 0.42 - 2, 5, self.safe(value), align="L")
        self.set_text_color(*BLACK)
        self.set_xy(self.l_margin, y0 + h)

    def rows(self, items):
        for i, item in enumerate(items):
            self.row(*item[:2], shade=(i % 2 == 0), color=item[2] if len(item) > 2 else BLACK)

    def para(self, text, style="", size=8.5):
        self.set_font(self.family, style, size)
        self.set_text_color(*BLACK)
        self.multi_cell(0, 4.6, self.safe(text), align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def generate_pdf_report(analysis, plant_name: str, contact_name: str, contact_email: str,
                        demo: bool = False) -> bytes:
    """Build the PDF from a ``water_intelligence.pipeline.Analysis``."""
    a = analysis
    s = a.settings
    cost = s.water_cost
    period_start, period_end = a.period
    mv = a.mv
    metered = mv.basis.startswith("metered")

    pdf = _Report(footer_text=f"{plant_name}  |  generated {datetime.now():%Y-%m-%d %H:%M}"
                              f"{'  |  DEMO DATA' if demo else ''}")
    pdf.add_page()

    # Header band, height grows with the plant name
    pdf.set_font(pdf.family, "B", 15)
    name_lines = pdf.multi_cell(pdf.w - 30, 7, pdf.safe(plant_name), dry_run=True, output="LINES")[:2]
    band_h = 8 + 7 * len(name_lines) + 12
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, band_h, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 6)
    pdf.multi_cell(pdf.w - 30, 7, pdf.safe("\n".join(name_lines)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(pdf.family, "", 8.5)
    n_days = (period_end - period_start).days + 1
    pdf.multi_cell(pdf.w - 30, 4.5, pdf.safe(
        f"Cooling-tower water report  |  data {period_start} to {period_end} ({n_days} days)\n"
        f"Prepared for {contact_name}  |  {contact_email}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(band_h + 2)

    if demo:
        pdf.set_text_color(*RED)
        pdf.para("DEMO DATA — illustrative only, not a plant result.", style="B", size=9)

    # 1. Data basis
    pdf.heading("Data basis")
    q = a.quality
    pdf.rows([
        ("Rows used / received", f"{q.rows_used:,} / {q.rows_in:,}"),
        ("Hours of data", f"{q.hours_covered:,.0f} h (interval {q.interval_h:g} h)"),
        ("Cycles-of-concentration tracer", q.tracer),
        ("Savings basis", mv.basis),
        ("Heat load basis", f"ΔT {a.hourly['delta_t_source'].iat[0]}"),
        ("Blowdown mode", a.daily.attrs.get("blowdown_mode", "")),
    ])
    for line in q.summary_lines()[2:6] + a.warnings:
        pdf.para(f"• {line}", size=8)

    # 2. Envelope and potential
    env = a.env
    status_col = RED if a.coc_status != "below" else GREEN
    pot = a.potential
    pdf.heading("Operating envelope and potential (model)")
    pdf.rows([
        ("Current cycles of concentration (time-weighted)", f"{a.current_coc:.2f}", status_col),
        ("Max safe CoC", f"{env.max_safe_coc:.2f}" if env.max_safe_coc > 1 else "none feasible"),
        ("Binding constraint", env.binding),
        (f"Makeup at current CoC (per 30 days, from {pot.hours_basis:,.0f} h)",
         f"{_fmt(pot.current_m3_per_30d)} m³"),
        ("Makeup at max safe CoC (per 30 days)", f"{_fmt(pot.optimised_m3_per_30d)} m³"),
        ("Potential change (per 30 days)",
         f"{_fmt(pot.savings_m3_per_30d)} m³  /  ₹{_fmt(pot.savings_m3_per_30d * cost)}"),
    ])
    if env.not_checked:
        pdf.para("Constraints NOT checked (data missing): " + "; ".join(env.not_checked), size=8)
    for n in env.notes:
        pdf.para(f"Note: {n}", size=8)

    # 3. M&V
    pdf.heading("Savings vs baseline (heat-load normalised)")
    bs, be = a.baseline
    items = [
        ("Baseline period (inclusive)", f"{bs} to {be}  ({mv.baseline_days} days)"),
        ("Reporting period", f"{mv.reporting_start} to {mv.reporting_end}  ({mv.reporting_days} days)"
         if mv.reporting_start else "no data yet"),
        ("Water intensity, baseline → reporting (m³ makeup per m³ evaporated)",
         f"{_fmt(mv.baseline_intensity, '.3f')} → {_fmt(mv.reporting_intensity, '.3f')}"),
        ("Savings over reporting period", f"{_fmt(mv.total_savings_m3)} m³  /  ₹{_fmt(mv.total_savings_m3 * cost)}"),
        (f"Service fee ({s.fee_share:.0%} of savings, not billed if negative)",
         f"₹{_fmt(max(mv.total_savings_m3, 0) * cost * s.fee_share) if np.isfinite(mv.total_savings_m3) else 'n/a'}"),
    ]
    if np.isfinite(mv.annual_m3):
        lo, hi = mv.annual_ci_m3
        items.append(("Annualised (mean daily × 365), 95% interval",
                      f"{_fmt(mv.annual_m3)} m³  ({_fmt(lo)} to {_fmt(hi)})"))
    pdf.rows(items)
    for m in mv.messages:
        pdf.para(f"• {m}", size=8)

    # 4. Scaling risk
    sk = a.lsi["lsi_skin"]
    lim = s.envelope.lsi_limit
    pdf.heading("Scaling and corrosion (Langelier index, hourly)")
    pdf.rows([
        (f"LSI limit (at {s.envelope.skin_temp_c:g} °C heat-exchanger skin)", f"{lim:+.2f}"),
        ("Skin LSI: mean / 95th percentile / max", f"{sk.mean():+.2f} / {sk.quantile(0.95):+.2f} / {sk.max():+.2f}"),
        ("Basin LSI: mean", f"{a.lsi['lsi_basin'].mean():+.2f}"),
        ("Hours with skin LSI above limit", f"{a.lsi_skin_hours_above:,.0f} h",
         RED if a.lsi_skin_hours_above else GREEN),
    ])

    # 5. Anomalies
    an = a.anomalies
    pdf.heading("Anomaly events (sustained deviations only)")
    pdf.rows([
        ("Events / hours in events", f"{len(an.events)} / {an.event_hours:,.0f} h"),
        ("Learned model trained on", an.trained_on),
    ])
    for _, e in an.events.head(8).iterrows():
        pdf.para(f"• {e['start']:%Y-%m-%d %H:%M} ({e['hours']:.0f} h) — {e['kind']}: {e['detail']}", size=8)
    if not len(an.events):
        pdf.para("No sustained anomalies detected.", size=8)

    # 6. Verification statement
    pdf.heading("Basis of figures")
    if metered:
        stmt = (f"Savings use metered makeup from the plant's makeup meter, normalised to heat load "
                f"via the baseline water intensity for {bs} to {be}.")
    else:
        stmt = ("No makeup meter was provided. Makeup is MODELLED from cycles of concentration and a heat "
                "balance, so savings figures are engineering estimates, not verified measurements. "
                "Install a makeup flow meter for billable verification.")
    if q.rows_dropped or q.coc_outliers or q.gaps:
        stmt += " Some readings were excluded by data-quality checks (see Data basis)."
    pdf.para(stmt, style="I")

    return bytes(pdf.output())
