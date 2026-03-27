from fpdf import FPDF
from io import BytesIO


DARK_NAVY = (10, 36, 99)
BLACK = (0, 0, 0)
LIGHT_GREY = (245, 245, 245)
MID_GREY = (200, 200, 200)


def generate_pdf_report(
    plant_name,
    report_start,
    report_end,
    current_CoC,
    optimised_CoC,
    current_makeup,
    optimised_makeup,
    monthly_savings_m3,
    monthly_savings_Rs,
    my_fees,
    avg_LSI,
    flagged_hours,
    baseline_start,
    baseline_end,
    contact_name,
    contact_email,
    dominant_cause
) -> bytes:
    """
    Generate a one-page PDF monthly report for a cooling tower water
    optimisation system. Returns the PDF as bytes.
    """

    # --- LSI status --------------------------------------------------------
    if avg_LSI > 0.5:
        lsi_status = "High Scaling Risk"
        lsi_color = (180, 30, 30)
    elif avg_LSI < -0.5:
        lsi_status = "Corrosive"
        lsi_color = (200, 100, 0)
    else:
        lsi_status = "Balanced"
        lsi_color = (30, 130, 60)

    # --- PDF setup ---------------------------------------------------------
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_margins(left=15, top=15, right=15)

    page_w = pdf.w - 30  # usable width (margins removed)

    # -----------------------------------------------------------------------
    # 1. HEADER BAND
    # -----------------------------------------------------------------------
    pdf.set_fill_color(*DARK_NAVY)
    pdf.rect(x=0, y=0, w=pdf.w, h=28, style="F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(15, 6)
    pdf.cell(w=page_w, h=8, text=plant_name, align="L")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(15, 15)
    pdf.cell(
        w=page_w,
        h=5,
        text=f"Monthly Water Optimisation Report  |  {report_start} - {report_end}",
        align="L",
    )
    pdf.set_xy(15, 21)
    pdf.cell(
        w=page_w,
        h=5,
        text=f"Prepared for: {contact_name}   |   {contact_email}",
        align="L",
    )

    # -----------------------------------------------------------------------
    # helper: section heading
    # -----------------------------------------------------------------------
    def section_heading(title: str, y: float) -> float:
        pdf.set_xy(15, y)
        pdf.set_fill_color(*DARK_NAVY)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(w=page_w, h=7, text=f"  {title}", fill=True, ln=True)
        return pdf.get_y()

    # helper: two-column key-value row with alternating shading
    def kv_row(label: str, value: str, y: float, shade: bool) -> float:
        col1 = page_w * 0.55
        col2 = page_w * 0.45
        if shade:
            pdf.set_fill_color(*LIGHT_GREY)
            pdf.rect(x=15, y=y, w=page_w, h=6.5, style="F")
        pdf.set_text_color(*BLACK)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_xy(17, y + 0.8)
        pdf.cell(w=col1, h=5, text=label)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_xy(15 + col1, y + 0.8)
        pdf.cell(w=col2, h=5, text=value)
        return y + 6.5

    # -----------------------------------------------------------------------
    # 2. WATER BALANCE SUMMARY
    # -----------------------------------------------------------------------
    y = 33
    y = section_heading("Water Balance Summary", y)
    y += 1

    rows = [
        ("Current Cycles of Concentration (CoC)", f"{current_CoC:.2f}"),
        ("Optimised Cycles of Concentration (CoC)", f"{optimised_CoC:.2f}"),
        (f"Current Make-up Water ({report_start} - {report_end})", f"{current_makeup:,.1f} m3"),
        (f"Optimised Make-up Water ({report_start} - {report_end})", f"{optimised_makeup:,.1f} m3"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = kv_row(lbl, val, y, shade=(i % 2 == 0))
    y += 4

    # -----------------------------------------------------------------------
    # 3. MONTHLY SAVINGS
    # -----------------------------------------------------------------------
    y = section_heading("Monthly Savings", y)
    y += 1

    rows = [
        ("Water Saved", f"{monthly_savings_m3:,.1f} m3"),
        ("Money Saved", f"Rs. {monthly_savings_Rs:,.0f}"),
        ("Service Fee", f"Rs. {my_fees:,.0f}"),
    ]
    for i, (lbl, val) in enumerate(rows):
        y = kv_row(lbl, val, y, shade=(i % 2 == 0))
    y += 4

    # -----------------------------------------------------------------------
    # 4. SCALING RISK
    # -----------------------------------------------------------------------
    y = section_heading("Scaling Risk", y)
    y += 1

    y = kv_row("Average Langelier Saturation Index (LSI)", f"{avg_LSI:+.3f}", y, shade=True)

    # status row with coloured text
    pdf.set_fill_color(*LIGHT_GREY)
    pdf.rect(x=15, y=y, w=page_w, h=6.5, style="F")
    pdf.set_text_color(*BLACK)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(17, y + 0.8)
    pdf.cell(w=page_w * 0.55, h=5, text="Status")
    pdf.set_text_color(*lsi_color)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(15 + page_w * 0.55, y + 0.8)
    pdf.cell(w=page_w * 0.45, h=5, text=lsi_status)
    y += 6.5
    y += 4

    # -----------------------------------------------------------------------
    # 5. ANOMALY SUMMARY
    # -----------------------------------------------------------------------
    y = section_heading("Anomaly Summary", y)
    y += 1

    y = kv_row(
    "Hours Above Scaling Threshold (LSI > 0.5)",
    f"{flagged_hours:,} hrs",
    y,
    shade=True,
    )
    y = kv_row(
        "Primary Anomaly Driver",
        dominant_cause,
        y,
        shade=False,
    )
    y += 4

    # -----------------------------------------------------------------------
    # 6. VERIFICATION STATEMENT
    # -----------------------------------------------------------------------
    y = section_heading("Verification", y)
    y += 3

    pdf.set_text_color(*BLACK)
    pdf.set_font("Helvetica", "I", 8.5)
    statement = (
        f"Savings calculated against baseline period {baseline_start} - "
        f"{baseline_end} using verified sensor data."
    )
    pdf.set_xy(15, y)
    pdf.multi_cell(w=page_w, h=5, text=statement, align="L")

    # -----------------------------------------------------------------------
    # FOOTER
    # -----------------------------------------------------------------------
    pdf.set_fill_color(*DARK_NAVY)
    pdf.rect(x=0, y=pdf.h - 10, w=pdf.w, h=10, style="F")
    pdf.set_text_color(180, 180, 180)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_xy(15, pdf.h - 7)
    pdf.cell(
        w=page_w,
        h=4,
        text=f"Confidential  |  {plant_name}  |  Generated for {contact_name}",
        align="C",
    )

    # -----------------------------------------------------------------------
    # Return as bytes
    # -----------------------------------------------------------------------
    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
