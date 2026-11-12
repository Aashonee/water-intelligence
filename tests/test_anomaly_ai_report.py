"""Anomaly detection, AI guard rails, PDF report and an app smoke test."""

import shutil
import subprocess
from datetime import date
from types import SimpleNamespace

import pytest
from conftest import ROOT, make_plant

from report_generator import generate_pdf_report
from water_intelligence import ai, anomaly, pipeline
from water_intelligence import chemistry as ch


def _run(raw):
    return pipeline.run(raw, pipeline.Settings(envelope=ch.EnvelopeSettings(lsi_limit=2.5),
                                               baseline_start=date(2024, 1, 1),
                                               baseline_end=date(2024, 1, 15)))


# --- anomalies ----------------------------------------------------------------
def test_healthy_plant_can_report_zero_anomalies():
    a = _run(make_plant(days=30, coc=2.2, noise=0.01, seed=3))
    assert len(a.anomalies.events) == 0  # v0.1 always flagged 5 % (36 h)


def test_flatlined_probe_is_detected():
    raw = make_plant(days=30, coc=2.2, noise=0.01, seed=4)
    raw.loc[500:512, "pH"] = 7.61
    a = _run(raw)
    kinds = set(a.anomalies.events["kind"])
    assert "Sensor flatline — pH probe" in kinds


def test_sustained_shift_is_learned_and_explained():
    raw = make_plant(days=30, coc=2.2, noise=0.01, seed=5)
    raw.loc[600:612, "pH"] += 0.9
    a = _run(raw)
    learned = a.anomalies.events[a.anomalies.events["kind"] == "Unusual pattern (learned)"]
    assert len(learned) >= 1
    assert "pH" in learned["detail"].iat[0] or "skin LSI" in learned["detail"].iat[0]


def test_coc_above_envelope_rule():
    raw = make_plant(days=30, coc=2.2, noise=0.005, seed=6)
    raw.loc[620:640, "TDS_circ"] *= 5   # blowdown valve stuck closed
    raw.loc[620:640, "calcium_hardness"] *= 5
    raw.loc[620:640, "alkalinity"] *= 5
    a = _run(raw)
    assert any(k.startswith("CoC above safe envelope") for k in a.anomalies.events["kind"])


def test_runs_helper_respects_duration():
    import pandas as pd
    m = pd.Series([0, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    dt = pd.Series([1.0] * 8)
    assert anomaly.runs(m, dt, 3) == [(4, 7, 4.0)]


# --- AI -----------------------------------------------------------------------
FACTS = {"Current CoC": "1.75", "Max safe CoC": "2.44", "Potential change (m3 per 30 days)": "13,485"}


class FakeClient:
    def __init__(self, text=None, exc=None):
        self.messages = SimpleNamespace(create=self._create)
        self.text, self.exc = text, exc

    def _create(self, **kw):
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def test_faithful_reply_passes():
    reply = "CoC is 1.75 and could rise to 2.4, saving about 13,500 m3 in 30 days."
    n = ai.narrate(FACTS, None, client=FakeClient(reply))
    assert n.source == "claude" and n.error is None


def test_invented_number_is_rejected():
    n = ai.narrate(FACTS, None, client=FakeClient("Raise CoC to 4.0 to save 20,000 m3."))
    assert n.source == "template" and 4.0 in n.unsupported


def test_api_failure_does_not_raise():
    n = ai.narrate(FACTS, None, client=FakeClient(exc=RuntimeError("rate limited")))
    assert n.source == "template" and "rate limited" in n.error


def test_missing_key_falls_back():
    n = ai.narrate(FACTS, None)
    assert n.source == "template" and "API_KEY" in n.error


# --- PDF ----------------------------------------------------------------------
def test_pdf_handles_unicode_and_long_names(sample_csv, tmp_path):
    a = pipeline.run(sample_csv, pipeline.Settings(envelope=ch.EnvelopeSettings(lsi_limit=1.8)))
    pdf = generate_pdf_report(a, "श्री Chemicals — " + "Very Long Plant Name " * 8, "Anaïs ₹", "a@b.c", demo=True)
    assert pdf[:4] == b"%PDF"
    if shutil.which("pdftotext"):
        p = tmp_path / "r.pdf"
        p.write_bytes(pdf)
        text = subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True).stdout
        assert "Hours with skin LSI above limit" in text
        assert "Anomaly events" in text
        assert "MODELLED" in text            # no "verified sensor data" claim without a meter
        assert "2024-01-16 to 2024-01-30" in text  # reporting period, not the baseline


# --- app smoke test -----------------------------------------------------------
@pytest.mark.parametrize("mode", ["Demo dataset", "Quick calculator (one reading)", "Upload CSV"])
def test_app_runs_without_exceptions(mode):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "cooling_tower_dashboard.py"), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(mode).run()
    assert not at.exception


def test_app_infeasible_programme_shows_error_not_crash():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "cooling_tower_dashboard.py"), default_timeout=120)
    at.run()
    at.sidebar.selectbox[1].set_value("No scale inhibitor").run()
    assert not at.exception
    assert any("No CoC above 1" in e.value for e in at.error)
