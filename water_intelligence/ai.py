"""Claude narration with guard rails.

* Every number the model may use is passed in ``facts``; the reply is checked and any
  number not traceable to a fact (within rounding) causes the reply to be rejected
  in favour of a deterministic template summary.
* API failures (missing key, rate limit, network) never crash the app: callers get
  the template summary plus an error string.
* The app gates calls behind an access code and a per-session call budget so a
  public URL cannot burn the API budget.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_MODEL = os.getenv("WI_CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You explain the output of a cooling-tower water optimisation tool to a plant operator. "
    "Use only the numbers in the FACTS block, copied as given (you may round them). "
    "Do not compute new numbers, do not convert units, and do not introduce any figure that "
    "is not in FACTS. Never suggest operating above the max safe CoC. If the facts say a "
    "constraint was not checked, say so. Write 3-4 plain sentences, no headings."
)

_NUM = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass
class Narration:
    text: str
    source: str             # "claude" or "template"
    error: str | None = None
    unsupported: list | None = None


def facts_block(facts: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in facts.items())


def template_summary(facts: dict) -> str:
    """Deterministic fallback; uses only the facts."""
    parts = [f"{k}: {v}." for k, v in facts.items()]
    return "Summary of computed results — " + " ".join(parts)


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unsupported_numbers(reply: str, facts: dict, rel_tol: float = 0.015) -> list[float]:
    """Numbers in ``reply`` that don't match any number in ``facts`` within rounding."""
    allowed = []
    for v in facts.values():
        allowed.extend(extract_numbers(str(v)))
    bad = []
    for x in extract_numbers(reply):
        if x in (1, 12, 30, 365):   # "one", "12 months", "30 days", "365 days"
            continue
        ok = any(abs(x - a) <= max(rel_tol * abs(a), 0.0051) or
                 # allow rounding to 2 or 1 decimals, or to the nearest 1 / 10 / 100 / 1000
                 any(abs(x - round(a, p)) < 1e-9 for p in range(-3, 3))
                 for a in allowed)
        if not ok:
            bad.append(x)
    return bad


def narrate(facts: dict, api_key: str | None, model: str = DEFAULT_MODEL,
            client=None) -> Narration:
    fallback = template_summary(facts)
    if client is None:
        if not api_key:
            return Narration(fallback, "template", error="No ANTHROPIC_API_KEY configured.")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=30.0, max_retries=2)
        except Exception as e:  # pragma: no cover - import/config failure
            return Narration(fallback, "template", error=f"Could not start Claude client: {e}")
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "FACTS:\n" + facts_block(facts)}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content).strip()
    except Exception as e:
        return Narration(fallback, "template", error=f"Claude request failed ({type(e).__name__}): {e}")
    if not text:
        return Narration(fallback, "template", error="Claude returned an empty reply.")
    bad = unsupported_numbers(text, facts)
    if bad:
        return Narration(fallback, "template",
                         error="Claude's reply contained numbers not in the computed results; "
                               "showing the computed summary instead.",
                         unsupported=bad)
    return Narration(text, "claude")
