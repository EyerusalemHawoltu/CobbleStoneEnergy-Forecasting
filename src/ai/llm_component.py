"""
AI / LLM Component — Groq (free tier) integration.

Uses Groq's free API with Llama 3 models.
Register at console.groq.com for a free API key.

Two programmatic AI features:
  1. QA Rule Proposer   — LLM proposes data validation rules from schema + sample
  2. Daily Commentary   — LLM narrates computed metrics as a market morning note

All LLM calls are:
  • Made from code (never manual)
  • Fully logged (prompt, response, latency, token usage) to outputs/logs/
  • Parameterised via GROQ_API_KEY environment variable (no secrets in code)
  • Gracefully degraded: failures return a sentinel string and log the error
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from groq import Groq
from loguru import logger

LOG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Free Groq models — no billing required
FAST_MODEL = "llama-3.1-8b-instant"        # ultra-low latency, great for QA rules
CAPABLE_MODEL = "llama-3.3-70b-versatile"  # higher quality, still free


def _get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not set. "
            "Register free at console.groq.com and set the env variable."
        )
    return Groq(api_key=key)


def _log_call(tag: str, prompt: str, response: str, meta: dict) -> Path:
    log_path = LOG_DIR / f"llm_{tag}_{int(time.time())}.json"
    record = {
        "tag": tag,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": meta.get("model", FAST_MODEL),
        "prompt_preview": prompt[:500],
        "response": response,
        "metadata": meta,
    }
    with open(log_path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    logger.debug(f"LLM call logged → {log_path}")
    return log_path


# ── Feature 1: QA Rule Proposer ───────────────────────────────────────────────

_QA_RULES_PROMPT = """\
You are a data quality engineer for a European power market pipeline that fetches \
hourly data from ENTSO-E Transparency for the German (DE) bidding zone.

DOMAIN KNOWLEDGE (critical — rules must respect these facts):
- da_price (Day-Ahead price EUR/MWh): CAN be negative. EPEX SPOT allows prices from \
-500 to +4000 EUR/MWh. Negative prices are normal during high-wind/low-demand hours. \
NEVER propose a non-negative rule for da_price.
- load_mw and load_forecast_mw: system load in MW — must be positive (15,000–90,000 MW for DE).
- wind_solar_da_mw, wind_solar_actual_mw: combined wind+solar generation in MW — must be \
non-negative. There is NO fixed relationship between load and wind_solar generation; \
do NOT propose rules comparing load to wind_solar.
- All columns are independently measured series. Cross-column arithmetic identities \
(e.g. load = forecast + generation) do not apply.

SCHEMA (column → dtype):
{schema}

DESCRIPTIVE STATISTICS:
{stats}

SAMPLE ROWS (first 5):
{sample}

Propose data validation rules for this dataset.
Return ONLY a valid JSON array with NO markdown fences, no explanation.
Each element must be an object with exactly these keys:
  "field"     : column name the rule applies to (string)
  "rule"      : short human-readable description (string)
  "condition" : a Python expression that evaluates to a boolean Series using \
'df' as the DataFrame variable. Use pandas methods only.
  "severity"  : "error" or "warning"

Focus on physical plausibility and range checks only. Do NOT invent cross-column \
arithmetic identities.
"""


def propose_qa_rules(df: pd.DataFrame, n_sample: int = 5) -> list[dict]:
    """Ask the LLM to propose validation rules. Returns [] on any failure."""
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    stats = df.describe().round(2).to_string()
    sample = df.head(n_sample).to_string()

    prompt = _QA_RULES_PROMPT.format(
        schema=json.dumps(schema, indent=2),
        stats=stats,
        sample=sample,
    )
    t0 = time.time()
    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        raw = completion.choices[0].message.content.strip()
        # Strip markdown code fences that some models add despite instructions
        clean = raw
        if "```" in clean:
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", clean)
            if match:
                clean = match.group(1).strip()
        latency = round(time.time() - t0, 2)
        meta = {
            "model": FAST_MODEL,
            "latency_s": latency,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }
        _log_call("qa_rules", prompt, raw, meta)
        rules = json.loads(clean)
        logger.success(f"LLM proposed {len(rules)} QA rules in {latency}s")
        return rules
    except json.JSONDecodeError as exc:
        logger.warning(f"QA rules: JSON parse failed — {exc}")
        _log_call("qa_rules_parse_error", prompt, str(exc), {"error": traceback.format_exc()})
        return []
    except Exception as exc:
        logger.warning(f"QA rules call failed: {exc}")
        _log_call("qa_rules_call_error", prompt, "", {"error": traceback.format_exc()})
        return []


# ── Feature 2: Daily Market Commentary ────────────────────────────────────────

_COMMENTARY_PROMPT = """\
You are a European power market analyst writing the daily morning note for the \
German day-ahead desk. Write a concise 3–4 sentence commentary based ONLY on \
the metrics below. Do NOT invent any numbers; use only the figures provided.

DATE: {date}

FORECAST DA PRICES (EUR/MWh):
  Base average:      {base_avg:.2f}
  Peak average:      {peak_avg:.2f}
  Off-peak average:  {offpeak_avg:.2f}
  P10:               {p10:.2f}
  P90:               {p90:.2f}

WEEK-OVER-WEEK CHANGE (vs same weekday -7d):
  Base WoW:          {wow_change:+.2f} EUR/MWh  ({wow_pct:+.1f}%)

FUNDAMENTALS:
  Load:              {load_mw:.0f} MW
  Wind total:        {wind_mw:.0f} MW  (penetration: {wind_pen:.1%})
  Solar:             {solar_mw:.0f} MW (penetration: {solar_pen:.1%})
  Residual load:     {residual_mw:.0f} MW

RECENT PRICE HISTORY:
  Yesterday avg:     {yesterday_avg:.2f} EUR/MWh
  7-day rolling avg: {roll7d:.2f} EUR/MWh
  30-day rolling avg:{roll30d:.2f} EUR/MWh

Close with one sentence on the implied direction for the prompt-month contract.
"""


def _fmt(metrics: dict[str, Any]) -> dict[str, Any]:
    """Replace NaN floats with readable 'N/A' strings for prompt formatting."""
    import math
    out = {}
    for k, v in metrics.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = "N/A"
        else:
            out[k] = v
    return out


def generate_daily_commentary(metrics: dict[str, Any]) -> str:
    """Generate market commentary from computed metrics. Returns fallback on failure."""
    safe = _fmt(metrics)
    # Reformat numeric fields that have format specs in the template
    def _s(v, spec):
        if v == "N/A":
            return "N/A"
        return format(v, spec)

    prompt = (
        _COMMENTARY_PROMPT
        .replace("{base_avg:.2f}", _s(safe["base_avg"], ".2f"))
        .replace("{peak_avg:.2f}", _s(safe["peak_avg"], ".2f"))
        .replace("{offpeak_avg:.2f}", _s(safe["offpeak_avg"], ".2f"))
        .replace("{p10:.2f}", _s(safe["p10"], ".2f"))
        .replace("{p90:.2f}", _s(safe["p90"], ".2f"))
        .replace("{wow_change:+.2f}", _s(safe["wow_change"], "+.2f") if safe["wow_change"] != "N/A" else "N/A")
        .replace("{wow_pct:+.1f}", _s(safe["wow_pct"], "+.1f") if safe["wow_pct"] != "N/A" else "N/A")
        .replace("{load_mw:.0f}", _s(safe["load_mw"], ".0f"))
        .replace("{wind_mw:.0f}", _s(safe["wind_mw"], ".0f"))
        .replace("{wind_pen:.1%}", _s(safe["wind_pen"], ".1%") if safe["wind_pen"] != "N/A" else "N/A")
        .replace("{solar_mw:.0f}", _s(safe["solar_mw"], ".0f"))
        .replace("{solar_pen:.1%}", _s(safe["solar_pen"], ".1%") if safe["solar_pen"] != "N/A" else "N/A")
        .replace("{residual_mw:.0f}", _s(safe["residual_mw"], ".0f"))
        .replace("{yesterday_avg:.2f}", _s(safe["yesterday_avg"], ".2f"))
        .replace("{roll7d:.2f}", _s(safe["roll7d"], ".2f"))
        .replace("{roll30d:.2f}", _s(safe["roll30d"], ".2f"))
        .replace("{date}", str(safe["date"]))
    )
    t0 = time.time()
    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=CAPABLE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        commentary = completion.choices[0].message.content.strip()
        latency = round(time.time() - t0, 2)
        meta = {
            "model": CAPABLE_MODEL,
            "latency_s": latency,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            "date": str(metrics.get("date")),
        }
        _log_call("daily_commentary", prompt, commentary, meta)
        logger.success(f"Commentary generated in {latency}s for {metrics.get('date')}")
        return commentary
    except Exception as exc:
        logger.warning(f"Commentary generation failed: {exc}")
        _log_call("commentary_error", prompt, "", {"error": traceback.format_exc()})
        return f"[LLM commentary unavailable — {exc}. Check GROQ_API_KEY.]"


def build_commentary_metrics(
    forecast: pd.Series,
    actuals: pd.Series,
    df_features: pd.DataFrame,
    target_date: pd.Timestamp,
) -> dict[str, Any]:
    """Assemble metric dict from pipeline outputs — no invented numbers."""
    day_mask = forecast.index.normalize() == target_date.normalize()
    day_fc = forecast.loc[day_mask]

    local_hour = day_fc.index.tz_convert("Europe/Berlin").hour if day_fc.index.tz else day_fc.index.hour
    peak_mask = pd.Series(np.isin(np.array(local_hour), list(range(8, 20))), index=day_fc.index)

    base_avg = float(day_fc.mean()) if len(day_fc) > 0 else float("nan")
    peak_avg = float(day_fc.loc[peak_mask].mean()) if peak_mask.any() else float("nan")
    offpeak_avg = float(day_fc.loc[~peak_mask].mean()) if (~peak_mask).any() else float("nan")
    p10 = float(day_fc.quantile(0.10)) if len(day_fc) > 0 else float("nan")
    p90 = float(day_fc.quantile(0.90)) if len(day_fc) > 0 else float("nan")

    wow_date = target_date - pd.Timedelta(days=7)
    wow_mask = actuals.index.normalize() == wow_date.normalize()
    wow_base = float(actuals.loc[wow_mask].mean()) if wow_mask.any() else float("nan")
    wow_change = base_avg - wow_base
    wow_pct = 100 * wow_change / wow_base if wow_base and wow_base != 0 else float("nan")

    past = actuals.loc[actuals.index < target_date]
    prev_day_mask = actuals.index.normalize() == (target_date - pd.Timedelta(days=1)).normalize()
    yesterday_avg = float(actuals.loc[prev_day_mask].mean()) if prev_day_mask.any() else float("nan")
    roll7d = float(past.tail(168).mean()) if len(past) >= 24 else float("nan")
    roll30d = float(past.tail(720).mean()) if len(past) >= 24 else float("nan")

    feat_day_mask = df_features.index.normalize() == target_date.normalize()
    feat_day = df_features.loc[feat_day_mask] if feat_day_mask.any() else pd.DataFrame()
    load_mw = float(feat_day["load_mw"].mean()) if "load_mw" in feat_day.columns and len(feat_day) > 0 else float("nan")
    wind_mw = float(feat_day["wind_total_mw"].mean()) if "wind_total_mw" in feat_day.columns and len(feat_day) > 0 else float("nan")
    solar_mw = float(feat_day["solar_mw"].mean()) if "solar_mw" in feat_day.columns and len(feat_day) > 0 else float("nan")
    residual_mw = float(feat_day["residual_load_mw"].mean()) if "residual_load_mw" in feat_day.columns and len(feat_day) > 0 else float("nan")
    wind_pen = wind_mw / load_mw if load_mw and load_mw > 0 else float("nan")
    solar_pen = solar_mw / load_mw if load_mw and load_mw > 0 else float("nan")

    return {
        "date": str(target_date.date()),
        "base_avg": base_avg, "peak_avg": peak_avg, "offpeak_avg": offpeak_avg,
        "p10": p10, "p90": p90, "wow_change": wow_change, "wow_pct": wow_pct,
        "load_mw": load_mw, "wind_mw": wind_mw, "solar_mw": solar_mw,
        "residual_mw": residual_mw, "wind_pen": wind_pen, "solar_pen": solar_pen,
        "yesterday_avg": yesterday_avg, "roll7d": roll7d, "roll30d": roll30d,
    }
