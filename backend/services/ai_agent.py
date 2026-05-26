"""
Groq AI Agent — interprets free-text questions and calls pipeline tools.

Uses Groq's free Llama 3.3 70B model with function calling (tool use).
The agent loop:
  1. Send user message + available tools to Groq
  2. If Groq requests a tool call, execute it against the pipeline
  3. Feed result back to Groq
  4. Return final natural-language response + structured data

All calls are logged to outputs/logs/.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

from groq import Groq
from groq import BadRequestError as GroqBadRequestError
from loguru import logger

AGENT_MODEL = "llama-3.3-70b-versatile"  # free tier, best reasoning

SYSTEM_PROMPT = """\
You are an expert European power market analyst assistant for the Cobblestone Energy \
German (DE) day-ahead forecasting pipeline.

You have access to tools that query the live forecasting pipeline. When a user asks \
about prices, forecasts, model performance, data quality, or market commentary, \
always call the appropriate tool first — never invent numbers.

After calling a tool, interpret the results clearly for a power trading desk. \
Be concise and trading-relevant. When you have forecast data, mention base, peak, \
and off-peak averages and their trading implications.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_daily_forecast",
            "description": (
                "Get the hourly day-ahead price forecast for a specific date. "
                "Returns hourly prices, base/peak/offpeak averages, and P10/P90 bands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (e.g. '2024-09-15')",
                    }
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_delivery_summary",
            "description": (
                "Get aggregated delivery-period forecasts (daily, weekly, or monthly averages). "
                "Shows base, peak, off-peak averages and P10/P90 bands for each period."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["daily", "weekly", "monthly"],
                        "description": "Aggregation period",
                    }
                },
                "required": ["period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": (
                "Get forecasting model performance metrics: MAE, RMSE, Tail-MAE (P90) "
                "for LightGBM, Seasonal Naive, and Ridge Linear models."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_qa_summary",
            "description": (
                "Get a data quality report: missingness, duplicates, temporal gaps, "
                "hard-limit violations, and LLM-proposed rule results."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trading_signal",
            "description": (
                "Get the current trading signal (LONG/SHORT/FLAT) based on the "
                "Z-score of forecast vs 30-day rolling benchmark."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Optional date YYYY-MM-DD; defaults to latest available",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_commentary",
            "description": (
                "Generate an AI market commentary for a date, using only computed pipeline "
                "metrics (no invented numbers). Suitable for a daily morning note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format; defaults to latest",
                    }
                },
                "required": [],
            },
        },
    },
]


def _execute_tool(name: str, args: dict, pipeline) -> Any:
    """Dispatch a tool call to the pipeline service."""
    if name == "get_daily_forecast":
        return pipeline.get_daily_forecast(args.get("date", ""))
    if name == "get_delivery_summary":
        return pipeline.get_delivery_summary(args.get("period", "monthly"))
    if name == "get_model_metrics":
        return pipeline.get_model_metrics()
    if name == "get_qa_summary":
        return pipeline.get_qa_summary()
    if name == "get_trading_signal":
        return pipeline.get_trading_signal(args.get("date"))
    if name == "generate_commentary":
        return pipeline.generate_commentary(args.get("date"))
    return {"error": f"Unknown tool: {name}"}


import re as _re

def _strip_function_syntax(text: str) -> str:
    """
    Remove raw <function=name>{...}</function> blocks that Llama on Groq
    occasionally leaks into message content alongside proper tool_calls.
    """
    # Remove <function=name>...</function> blocks
    text = _re.sub(r"<function=[^>]+>.*?</function>", "", text, flags=_re.DOTALL)
    # Remove orphan closing tags
    text = _re.sub(r"</function>", "", text)
    # Collapse extra blank lines left behind
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chat(
    message: str,
    pipeline,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run the agent loop for a user message.

    Returns:
      {
        "response": str,           # natural-language answer
        "tool_calls": list[dict],  # which tools were called and their results
        "data": dict | None,       # structured data for frontend charts
      }
    """
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return {
            "response": (
                "AI agent unavailable: GROQ_API_KEY not set. "
                "Register free at console.groq.com and add it to your .env file."
            ),
            "tool_calls": [],
            "data": None,
        }

    from datetime import date, timedelta
    today = date.today()
    tomorrow = today + timedelta(days=1)

    forecast_range = ""
    if pipeline.ready and pipeline.predictions is not None:
        fc_start = pipeline.predictions.index.min().date().isoformat()
        fc_end = pipeline.predictions.index.max().date().isoformat()
        forecast_range = (
            f" Forecast data covers {fc_start} to {fc_end}."
            f" If the user asks for a date outside this range, use {fc_end} as the latest available date."
        )

    system_with_date = (
        SYSTEM_PROMPT
        + f"\n\nToday's date is {today.isoformat()}. Tomorrow is {tomorrow.isoformat()}."
        + " Always use YYYY-MM-DD format when calling date-based tools."
        + forecast_range
    )

    client = Groq(api_key=key)
    messages = [{"role": "system", "content": system_with_date}]
    if history:
        messages.extend(history[-6:])  # keep last 3 turns for context
    messages.append({"role": "user", "content": message})

    tool_results: list[dict] = []
    structured_data: dict | None = None
    t0 = time.time()

    try:
        # ── Round 1: let Groq decide if it needs a tool ────────────────────────
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.1,
        )
        msg = response.choices[0].message

        # ── Execute tool calls if any ──────────────────────────────────────────
        if msg.tool_calls:
            # Llama on Groq sometimes emits raw <function=...> syntax in msg.content
            # alongside the proper tool_calls. Strip it so it doesn't pollute round 2.
            clean_content = _strip_function_syntax(msg.content or "")
            messages.append({"role": "assistant", "content": clean_content, "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    # json.loads("null") → None; guard with `or {}`
                    fn_args = json.loads(tc.function.arguments or "{}") or {}
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                result = _execute_tool(fn_name, fn_args, pipeline)
                tool_results.append({"tool": fn_name, "args": fn_args, "result": result})

                # Surface first meaningful result as structured data for charts
                if structured_data is None and isinstance(result, dict) and "error" not in result:
                    structured_data = {"type": fn_name, **result}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

            # ── Round 2: Groq synthesises answer from tool results ─────────────
            response2 = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )
            final_text = _strip_function_syntax(response2.choices[0].message.content or "")
        else:
            final_text = msg.content or ""

        latency = round(time.time() - t0, 2)
        logger.info(f"Agent response in {latency}s, tools called: {[t['tool'] for t in tool_results]}")

        return {
            "response": final_text,
            "tool_calls": tool_results,
            "data": structured_data,
            "latency_s": latency,
        }

    except Exception as exc:
        logger.error(f"Agent error: {exc}\n{traceback.format_exc()}")
        return {
            "response": f"Agent error: {exc}. Check logs for details.",
            "tool_calls": tool_results,
            "data": None,
        }
