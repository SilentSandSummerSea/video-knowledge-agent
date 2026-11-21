"""Token accounting and transparent DeepSeek Flash cost estimates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PRICING_VERSION = "2026-09-10"
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
)

# CNY per one million tokens. Output tokens already include reasoning tokens.
DEEPSEEK_FLASH_PRICES = {
    "off_peak": {"cached_input": 0.02, "uncached_input": 1.0, "output": 4.0},
    "peak": {"cached_input": 0.04, "uncached_input": 2.0, "output": 8.0},
}

try:
    BEIJING_TIME = ZoneInfo("Asia/Shanghai")
except ZoneInfoNotFoundError:
    # Windows Python installations may not include the IANA timezone database.
    BEIJING_TIME = timezone(timedelta(hours=8))


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def response_usage(response: Any) -> dict[str, int]:
    """Normalize Responses API token usage without double-counting reasoning."""
    usage = _value(response, "usage", {}) or {}
    input_details = _value(usage, "input_tokens_details", {}) or {}
    output_details = _value(usage, "output_tokens_details", {}) or {}
    return {
        "input_tokens": int(_value(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(_value(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(_value(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(_value(output_details, "reasoning_tokens", 0) or 0),
    }


def empty_usage() -> dict[str, int]:
    return {key: 0 for key in TOKEN_FIELDS}


def add_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    for key in TOKEN_FIELDS:
        total[key] = int(total.get(key, 0)) + int(usage.get(key, 0) or 0)


def pricing_period(at: datetime | None = None) -> str:
    """Return the official DeepSeek billing period using Beijing time."""
    moment = at or datetime.now(BEIJING_TIME)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=BEIJING_TIME)
    else:
        moment = moment.astimezone(BEIJING_TIME)
    hour = moment.hour
    is_weekday = moment.weekday() < 5
    return "peak" if is_weekday and (9 <= hour < 12 or 14 <= hour < 18) else "off_peak"


def estimate_deepseek_flash_cost(
    usage: dict[str, Any], *, period: str | None = None
) -> dict[str, object]:
    selected = period or pricing_period()
    if selected not in DEEPSEEK_FLASH_PRICES:
        raise ValueError("period must be 'peak' or 'off_peak'")
    prices = DEEPSEEK_FLASH_PRICES[selected]
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached_tokens = min(input_tokens, int(usage.get("cached_input_tokens", 0) or 0))
    uncached_tokens = input_tokens - cached_tokens
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cost = (
        cached_tokens * prices["cached_input"]
        + uncached_tokens * prices["uncached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000
    return {
        "currency": "CNY",
        "model": "deepseek-flash",
        "pricing_version": PRICING_VERSION,
        "pricing_period": selected,
        "cached_input_cny_per_million": prices["cached_input"],
        "uncached_input_cny_per_million": prices["uncached_input"],
        "output_cny_per_million": prices["output"],
        "estimated_cost_cny": round(cost, 6),
    }


def local_text_cost(model: str) -> dict[str, object]:
    """Local inference has no external per-token API charge."""
    return {
        "currency": "CNY",
        "provider": "local",
        "model": model,
        "pricing_period": "local",
        "estimated_cost_cny": 0.0,
    }
