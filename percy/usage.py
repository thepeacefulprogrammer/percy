from decimal import Decimal

from agent_framework import AgentSession
from rich.table import Table
from rich.text import Text

from percy.config import load_local_config
from percy.theme import CATPPUCCIN_MOCHA


USAGE_TALLY_KEYS = (
    "input_token_count",
    "output_token_count",
    "total_token_count",
    "openai.cached_input_tokens",
    "openai.reasoning_tokens",
)


def usage_int(usage: dict | None, key: str) -> int:
    if not usage:
        return 0
    value = usage.get(key)
    return int(value or 0)


def format_cost(value: float | Decimal | None) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def estimate_cost(usage: dict | None) -> float | None:
    if not usage:
        return None

    pricing = load_local_config().get("pricing", {})
    if not pricing:
        return None

    input_rate = pricing.get("input_per_million")
    output_rate = pricing.get("output_per_million")
    cached_input_rate = pricing.get("cached_input_per_million", input_rate)
    reasoning_output_rate = pricing.get("reasoning_output_per_million", output_rate)

    if input_rate is None or output_rate is None:
        return None

    input_tokens = usage_int(usage, "input_token_count")
    output_tokens = usage_int(usage, "output_token_count")
    cached_input_tokens = usage_int(usage, "openai.cached_input_tokens")
    reasoning_tokens = usage_int(usage, "openai.reasoning_tokens")

    non_cached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    non_reasoning_output_tokens = max(output_tokens - reasoning_tokens, 0)

    cost = 0.0
    cost += (non_cached_input_tokens / 1_000_000) * float(input_rate)
    cost += (cached_input_tokens / 1_000_000) * float(cached_input_rate)
    cost += (non_reasoning_output_tokens / 1_000_000) * float(output_rate)
    cost += (reasoning_tokens / 1_000_000) * float(reasoning_output_rate)
    return cost


def update_session_usage_tally(
    session: AgentSession, usage: dict | None, turn_cost: float | None
) -> dict:
    tally = dict(session.state.get("_usage_tally", {}))

    for key in USAGE_TALLY_KEYS:
        tally[key] = int(tally.get(key, 0)) + usage_int(usage, key)

    tally["turn_count"] = int(tally.get("turn_count", 0)) + 1
    tally["cost_usd"] = float(tally.get("cost_usd", 0.0)) + float(turn_cost or 0.0)
    session.state["_usage_tally"] = tally
    return tally


def build_usage_renderable(usage: dict | None, session_tally: dict | None = None):
    usage_row = Table.grid(expand=True)
    usage_row.add_column(ratio=1, justify="left")
    usage_row.add_column(ratio=1, justify="right")

    if not usage:
        usage_row.add_row(
            Text("context remaining —", style=CATPPUCCIN_MOCHA["overlay1"]),
            Text("", style=CATPPUCCIN_MOCHA["overlay1"]),
        )
        return usage_row

    token_limits = load_local_config().get("token_limits", {})
    context_window = int(token_limits.get("context_window", 0) or 0)
    total_tokens = usage_int(usage, "total_token_count")
    turn_cost = estimate_cost(usage)
    used_pct = ((total_tokens / context_window) * 100) if context_window and total_tokens else 0
    remaining_pct = max(0.0, 100.0 - used_pct) if context_window else None
    session_cost = float(session_tally.get("cost_usd", 0.0)) if session_tally is not None else None

    left_text = Text(style=CATPPUCCIN_MOCHA["overlay1"])
    left_text.append(
        f"context remaining {remaining_pct:.1f}%" if remaining_pct is not None else "context remaining —"
    )

    right_text = Text(style=CATPPUCCIN_MOCHA["overlay1"])
    right_text.append(f"turn {format_cost(turn_cost)}")
    if session_cost is not None:
        right_text.append(" · ")
        right_text.append(f"session {format_cost(session_cost)}")

    usage_row.add_row(left_text, right_text)
    return usage_row
