import json
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from agent_framework import FunctionInvocationContext, FunctionMiddleware
from rich.console import Console, Group
from rich.json import JSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

theme = Theme(
    {
        "prompt": "bold #89b4fa",
        "user_prompt": "bold #cdd6f4 on #313244",
        "agent": "bold #a6e3a1",
        "system": "#a6adc8",
        "banner": "bold #cba6f7",
        "tool": "bold #89dceb",
        "tool.meta": "#bac2de",
        "tool.success": "bold #a6e3a1",
        "tool.error": "bold #f38ba8",
        "tool.dim": "#6c7086",
    }
)

console = Console(theme=theme)


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_plain_data(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "to_dict"):
        try:
            return _to_plain_data(value.to_dict())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _to_plain_data(vars(value))
        except Exception:
            return str(value)
    return str(value)


def _pretty_json(value: Any) -> str:
    return json.dumps(_to_plain_data(value), indent=2, ensure_ascii=False, default=str)


def _render_payload(value: Any):
    if value is None:
        return Text("(none)", style="tool.dim")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return Text("(empty string)", style="tool.dim")
        if "\n" in value or len(value) > 160:
            return Syntax(value, "markdown", theme="monokai", word_wrap=True)
        return Markdown(value)
    try:
        return JSON.from_data(_to_plain_data(value))
    except Exception:
        return Syntax(_pretty_json(value), "json", theme="monokai", word_wrap=True)


def _normalize_result(value: Any) -> Any:
    if isinstance(value, list):
        normalized = []
        for item in value:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    stripped = text.strip()
                    if stripped:
                        try:
                            normalized.append(json.loads(stripped))
                            continue
                        except Exception:
                            normalized.append(text)
                            continue
            if hasattr(item, "to_dict"):
                try:
                    normalized.append(item.to_dict())
                    continue
                except Exception:
                    pass
            normalized.append(_to_plain_data(item))
        return normalized
    return value


class PercyToolCallMiddleware(FunctionMiddleware):
    """Render Percy-local tool calls and results with Rich panels."""

    def __init__(self, show_user_updates: bool = False):
        self.show_user_updates = show_user_updates

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        function_name = context.function.name
        if function_name == "user_update" and not self.show_user_updates:
            await call_next()
            return

        start = time.perf_counter()
        call_id = context.kwargs.get("tool_call_id") or context.kwargs.get("call_id")
        console.print()

        console.print(
            Panel(
                Group(
                    Text(function_name, style="tool"),
                    Text(
                        f"call_id={call_id}" if call_id else "tool invocation started",
                        style="tool.meta",
                    ),
                    _render_payload(context.arguments),
                ),
                title="[tool]Tool Call[/tool]",
                border_style="#89dceb",
                padding=(0, 1),
            )
        )

        try:
            await call_next()
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            console.print(
                Panel(
                    Group(
                        Text(function_name, style="tool.error"),
                        Text(f"failed in {duration_ms:.1f} ms", style="tool.meta"),
                        _render_payload({"error": str(exc)}),
                    ),
                    title="[tool.error]Tool Error[/tool.error]",
                    border_style="#f38ba8",
                    padding=(0, 1),
                )
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        console.print(
            Panel(
                Group(
                    Text(function_name, style="tool.success"),
                    Text(f"completed in {duration_ms:.1f} ms", style="tool.meta"),
                    _render_payload(_normalize_result(context.result)),
                ),
                title="[tool.success]Tool Result[/tool.success]",
                border_style="#a6e3a1",
                padding=(0, 1),
            )
        )
