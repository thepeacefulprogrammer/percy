from ventures_agent_framework import HandOver, get_todo_store

# ──────────────────────────────────────────────
# CONTEXT SIZE MANAGEMENT
# ──────────────────────────────────────────────
MAX_CONTEXT_CHARS = 80_000


def _truncate_section(text: str, max_chars: int, label: str) -> str:
    """Truncate a section from the front, keeping the most recent content."""
    if len(text) <= max_chars:
        return text
    truncated = text[-max_chars:]
    return f"[...{label} truncated, showing last {max_chars} chars...]\n{truncated}"


def build_handover_prompt(
    handover: HandOver,
    task_queue_status: str,
    new_input: str,
) -> str:
    """Build context prompt with size management."""
    report_budget = int(MAX_CONTEXT_CHARS * 0.15)
    history_budget = int(MAX_CONTEXT_CHARS * 0.30)
    queue_budget = int(MAX_CONTEXT_CHARS * 0.10)

    report = _truncate_section(handover.report, report_budget, "report")
    history_str = "\n".join(handover.message_history)
    history_str = _truncate_section(history_str, history_budget, "message history")

    queue_str = _truncate_section(task_queue_status, queue_budget, "task queue")

    try:
        todo_store = get_todo_store(handover.output_dir)
        todo_str = todo_store.format_active()
    except Exception:
        todo_str = "(todo store unavailable)"

    prompt = f"""# Handover Report
{report}

# Active TODOs
{todo_str}

# Task Queue Status
{queue_str}

# Message history
{history_str}

# New Input
{new_input}
"""
    return prompt
