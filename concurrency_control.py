import asyncio
import threading
from debug_logger import log_prompt
from ventures_agent_framework import agent, logger
import tiktoken

_enc = tiktoken.encoding_for_model("gpt-5")
MAX_PROMPT_TOKENS = 100_000


def estimate_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


# ──────────────────────────────────────────────
# CONCURRENCY CONTROL
# ──────────────────────────────────────────────

_call_counter = 0
_call_counter_lock = threading.Lock()


def _next_call_id() -> int:
    global _call_counter
    with _call_counter_lock:
        _call_counter += 1
        return _call_counter


MAX_CONCURRENT_API_CALLS = 3
_api_semaphore: asyncio.Semaphore | None = None


def _get_api_semaphore() -> asyncio.Semaphore:
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
    return _api_semaphore


async def throttled_agent(**kwargs):
    """Wraps `agent()` with semaphore-based throttling, retry with backoff, and prompt logging."""
    sem = _get_api_semaphore()
    max_retries = 5
    base_delay = 2.0

    prompt = kwargs.get("prompt", "")
    instructions = kwargs.get("instructions", "")
    total = estimate_tokens(prompt) + estimate_tokens(instructions)

    if total > MAX_PROMPT_TOKENS:
        # Truncate the prompt (not instructions) to fit
        overshoot = total - MAX_PROMPT_TOKENS
        # Rough: remove overshoot * 4 chars from the front of prompt
        chars_to_remove = overshoot * 4
        kwargs["prompt"] = "[...truncated...]\n" + prompt[chars_to_remove:]
        logger.warning(
            f"Call #{_next_call_id()} prompt truncated: "
            f"{total} tokens -> ~{MAX_PROMPT_TOKENS}"
        )

    call_id = _next_call_id()
    agent_name = kwargs.get("name", "unknown")

    # Extract tool names for logging
    agent_tools = kwargs.get("tools", [])
    tool_names = []
    for t in agent_tools:
        if hasattr(t, "name"):
            tool_names.append(t.name)
        elif hasattr(t, "__name__"):
            tool_names.append(t.__name__)
        else:
            tool_names.append(str(t))

    log_prompt(
        call_id,
        "REQUEST",
        agent_name,
        prompt=kwargs.get("prompt", ""),
        instructions=kwargs.get("instructions", ""),
        tool_names=tool_names,
        response_format=str(kwargs.get("response_format", None)),
    )

    start_time = asyncio.get_event_loop().time()

    for attempt in range(max_retries + 1):
        async with sem:
            try:
                result = await agent(**kwargs)

                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                log_prompt(
                    call_id,
                    "RESPONSE",
                    agent_name,
                    response=result,
                    duration_ms=duration_ms,
                )

                return result

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "Too Many Requests" in err_str
                is_context = "context_length_exceeded" in err_str

                if is_context:
                    log_prompt(call_id, "ERROR", agent_name, error=err_str)
                    raise

                if is_rate_limit and attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Rate limited call #{call_id} (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                log_prompt(call_id, "ERROR", agent_name, error=err_str)
                raise
