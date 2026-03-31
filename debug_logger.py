from datetime import datetime

# ──────────────────────────────────────────────
# DEBUG / PROMPT LOGGING
# ──────────────────────────────────────────────
DEBUG_PROMPTS = False  # Set to False to silence console output
LOG_PROMPTS_TO_DISK = True  # Set to False to stop writing files
PROMPT_LOG_DIR = None  # Initialized in initialize()


def _extract_dbms_notifications(response) -> str | None:
    """Return only received DBMS notifications from a response object, if present."""
    if response is None:
        return None

    notifications = []

    direct_notifications = getattr(response, "received_notifications", None)
    if direct_notifications:
        notifications.extend(direct_notifications)

    nested_response = getattr(response, "response", None)
    nested_notifications = getattr(nested_response, "received_notifications", None)
    if nested_notifications:
        notifications.extend(nested_notifications)

    if not notifications:
        return None

    lines = []
    for idx, notification in enumerate(notifications, start=1):
        if isinstance(notification, dict):
            level = notification.get("level") or notification.get("severity") or "info"
            source = notification.get("source") or notification.get("from") or "DBMS"
            message = notification.get("message") or notification.get("text") or str(notification)
            lines.append(f"[{idx}] {level.upper()} {source}: {message}")
        else:
            lines.append(f"[{idx}] {notification}")

    return "\n".join(lines)


def log_prompt(call_id: int, direction: str, agent_name: str, **payload):
    """
    Log an outgoing prompt or incoming response.
    direction: 'REQUEST' or 'RESPONSE'
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    header = f"{'─' * 60}"

    # ── Console output ──
    if DEBUG_PROMPTS:
        logger = payload.get("logger")

        if logger is None:
            return

        logger.debug(f"\n{header}")
        logger.debug(f"🔍 [{timestamp}] CALL #{call_id} {direction} — agent: {agent_name}")
        logger.debug(header)

        if direction == "REQUEST":
            prompt = payload.get("prompt", "")
            instructions = payload.get("instructions", "")
            tool_names = payload.get("tool_names", [])
            response_format = payload.get("response_format", None)

            logger.debug(f"📏 Prompt length: {len(prompt):,} chars")
            logger.debug(f"📏 Instructions length: {len(instructions):,} chars")
            logger.debug(f"📏 Total context: {len(prompt) + len(instructions):,} chars")
            logger.debug(f"🔧 Tools: {', '.join(tool_names) if tool_names else 'none'}")
            logger.debug(f"📋 Response format: {response_format}")
            logger.debug(f"\n{'─' * 30} INSTRUCTIONS {'─' * 30}")
            logger.debug(instructions[:500] + ("..." if len(instructions) > 500 else ""))
            logger.debug(f"\n{'─' * 30} PROMPT {'─' * 30}")
            logger.debug(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
            logger.debug(header)

        elif direction == "RESPONSE":
            response = payload.get("response", "")
            duration = payload.get("duration_ms", 0)
            logger.debug(f"⏱️  Duration: {duration:,.0f}ms")
            logger.debug(f"📏 Response length: {len(str(response)):,} chars")
            logger.debug(f"\n{'─' * 30} RESPONSE {'─' * 30}")
            notification_text = _extract_dbms_notifications(response)
            if notification_text:
                logger.debug(notification_text)
            else:
                logger.debug("No DBMS notifications received.")
            logger.debug(header)

        elif direction == "ERROR":
            error = payload.get("error", "")
            logger.debug(f"❌ Error: {error}")
            logger.debug(header)

    # ── Disk output (full, untruncated) ──
    if LOG_PROMPTS_TO_DISK and PROMPT_LOG_DIR is not None:
        PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = (
            PROMPT_LOG_DIR
            / f"{ts}_call{call_id:04d}_{direction.lower()}_{agent_name}.md"
        )

        with open(filename, "w") as f:
            f.write(f"# Call #{call_id} — {direction} — {agent_name}\n")
            f.write(f"**Timestamp:** {timestamp}\n\n")

            if direction == "REQUEST":
                prompt = payload.get("prompt", "")
                instructions = payload.get("instructions", "")
                tool_names = payload.get("tool_names", [])
                response_format = payload.get("response_format", None)

                f.write(f"**Prompt length:** {len(prompt):,} chars\n")
                f.write(f"**Instructions length:** {len(instructions):,} chars\n")
                f.write(
                    f"**Total context:** {len(prompt) + len(instructions):,} chars\n"
                )
                f.write(
                    f"**Tools:** {', '.join(tool_names) if tool_names else 'none'}\n"
                )
                f.write(f"**Response format:** {response_format}\n\n")
                f.write("## Instructions\n\n")
                f.write(f"```\n{instructions}\n```\n\n")
                f.write("## Prompt\n\n")
                f.write(f"```\n{prompt}\n```\n\n")

            elif direction == "RESPONSE":
                response = payload.get("response", "")
                duration = payload.get("duration_ms", 0)
                f.write(f"**Duration:** {duration:,.0f}ms\n")
                f.write(f"**Response length:** {len(str(response)):,} chars\n\n")
                f.write("## Response\n\n")
                f.write(f"```\n{response}\n```\n\n")

            elif direction == "ERROR":
                error = payload.get("error", "")
                f.write(f"## Error\n\n```\n{error}\n```\n\n")
