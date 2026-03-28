from datetime import datetime

# ──────────────────────────────────────────────
# DEBUG / PROMPT LOGGING
# ──────────────────────────────────────────────
DEBUG_PROMPTS = False  # Set to False to silence console output
LOG_PROMPTS_TO_DISK = True  # Set to False to stop writing files
PROMPT_LOG_DIR = None  # Initialized in initialize()


def log_prompt(call_id: int, direction: str, agent_name: str, **payload):
    """
    Log an outgoing prompt or incoming response.
    direction: 'REQUEST' or 'RESPONSE'
    """
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    header = f"{'─' * 60}"

    # ── Console output ──
    if DEBUG_PROMPTS:
        print(f"\n{header}")
        print(f"🔍 [{timestamp}] CALL #{call_id} {direction} — agent: {agent_name}")
        print(header)

        if direction == "REQUEST":
            prompt = payload.get("prompt", "")
            instructions = payload.get("instructions", "")
            tool_names = payload.get("tool_names", [])
            response_format = payload.get("response_format", None)

            print(f"📏 Prompt length: {len(prompt):,} chars")
            print(f"📏 Instructions length: {len(instructions):,} chars")
            print(f"📏 Total context: {len(prompt) + len(instructions):,} chars")
            print(f"🔧 Tools: {', '.join(tool_names) if tool_names else 'none'}")
            print(f"📋 Response format: {response_format}")
            print(f"\n{'─' * 30} INSTRUCTIONS {'─' * 30}")
            print(instructions[:500] + ("..." if len(instructions) > 500 else ""))
            print(f"\n{'─' * 30} PROMPT {'─' * 30}")
            print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
            print(header)

        elif direction == "RESPONSE":
            response = payload.get("response", "")
            duration = payload.get("duration_ms", 0)
            print(f"⏱️  Duration: {duration:,.0f}ms")
            print(f"📏 Response length: {len(str(response)):,} chars")
            print(f"\n{'─' * 30} RESPONSE {'─' * 30}")
            resp_str = str(response)
            print(resp_str[:1000] + ("..." if len(resp_str) > 1000 else ""))
            print(header)

        elif direction == "ERROR":
            error = payload.get("error", "")
            print(f"❌ Error: {error}")
            print(header)

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
