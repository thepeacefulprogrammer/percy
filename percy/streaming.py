def get_incremental_text(seen: dict[str, str], key: str, incoming: str | None) -> str:
    if not incoming:
        return ""

    previous = seen.get(key, "")
    if not previous:
        seen[key] = incoming
        return incoming

    if incoming.startswith(previous):
        seen[key] = incoming
        return incoming[len(previous) :]

    max_overlap = min(len(previous), len(incoming))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if previous.endswith(incoming[:size]):
            overlap = size
            break

    new_text = incoming[overlap:]
    seen[key] = previous + new_text
    return new_text


def join_stream_text(parts: dict[str, str], order: list[str]) -> str:
    return "\n\n".join(
        parts[key].strip() for key in order if parts.get(key, "").strip()
    )


def content_event_type(content) -> str | None:
    raw = getattr(content, "raw_representation", None)
    return getattr(raw, "type", None)
