from collections.abc import Sequence

from agent_framework import (
    CompactionProvider,
    InMemoryHistoryProvider,
    SlidingWindowStrategy,
    ToolResultCompactionStrategy,
    annotate_message_groups,
)

from percy.config import load_local_config


def _config_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class SequentialCompactionStrategy:
    """Run simple compaction strategies in order."""

    def __init__(self, strategies: Sequence):
        self.strategies = list(strategies)

    @staticmethod
    def _normalize_message_ids(messages) -> None:
        for index, message in enumerate(messages):
            message.message_id = f"cmp_msg_{index}"

    async def __call__(self, messages) -> bool:
        self._normalize_message_ids(messages)
        annotate_message_groups(messages, force_reannotate=True)

        changed = False
        for strategy in self.strategies:
            changed = (await strategy(messages)) or changed
            annotate_message_groups(messages, force_reannotate=True)
        return changed


def build_history_and_compaction_providers() -> list:
    local_config = load_local_config()
    compaction_config = local_config.get("compaction", {})

    history_provider = InMemoryHistoryProvider(skip_excluded=True)
    providers: list = [history_provider]

    if not _config_bool(compaction_config.get("enabled"), True):
        return providers

    keep_last_groups = int(compaction_config.get("keep_last_groups", 24) or 24)
    keep_last_tool_call_groups = int(
        compaction_config.get("keep_last_tool_call_groups", 1) or 0
    )

    after_strategy = SequentialCompactionStrategy(
        [
            ToolResultCompactionStrategy(
                keep_last_tool_call_groups=keep_last_tool_call_groups
            ),
            SlidingWindowStrategy(keep_last_groups=keep_last_groups),
        ]
    )

    # Compaction provider must come before the history provider so its after_run
    # executes after history has stored the latest turn.
    return [
        CompactionProvider(
            after_strategy=after_strategy,
            history_source_id=history_provider.source_id,
        ),
        history_provider,
    ]
