import pickle
from pathlib import Path

from agent_framework import AgentSession
from ventures_agent_framework import logger


def _clear_service_session_id(data: dict, *, memory_file: Path) -> dict:
    sanitized = dict(data)
    if sanitized.get("service_session_id"):
        logger.info(
            f"Clearing persisted service_session_id for local resume: {memory_file}"
        )
        sanitized["service_session_id"] = None
    return sanitized


def persist_session_memory(session: AgentSession, memory_file: Path):
    if memory_file.suffix != ".pk1":
        logger.warning(
            f"Memory could not be persisted. File must be a pickle .pk1 file. Received {memory_file}"
        )
        return

    memory_file.parent.mkdir(parents=True, exist_ok=True)
    serialized = _clear_service_session_id(session.to_dict(), memory_file=memory_file)
    with open(memory_file, "wb") as f:
        pickle.dump(serialized, f)


def load_persistant_memory(memory_file: Path) -> AgentSession | None:
    if memory_file.suffix != ".pk1":
        logger.warning(
            f"Memory could not be loaded. File must be a pickle .pk1 file. Received {memory_file}"
        )
        return None

    if not memory_file.exists():
        logger.warning(f"Memory file did not exist. Creating file: {memory_file}")
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        memory_file.touch()
        return None

    if memory_file.stat().st_size == 0:
        logger.warning(f"Memory file was empty. Ignoring file: {memory_file}")
        return None

    try:
        with open(memory_file, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            logger.warning(
                f"Memory file did not contain a serialized session dict. Ignoring file: {memory_file}"
            )
            return None
        return AgentSession.from_dict(
            _clear_service_session_id(data, memory_file=memory_file)
        )
    except (EOFError, pickle.PickleError, ValueError, TypeError) as exc:
        logger.warning(f"Failed to load persisted memory from {memory_file}: {exc}")
        return None
