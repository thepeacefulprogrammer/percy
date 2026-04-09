import pickle
from pathlib import Path
from ventures_agent_framework import logger
from agent_framework import AgentSession


def persist_session_memory(session: AgentSession, memory_file: Path):
    if ".pk1" not in str(memory_file):
        logger.warning(
            f"Memory could not be persisted. File must be a pickle .pk1 file. Recieved {memory_file}"
        )
        return

    with open(memory_file, "wb") as f:
        pickle.dump(session.to_dict(), f)


def load_persistant_memory(memory_file: Path) -> AgentSession | None:
    if ".pk1" not in str(memory_file):
        logger.warning(
            f"Memory could not be loaded. File must be a pickle .pk1 file. Recieved {memory_file}"
        )
        return None
    if not memory_file.exists():
        logger.warning(f"Memory file did not exist. Creating file: {memory_file}")
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        memory_file.touch()
        return None
    with open(memory_file, "rb") as f:
        session = AgentSession.from_dict(pickle.load(f))
        return session
