import sys
from pathlib import Path

from memory import load_persistant_memory, persist_session_memory
from percy.agent import build_agent
from percy.ui.app import PercyApp
from ventures_agent_framework import config


def run() -> None:
    agent = build_agent()
    memory_file = Path(config.sections["memory"]["session_file"]).expanduser().absolute()
    session = load_persistant_memory(memory_file)
    if session is None:
        session = agent.create_session()

    app = PercyApp(agent, session, memory_file)
    try:
        app.run()
    finally:
        persist_session_memory(session, memory_file)

    sys.exit()
