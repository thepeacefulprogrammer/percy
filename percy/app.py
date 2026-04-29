import os
import sys
from pathlib import Path

from agent_framework import AgentSession
from memory import load_persistant_memory, persist_session_memory
from percy.agent import build_agent
from percy.ui.app import PercyApp
from ventures_agent_framework import config, logger, initialize_from_toml

initialize_from_toml()


def run() -> None:

    workdir = Path(os.environ.get("PERCY_WORKDIR", Path.cwd())).expanduser().resolve()
    config.project.output_dir = workdir
    agent = build_agent()
    memory_file = (
        Path(config.sections["memory"]["session_file"]).expanduser().absolute()
    )
    session = load_persistant_memory(memory_file)
    if session is None:
        session = agent.create_session()
    elif isinstance(session, AgentSession) and session.service_session_id:
        logger.info(
            "Clearing persisted service_session_id so Percy resumes from local session history."
        )
        session.service_session_id = None

    os.chdir(workdir)
    app = PercyApp(agent, session, memory_file)
    try:
        app.run()
    finally:
        persist_session_memory(session, memory_file)

    sys.exit()
