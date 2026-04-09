import sys
import asyncio
import pickle
from pathlib import Path
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent, AgentSession
from azure.identity import AzureCliCredential
from ventures_agent_framework import config, tools, logger


async def handle_prompt(prompt: str, agent: Agent, session: AgentSession):
    print()
    async for chunk in agent.run(prompt, stream=True, session=session):
        if chunk.text:
            print(chunk.text, end="", flush=True)
    print()


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


def persist_session_memory(session: AgentSession, memory_file: Path):
    if ".pk1" not in str(memory_file):
        logger.warning(
            f"Memory could not be persisted. File must be a pickle .pk1 file. Recieved {memory_file}"
        )
        return

    with open(memory_file, "wb") as f:
        pickle.dump(session.to_dict(), f)


async def main():

    agent_tools = [tools.run_shell_command, tools.apply_patch]

    agent = OpenAIChatClient(
        model=config.azure.deployment,
        azure_endpoint=config.azure.endpoint,
        credential=AzureCliCredential(),
    ).as_agent(
        instructions="You are Percy, an autonomous AI assistant that can run tools.",
        tools=agent_tools,
    )

    # Persistant Memory
    memory_file = (
        Path(config.sections["memory"]["session_file"]).expanduser().absolute()
    )
    session = load_persistant_memory(memory_file)
    if session is None:
        session = agent.create_session()

    while True:
        prompt = input("> ")
        if prompt in ["/quit", "/exit", "exit", "quit", "q"]:
            print("Goodbye")
            persist_session_memory(session, memory_file)
            sys.exit()
        await handle_prompt(prompt, agent, session)


if __name__ == "__main__":
    asyncio.run(main())
