import sys
import asyncio
from pathlib import Path
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent, AgentSession
from azure.identity import AzureCliCredential
from rich.console import Console
from rich.markdown import Markdown
from ventures_agent_framework import config, tools
from memory import load_persistant_memory, persist_session_memory


console = Console()


def _get_incremental_text(seen: dict[str, str], key: str, incoming: str | None) -> str:
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


async def handle_prompt(prompt: str, agent: Agent, session: AgentSession):
    console.print()

    usage = None
    reasoning_seen: dict[str, str] = {}
    showed_reasoning = False

    stream = agent.run(prompt, stream=True, session=session)
    async for chunk in stream:
        if chunk.contents:
            for index, content in enumerate(chunk.contents):
                if content.type == "text_reasoning":
                    key = f"reasoning:{content.id or index}"
                    incremental_text = _get_incremental_text(reasoning_seen, key, content.text)
                    if incremental_text:
                        console.print(incremental_text, style="blue", end="")
                        showed_reasoning = True
                elif content.type == "usage":
                    usage = content

    response = await stream.get_final_response()

    if response.text:
        if showed_reasoning:
            console.print()
        console.print(Markdown(response.text))

    if usage:
        console.print(f"Usage: {usage.usage_details}", style="dim")

    console.print()


async def main():

    agent_tools = [tools.run_shell_command, tools.apply_patch, tools.web_search]

    agent = OpenAIChatClient(
        model=config.azure.deployment,
        azure_endpoint=config.azure.endpoint,
        credential=AzureCliCredential(),
    ).as_agent(
        instructions="You are Percy, an autonomous AI assistant that can run tools. Reason over how best to approach the query and share your thought process.",
        tools=agent_tools,
        default_options={"reasoning": {"effort": "high", "summary": "detailed"}},
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
