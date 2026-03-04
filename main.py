import sys
import asyncio
import typer

from typing import Annotated
from ventures_agent_framework import logger, agent, tools, config, context
from percy_datatypes import HandOver, HandOverAgentUpdate

app = typer.Typer(invoke_without_command=True)

percy_tools = [
    tools.web_search,
    tools.ripgrep_search,
    tools.apply_patch,
    tools.file_info,
    tools.read_file_window,
    tools.user_update,
    tools.list_tree,
    tools.image_analyzer,
    tools.run_shell_command,
]


@app.callback()
def setup(
    mode: Annotated[
        bool,
        typer.Option("--mode", "-m", help="Some future mode support. Does nothing now"),
    ] = (False),
):
    if mode:
        logger.info("Some future mode support")
        return


async def send_to_agent(prompt: str) -> str:
    handover: HandOver = context.handover
    handover_prompt = await handover.get_handover_prompt(prompt)
    response = await agent(
        name="Percy",
        prompt=handover_prompt,
        instructions="You are Percy. An AI assistant with tools. You respond with a message to answer the user. You record information about what you did and what should be done next in a formatted report. Be sure to capture specific details. If you discover anything that would be useful for an AI Agent in the future to be able to do their job, then you return that as a lesson learned. Prefer using available tools to gather needed information instead of asking the user to provide it, unless the user is the only possible source.",
        tools=percy_tools,
        response_format=HandOverAgentUpdate,
    )

    if isinstance(response, HandOverAgentUpdate):
        await handover.add_response(response)
        return response.message
    return "No response from agent"


def initialize():
    context.output_dir = config.project.output_dir
    context.handover = HandOver(context.output_dir)


async def main():
    logger.info("Percy")
    initialize()
    while True:
        prompt = input(">>> ")
        if prompt.rstrip().lower() in ["quit", "exit", "q", "bye"]:
            sys.exit()
        response = await send_to_agent(prompt)
        print(f"Agent: {response}")


if __name__ == "__main__":
    app(standalone_mode=False)
    asyncio.run(main())
