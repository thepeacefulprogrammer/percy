from pathlib import Path
from types import MethodType

from agent_framework.openai import OpenAIChatClient
from azure.identity import AzureCliCredential
from ventures_agent_framework import config, tools

from percy.compaction import build_history_and_compaction_providers
from percy.skills import get_skills_provider


def _patch_store_false_input_ids(chat_client: OpenAIChatClient) -> None:
    original_prepare_options = chat_client._prepare_options

    async def _patched_prepare_options(self, messages, options):
        run_options = await original_prepare_options(messages, options)
        if run_options.get("store") is False:
            for item in run_options.get("input", []):
                if isinstance(item, dict) and item.get("type") in {
                    "reasoning",
                    "function_call",
                }:
                    item.pop("id", None)
        return run_options

    chat_client._prepare_options = MethodType(_patched_prepare_options, chat_client)


def build_agent():
    agent_tools = [tools.run_shell_command, tools.apply_patch, tools.web_search]
    skills_dir = Path(__file__).resolve().parent.parent / "skills"
    skills_provider = get_skills_provider(skills_dir)

    context_providers = build_history_and_compaction_providers()
    if skills_provider is not None:
        context_providers.append(skills_provider)

    chat_client = OpenAIChatClient(
        model=config.azure.deployment,
        azure_endpoint=config.azure.endpoint,
        credential=AzureCliCredential(),
    )
    _patch_store_false_input_ids(chat_client)

    return chat_client.as_agent(
        instructions="You are Percy, an autonomous AI assistant that is participating in a bug bounty competition (hackmontyh.com). You can run tools. Reason over how best to approach the query and share your thought process. Details are in the README.md and the report.md doc. You must use the uv run hackmonty.py script as directed and within the bounds of the competition. Follow the other restrictions outlined in the README.md",
        tools=agent_tools,
        context_providers=context_providers,
        default_options={
            "reasoning": {"effort": "medium", "summary": "detailed"},
            # Persist history in AgentSession.state so we can safely resume from the
            # local pickle without depending on a remote Responses API conversation.
            "store": False,
        },
    )
