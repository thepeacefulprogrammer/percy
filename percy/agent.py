from agent_framework.openai import OpenAIChatClient
from azure.identity import AzureCliCredential
from ventures_agent_framework import config, tools


def build_agent():
    agent_tools = [tools.run_shell_command, tools.apply_patch, tools.web_search]
    return OpenAIChatClient(
        model=config.azure.deployment,
        azure_endpoint=config.azure.endpoint,
        credential=AzureCliCredential(),
    ).as_agent(
        instructions="You are Percy, an autonomous AI assistant that can run tools. Reason over how best to approach the query and share your thought process.",
        tools=agent_tools,
        default_options={
            "reasoning": {"effort": "medium", "summary": "detailed"},
        },
    )
