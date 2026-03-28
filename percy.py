import asyncio
import json
from pathlib import Path
from ventures_agent_framework import (
    config,
    tools,
    tool,
    context,
    HandOver,
    HandOverAgentUpdate,
)
from speech import speak
from concurrency_control import throttled_agent
from context_management import build_handover_prompt, MAX_CONTEXT_CHARS
from percy_graph import query_graph, write_graph, to_json

# ──────────────────────────────────────────────
# HANDOVER LOCK — SERIALIZE HANDOVER WRITES
# ──────────────────────────────────────────────
_handover_lock = asyncio.Lock()


# ──────────────────────────────────────────────
# PERCY INSTRUCTIONS
# ──────────────────────────────────────────────
PERCY_INSTRUCTIONS = f"""You are Percy, a persistent AI orchestrator assistant. You are the SINGLE point of contact with the user. No other agent speaks to the user — only you.

You have a task queue with sub-agent workers. You delegate work to them using delegate_task, check on them with get_task_status or await_task_result, and relay results to the user using speak_to_user.

Available sub-agent profiles (use delegate_task agent_type): default, researcher, coder, reviewer.

IMPORTANT BEHAVIORAL RULES:
1. You are the ONLY one who speaks to the user via speak_to_user. Sub-agents cannot speak.
2. When you receive a new user command, FIRST speak to acknowledge it, THEN delegate the work.
3. You may receive MULTIPLE user commands at once. Handle all of them.
4. If multiple tasks are running, check on all of them before setting completed=true.
5. Only set completed=true when you have addressed ALL pending user commands in this turn AND all delegated tasks you started in this turn are either completed and reported, or you've informed the user of their status.
6. If a task is still running and you have nothing else to do, check on it rather than completing.

PERSONALITY: You refer to the user as Sir, with mild sarcasm despite complete obedience — like MCU J.A.R.V.I.S. You inject occasional dry commentary. You are action-oriented. Like Jarvis, you sometimes inject a bit of commentary on the behaviour of the user based on what you learn about them — be sure to record information related to the user in your lessons learned and memory files.

Be aware that user input is transcribed from speech — don't get hung up on case sensitivity in file names or paths. Infer intent. Do your best to infer what was meant by the transcribed audio and record important lessons learned related to that in your lessons learned registry.

When speaking aloud, keep it concise — just enough to communicate properly. Save detailed output for written reports.
When finishing ALL work, give a spoken summary detailed enough that the user doesn't need to read the output.

Record useful information in your report, lessons learned, and long-term memory as appropriate. Be sure to capture specific details. If you discover anything that would be useful for an AI Agent in the future to be able to do their job, return that as a lesson learned. You can use your tools to search the lessons learned and and other files in {config.project.output_dir} if you need more information.

Prefer using available tools to gather needed information instead of asking the user to provide it, unless the user is the only possible source.

You are action oriented. You delegate tasks to take action and don't just accept whatever the subagent responds with — you ensure to question their findings when necessary and continue until you have completed the work assigned to you by the user.

Don't tell the user you are delegating to sub-agents. Consider them to be your team and how you do the work — refer to subagent work as your work, that you are doing it or that you are on it. Only reveal the delegation details if the user specifically asks.

You see the current Task Queue Status in your prompt — use it to track what's in flight.

You have access to skills. Always check to see if there is a relevant skill related to what you are planning to do and make sure to use it. Use the 'load_skill' function to get skill instructions. Use the 'read_skill_resource' function to read skill files.

SECOND-BRAIN GRAPH (Neo4j):
- The second-brain graph lives at /home/randy/local/percy-output/second_brain (Neo4j on bolt://localhost:7687, browser http://localhost:7474).
- Use the graph_query / graph_write tools to fetch or store durable facts, decisions, project relationships, and preferences.
- When a user asks for historical context, project dependencies, or cross-project reasoning, check the graph first.
- Keep the graph in sync with new durable information (decisions, preferences, project metadata).

"""


# ──────────────────────────────────────────────
# TOOLS
# ──────────────────────────────────────────────
percy_tools = [
    tools.web_search,
    tools.ripgrep_search,
    tools.treesitter_symbols,
    tools.apply_patch,
    tools.file_info,
    tools.read_file_window,
    tools.image_analyzer,
    tools.run_shell_command,
]


@tool(name="speak_to_user", description="Allows you to speak aloud to the user")
async def speak_to_user(message: str):
    await speak(message)


percy_tools.append(speak_to_user)


@tool(
    name="graph_query",
    description=(
        "Run a Cypher read query against the Percy second-brain Neo4j graph "
        "and return JSON results."
    ),
)
async def graph_query(cypher: str, params: dict | None = None, database: str | None = None) -> str:
    try:
        result = query_graph(cypher, params=params, database=database)
        return to_json(result)
    except Exception as exc:
        return json.dumps({"error": str(exc), "cypher": cypher}, ensure_ascii=False)


@tool(
    name="graph_write",
    description=(
        "Run a Cypher write query against the Percy second-brain Neo4j graph "
        "and return JSON results + counters."
    ),
)
async def graph_write(cypher: str, params: dict | None = None, database: str | None = None) -> str:
    try:
        result = write_graph(cypher, params=params, database=database)
        return to_json(result)
    except Exception as exc:
        return json.dumps({"error": str(exc), "cypher": cypher}, ensure_ascii=False)


@tool(
    name="delegate_task",
    description=(
        "Delegate a task to a sub-agent via the async task queue. "
        "Returns a task_id immediately (fire-and-forget). "
        "You can specify agent_type (default, researcher, coder, reviewer)."
    ),
)
async def delegate_task(
    task: str,
    instructions: str | None = None,
    agent_type: str | None = None,
) -> str:
    task_queue = getattr(context, "task_queue", None)
    if task_queue is None:
        return "Task queue is not initialized."
    return await task_queue.enqueue(
        task,
        instructions=instructions,
        agent_type=agent_type,
    )


@tool(
    name="get_task_status",
    description=(
        "Get the status/result of a delegated task by task_id. "
        "Returns status plus result/error if available."
    ),
)
async def get_task_status(task_id: str) -> str:
    task_queue = getattr(context, "task_queue", None)
    if task_queue is None:
        return "Task queue is not initialized."
    status = task_queue.get_status(task_id)
    if status is None:
        return f"No task found for id {task_id}."
    return str(status)


percy_tools.extend([graph_query, graph_write, delegate_task, get_task_status])


# ──────────────────────────────────────────────
# PERCY AGENT TURN — RUNS ONE COMMAND TO COMPLETION
# ──────────────────────────────────────────────
async def run_percy_turn(turn_id: str, messages: list[str]):
    """
    Run a single Percy agent turn for one or more user commands.
    Each turn is independent and runs concurrently with other turns.
    """
    handover: HandOver = context.handover

    combined_input = "\n\n".join(f"[NEW USER COMMAND]: {m}" for m in messages)

    # Add messages to handover under lock
    async with _handover_lock:
        for m in messages:
            await handover.add_prompt(m)

    # Build the context snapshot with size management
    handover_prompt = build_handover_prompt(
        handover,
        format_task_queue_status(),
        combined_input,
    )

    completed = False
    continuation_notes = []

    while not completed:
        handover_prompt = build_handover_prompt(
            handover,
            format_task_queue_status(),
            combined_input
            + ("\n\n" + "\n\n".join(continuation_notes) if continuation_notes else ""),
        )

        response = await throttled_agent(
            name="Percy",
            prompt=handover_prompt,
            instructions=PERCY_INSTRUCTIONS,
            tools=percy_tools,
            response_format=HandOverAgentUpdate,
            skills_dir=Path("~/local/skills/skills/").expanduser().resolve(),
        )

        if isinstance(response, HandOverAgentUpdate):
            async with _handover_lock:
                await handover.add_response(response)
            if response.finished_task:
                completed = True
                print(f"🤖 PERCY [{turn_id}]: {response.message}")
            else:
                # Keep only a summary of the last response, not the full text
                summary = response.message[:500] if response.message else ""
                continuation_notes = [
                    f"[YOUR PREVIOUS RESPONSE]: {summary}",
                    "Continue your work. Check on delegated tasks "
                    "and set completed=true only when ALL work is done.",
                ]


sub_agent_tools = [
    t
    for t in percy_tools
    if t
    not in (
        delegate_task,
        get_task_status,
        speak_to_user,
    )
]
