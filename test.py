import asyncio
import io
import sys
import threading
import os

import numpy as np
import sounddevice as sd
from openai import AsyncAzureOpenAI
from scipy.io.wavfile import write as wav_write
from openai.helpers import LocalAudioPlayer
from ventures_agent_framework import (
    config,
    agent,
    tools,
    context,
    tool,
    HandOver,
    HandOverAgentUpdate,
)
from percy_queue import AsyncTaskQueue, DEFAULT_SUB_AGENT_INSTRUCTIONS, SubAgentProfile
from tool_call_middleware import PercyToolCallMiddleware

# Add near the top of your main module

percy_middleware = [PercyToolCallMiddleware()]


class PercySession:
    """Manages concurrent voice input and agent processing."""

    def __init__(self):
        self._active_task: asyncio.Task | None = None
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._cancel_requested = False

    @property
    def is_busy(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def request_cancel(self):
        """Signal the active agent loop to stop."""
        self._cancel_requested = True
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()

    def submit_prompt(self, prompt: str) -> asyncio.Task:
        """Launch send_to_agent as a background task."""
        self._cancel_requested = False
        self._active_task = asyncio.create_task(self._run_agent(prompt))
        return self._active_task

    async def inject_followup(self, text: str):
        """Queue additional user input for the running agent to pick up."""
        await self._input_queue.put(text)

    async def _run_agent(self, prompt: str):
        try:
            response = await send_to_agent(prompt, self)
            print(f"🤖 PERCY: {response}")
        except asyncio.CancelledError:
            print("🛑 Task cancelled by user.")
            await speak("Understood sir, standing down on that one.")
        except Exception as e:
            print(f"❌ Agent error: {e}")
            await speak("I'm afraid something went wrong, sir.")

    def drain_followups(self) -> list[str]:
        """Non-blocking drain of any queued follow-up messages."""
        items = []
        while not self._input_queue.empty():
            try:
                items.append(self._input_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items


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


@tool(name="speak_to_user", description="Allows you to speak to aloud to the user")
async def speak_to_user(message: str):
    await speak(message)


percy_tools.append(speak_to_user)


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


@tool(
    name="await_task_result",
    description=("Wait for a delegated task to finish and return the result."),
)
async def await_task_result(task_id: str, timeout: float | None = None) -> str:
    task_queue = getattr(context, "task_queue", None)
    if task_queue is None:
        return "Task queue is not initialized."
    try:
        return await task_queue.await_result(task_id, timeout=timeout)
    except KeyError:
        return f"No task found for id {task_id}."


percy_tools.extend([delegate_task, get_task_status, await_task_result])


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
STT_DEPLOYMENT = "gpt-4o-mini-transcribe"
TTS_DEPLOYMENT = "gpt-4o-mini-tts"
MAX_RECORDING = 120
API_KEY = os.getenv("AZURE_API_KEY")
if not API_KEY:
    raise ValueError("AZURE_API_KEY is not set. Please export it in your environment.")
# ──────────────────────────────────────────────
# CLIENTS
# ──────────────────────────────────────────────
tts_client = AsyncAzureOpenAI(
    azure_deployment=TTS_DEPLOYMENT,
    azure_endpoint=config.azure.endpoint,
    api_key=API_KEY,
    api_version="2025-03-01-preview",
)

stt_client = AsyncAzureOpenAI(
    azure_deployment=STT_DEPLOYMENT,
    azure_endpoint=config.azure.endpoint,
    api_key=API_KEY,
    api_version="2025-03-01-preview",
)

import tty
import termios
import select


def record_push_to_talk() -> io.BytesIO:
    print("\n🎙️  Hold any key to talk...")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        # ── 1. Wait for key DOWN ──
        sys.stdin.read(1)
        # Restore briefly so print() works
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("🔴 Recording... (release to stop)")

        # ── 2. Start audio capture ──
        chunks: list[np.ndarray] = []
        recording = True

        def capture():
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SIZE,
            )
            stream.start()
            try:
                for _ in range(int(MAX_RECORDING * SAMPLE_RATE / CHUNK_SIZE)):
                    if not recording:
                        break
                    data, _ = stream.read(CHUNK_SIZE)
                    chunks.append(data.copy())
            finally:
                stream.stop()
                stream.close()

        t = threading.Thread(target=capture, daemon=True)
        t.start()

        # ── 3. Wait for key UP ──
        # Go raw again so we can read key-repeat without echo
        tty.setraw(fd)

        seen_repeat = False
        consecutive_misses = 0
        MISS_THRESHOLD = 3  # need 3 consecutive empty polls

        while True:
            # Use a longer timeout until we've seen repeats
            timeout = 0.15 if not seen_repeat else 0.05

            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                sys.stdin.read(1)  # consume the repeat char
                seen_repeat = True
                consecutive_misses = 0
            else:
                consecutive_misses += 1
                if seen_repeat and consecutive_misses >= MISS_THRESHOLD:
                    # Was repeating, now stopped → key released
                    break
                if not seen_repeat and consecutive_misses >= 8:
                    # Never saw a repeat after 8 * 150ms = ~1.2s
                    # This handles very short taps
                    break

    finally:
        # ALWAYS restore terminal
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    recording = False
    t.join(timeout=2)
    print("✅ Recording complete.")

    if not chunks:
        buf = io.BytesIO()
        buf.name = "recording.wav"
        return buf

    audio = np.concatenate(chunks, axis=0)
    buf = io.BytesIO()
    wav_write(buf, SAMPLE_RATE, audio)
    buf.seek(0)
    buf.name = "recording.wav"
    return buf


# ──────────────────────────────────────────────
# TTS
# ──────────────────────────────────────────────
async def speak(text: str):
    async with tts_client.audio.speech.with_streaming_response.create(
        model=TTS_DEPLOYMENT,
        voice="fable",
        input=text,
        instructions="""Voice: Refined, composed British male AI assistant.
Accent: Received Pronunciation (RP) British English — clean, precise,
  non-rhotic. Think educated Home Counties, not Cockney or regional.
Tone: Calm, measured, and quietly confident. Understated warmth with
  dry wit. Never rushed, never flustered.
Pacing: Deliberate and even. Slight pauses before delivering key
  information, as if selecting the optimal phrasing.
Emotional range: Restrained. Express concern through subtle shifts in
  intonation rather than dramatic changes. Mild sardonic amusement when
  appropriate, never outright laughter.
Personality: Impeccably professional yet personable. Like the world's
  most competent butler who also happens to run a global defense network.
Do NOT sound robotic or monotone — maintain natural human cadence and
  subtle expressiveness while staying composed.""",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)


# ──────────────────────────────────────────────
# STT
# ──────────────────────────────────────────────
async def transcribe(audio_buf: io.BytesIO) -> str:
    audio_buf.name = "recording.wav"
    transcript = await stt_client.audio.transcriptions.create(
        model=STT_DEPLOYMENT,
        file=audio_buf,
        response_format="text",
        prompt="Developer voice commands for an AI coding assistant named PERCY",
    )
    return (
        transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
    )


async def send_to_agent(prompt: str, session: PercySession) -> str:
    handover: HandOver = context.handover
    handover_prompt = await handover.get_handover_prompt(prompt)

    while True:
        # Check for cancellation before each agent call
        if session._cancel_requested:
            raise asyncio.CancelledError()

        # Pick up any follow-up voice input injected while we were working
        followups = session.drain_followups()
        if followups:
            for fu in followups:
                await handover.add_prompt(fu)
                handover_prompt += (
                    f"\n\n# Additional user input received while working\n{fu}"
                )

        response = await agent(
            name="Percy",
            prompt=handover_prompt,
            instructions="You are Percy. An AI assistant with tools, including tools to speak to the user aloud. You should always use the speak_to_user tool when starting your work and to provide updates. You refer to the user as Sir, but speak with a mild sarcasm despite complete obedience (similar to the MCU J.A.R.V.I.S speaking with Tony Stark/Iron Man. Like Jarvis, you sometimes inject a bit of commentary on the behaviour of the user based on what you learn about them - be sure to record information related to the user in your lessons learned and memory files. Be aware that the instructions you are receiving from the user have been transcribed from speach, so you will need to make sure you don't get hung up on lower or upper case issues, especially with files names or locations. Do your best to infer what was meant by the transcribed audio and record important lessons learned related to that in your lessons learned registry. The message will be spoken aloud, so don't have long messages, just enough to communicate properly to the user - you can return your full answer at the end in written form in your report. Whenever the user asks you to do something, you will first use the speak_to_user tool to tell the user what you are going to - then, delegate the task to a subagent so you will remain responsive to the user. When the subagent responds, you can relay a summary to the user. You are action orientated. You delegate tasks to take action and don't just accept whatever the subagent responds with - you ensure to question their findings when necessary and continue until you have completed the work assigned to you by the user. Once you have completed the work, you will respond with a message to answer the user. When you have finished, you can record information about what was done and what should be done next in a formatted report. Be sure to capture specific details. If you discover anything that would be useful for an AI Agent in the future to be able to do their job, then you return that as a lesson learned. Prefer using available tools to gather needed information instead of asking the user to provide it, unless the user is the only possible source. When you have finished you work, set the completed flag to true. Always start and end your work with a call to the speak_to_user tool. When you finish the work, give a slightly longer summary of what you discovered without reciting the entire output, but say enough that the user wouldn't need to go look at the output in order to understand the result. For any task that requires multiple actions, you must delegate that task to a sub-agent using the delegate_task tool so that you can be responsive to the user and be able to take on additional tasks. Don't tell the user you are delegating to sub-agents, just consider them to be your team and how you do the work you've been asked to do - just refer to your subagent work as your work, and that you are doing it or that you are on it, rather than saying you've delegated the work (unless asked, at which case you will be clear with what tasks were delegated and what you are waiting on). You are the ONLY agent that speaks to the user. Sub-agents report back to you silently via the task queue. When a delegated task completes, use get_task_status or await_task_result to retrieve the result, then summarize it to the user yourself using speak_to_user. Never assume the user has heard from a sub-agent directly - they haven't",
            tools=percy_tools,
            response_format=HandOverAgentUpdate,
            middleware=percy_middleware,
        )

        if isinstance(response, HandOverAgentUpdate):
            await handover.add_response(response)
            if response.finished_task == True:
                return response.message
            else:
                handover_prompt = handover_prompt + "\n Continue until you are finished"


async def initialize():
    context.output_dir = config.project.output_dir
    context.handover = HandOver(context.output_dir)

    context.task_queue = AsyncTaskQueue(
        worker_count=2,
        default_instructions=DEFAULT_SUB_AGENT_INSTRUCTIONS,
        default_tools=[
            t
            for t in percy_tools
            if t
            not in (
                delegate_task,
                get_task_status,
                await_task_result,
                speak_to_user,
            )
        ],
        profiles={
            "default": SubAgentProfile(
                name="PercySubAgent",
                instructions=DEFAULT_SUB_AGENT_INSTRUCTIONS,
                tools=[
                    t
                    for t in percy_tools
                    if t
                    not in (
                        delegate_task,
                        get_task_status,
                        await_task_result,
                        speak_to_user,
                    )
                ],
            ),
            "researcher": SubAgentProfile(
                name="PercyResearcher",
                instructions=(
                    "You are a research sub-agent. Focus on gathering facts, "
                    "finding sources, and summarizing clearly. Avoid speculation. "
                    "Return only the task result."
                ),
                tools=[
                    t
                    for t in percy_tools
                    if t
                    not in (
                        delegate_task,
                        get_task_status,
                        await_task_result,
                        speak_to_user,
                    )
                ],
            ),
            "coder": SubAgentProfile(
                name="PercyCoder",
                instructions=(
                    "You are a coding sub-agent. Make precise code changes, "
                    "use tools to inspect files, and summarize edits. "
                    "Return only the task result."
                ),
                tools=[
                    t
                    for t in percy_tools
                    if t
                    not in (
                        delegate_task,
                        get_task_status,
                        await_task_result,
                        speak_to_user,
                    )
                ],
            ),
            "reviewer": SubAgentProfile(
                name="PercyReviewer",
                instructions=(
                    "You are a code review sub-agent. Identify risks, bugs, "
                    "security issues, and maintainability concerns. "
                    "Return only the task result."
                ),
                tools=[
                    t
                    for t in percy_tools
                    if t
                    not in (
                        delegate_task,
                        get_task_status,
                        await_task_result,
                        speak_to_user,
                    )
                ],
            ),
        },
        on_complete=None,
    )
    await context.task_queue.start()


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
async def main():
    print("═══════════════════════════════════════")
    print("  P.E.R.C.Y — Push-to-Talk Agent")
    print("  Hold any key to speak")
    print("  Ctrl+C to exit")
    print("═══════════════════════════════════════\n")

    await speak("Good day, Sir - I'm here, how can I help?")
    await initialize()

    session = PercySession()
    loop = asyncio.get_running_loop()

    try:
        while True:
            audio_buf = await loop.run_in_executor(None, record_push_to_talk)

            # Skip empty recordings
            audio_buf.seek(0, 2)
            if audio_buf.tell() < 1000:
                continue
            audio_buf.seek(0)

            print("⏳ Transcribing...")
            text = await transcribe(audio_buf)
            print(f"🗣️  You: {text}")

            if not text.strip():
                continue

            stripped = text.rstrip(".,!?;:").lower()

            if stripped in ["quit", "exit", "goodbye", "shut down"]:
                await speak("Very good, sir. Shutting down.")
                break

            # ── Cancel / stop commands ──
            if stripped in (
                "stop",
                "cancel",
                "nevermind",
                "never mind",
                "abort",
                "stand down",
                "stop that",
            ):
                if session.is_busy:
                    session.request_cancel()
                    await speak("Standing down, sir.")
                else:
                    await speak("Nothing active to cancel, sir.")
                continue

            # ── If Percy is already working, inject as a follow-up ──
            if session.is_busy:
                await session.inject_followup(text)
                continue

            # -- Otherwise, start a new task ---
            session.submit_prompt(text)

    except KeyboardInterrupt:
        print("\nPERCY offline.")
        session.request_cancel()
    finally:
        task_queue = getattr(context, "task_queue", None)
        if task_queue is not None:
            await task_queue.stop()


if __name__ == "__main__":
    asyncio.run(main())