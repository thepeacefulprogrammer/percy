import asyncio

from ventures_agent_framework import (
    config,
    context,
    HandOver,
)
from speech import (
    start_speech_worker,
    stop_speech_worker,
    speak_and_wait,
    record_push_to_talk,
    transcribe,
)
from percy_session import PercySession


# ──────────────────────────────────────────────
# INITIALIZATION
# ──────────────────────────────────────────────
async def initialize():
    global PROMPT_LOG_DIR

    context.output_dir = config.project.output_dir
    context.handover = HandOver(
        context.output_dir, deployment=config.sections["models"]["summary_deployment"]
    )

    # Set up prompt log directory
    PROMPT_LOG_DIR = context.output_dir / "prompt_logs"
    PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📝 Prompt logs writing to: {PROMPT_LOG_DIR}")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
async def main():
    print("═══════════════════════════════════════")
    print("  P.E.R.C.Y Agent")
    print("  Hold any key to speak")
    print("  Ctrl+C to exit")
    print("═══════════════════════════════════════\n")

    await start_speech_worker()
    await speak_and_wait("Good day, Sir — I'm here, how can I help?")
    await initialize()

    session = PercySession()
    await session.start()

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

            if stripped in ("quit", "exit", "goodbye", "shut down"):
                await speak_and_wait("Very good, sir. Shutting down.")
                break

            # Everything else goes into Percy's inbox
            await session.submit(text)

    except KeyboardInterrupt:
        print("\nPERCY offline.")
    finally:
        await session.stop()
        task_queue = getattr(context, "task_queue", None)
        if task_queue is not None:
            await task_queue.stop()
        await stop_speech_worker()


if __name__ == "__main__":
    asyncio.run(main())
