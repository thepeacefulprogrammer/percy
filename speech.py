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
from ventures_agent_framework import config

import tty
import termios
import select

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


# ──────────────────────────────────────────────
# TTS QUEUE — SERIALIZE ALL SPEECH
# ──────────────────────────────────────────────
_speech_queue: asyncio.Queue[str] = asyncio.Queue()
_speech_worker_task: asyncio.Task | None = None


async def _speech_worker():
    """Single worker that processes speech requests sequentially."""
    while True:
        text = await _speech_queue.get()
        if text == "__STOP__":
            _speech_queue.task_done()
            break
        try:
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
        except Exception as e:
            print(f"❌ Speech error: {e}")
        finally:
            _speech_queue.task_done()


async def start_speech_worker():
    global _speech_worker_task
    if _speech_worker_task is None:
        _speech_worker_task = asyncio.create_task(_speech_worker())


async def stop_speech_worker():
    global _speech_worker_task
    if _speech_worker_task is not None:
        await _speech_queue.put("__STOP__")
        await _speech_worker_task
        _speech_worker_task = None


async def speak(text: str):
    """Enqueue speech — returns immediately, speech plays in order."""
    await _speech_queue.put(text)


async def speak_and_wait(text: str):
    """Enqueue speech and wait for the queue to drain up to this item."""
    await _speech_queue.put(text)
    await _speech_queue.join()


# ──────────────────────────────────────────────
# STT
# ──────────────────────────────────────────────
async def transcribe(audio_buf: io.BytesIO) -> str:
    audio_buf.name = "recording.wav"
    transcript = await stt_client.audio.transcriptions.create(
        model=STT_DEPLOYMENT,
        file=audio_buf,
        response_format="text",
    )
    return (
        transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
    )


# ──────────────────────────────────────────────
# PUSH-TO-TALK RECORDING
# ──────────────────────────────────────────────
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
        tty.setraw(fd)

        seen_repeat = False
        consecutive_misses = 0
        MISS_THRESHOLD = 3

        while True:
            timeout = 0.15 if not seen_repeat else 0.05

            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                sys.stdin.read(1)
                seen_repeat = True
                consecutive_misses = 0
            else:
                consecutive_misses += 1
                if seen_repeat and consecutive_misses >= MISS_THRESHOLD:
                    break
                if not seen_repeat and consecutive_misses >= 8:
                    break

    finally:
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