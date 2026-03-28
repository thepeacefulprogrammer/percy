import asyncio
import traceback
from percy import run_percy_turn
from speech import speak


# ──────────────────────────────────────────────
# PERCY SESSION — CONCURRENT ORCHESTRATOR
# ──────────────────────────────────────────────
class PercySession:
    """
    Spawns a new concurrent Percy turn for each batch of user commands.
    Never blocks the main loop.
    """

    def __init__(self):
        self._inbox: asyncio.Queue[str] = asyncio.Queue()
        self._dispatcher_task: asyncio.Task | None = None
        self._active_turns: dict[str, asyncio.Task] = {}
        self._shutdown = False
        self._turn_counter = 0

    async def start(self):
        self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())

    async def stop(self):
        self._shutdown = True
        await self._inbox.put("__SHUTDOWN__")
        # Cancel all active turns
        for _, task in self._active_turns.items():
            if not task.done():
                task.cancel()
        # Wait for dispatcher to exit
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except (asyncio.CancelledError, Exception):
                pass
        # Wait for active turns to finish
        if self._active_turns:
            await asyncio.gather(*self._active_turns.values(), return_exceptions=True)
            self._active_turns.clear()

    async def submit(self, text: str):
        """Push a voice command into Percy's inbox."""
        await self._inbox.put(text)

    def _drain_inbox(self) -> list[str]:
        """Non-blocking drain of all queued user messages."""
        items = []
        while not self._inbox.empty():
            try:
                msg = self._inbox.get_nowait()
                if msg != "__SHUTDOWN__":
                    items.append(msg)
            except asyncio.QueueEmpty:
                break
        return items

    def _cleanup_finished_turns(self):
        finished = [tid for tid, t in self._active_turns.items() if t.done()]
        for tid in finished:
            task = self._active_turns.pop(tid)
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc:
                    print(f"❌ Turn {tid} failed: {exc}")

    async def _dispatcher_loop(self):
        """
        Waits for inbox messages, then spawns a new concurrent Percy
        turn for each batch. Never blocks on agent calls.
        """
        while not self._shutdown:
            try:
                self._cleanup_finished_turns()

                # Wait for the first message
                msg = await self._inbox.get()
                if msg == "__SHUTDOWN__" or self._shutdown:
                    break

                new_messages = [msg]
                # Brief yield to let any rapid follow-ups arrive
                await asyncio.sleep(0.1)
                new_messages.extend(self._drain_inbox())

                if not new_messages:
                    continue

                # Spawn a new concurrent Percy turn
                self._turn_counter += 1
                turn_id = f"turn-{self._turn_counter}"
                turn_task = asyncio.create_task(
                    self._run_turn_safe(turn_id, new_messages)
                )
                self._active_turns[turn_id] = turn_task
                print(
                    f"📋 Dispatched {turn_id} with {len(new_messages)} command(s) "
                    f"({len(self._active_turns)} active turns)"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Dispatcher error: {e}")

    async def _run_turn_safe(self, turn_id: str, messages: list[str]):
        """Wrapper that catches errors so they don't crash the session."""
        try:
            await run_percy_turn(turn_id, messages)
        except asyncio.CancelledError:
            print(f"🛑 Turn {turn_id} cancelled.")
            await speak("Understood sir, standing down on that one.")
        except Exception as e:
            print(f"❌ Turn {turn_id} error: {e}")
            traceback.print_exc()
            await speak("I'm afraid something went wrong, sir. Standing by.")
        finally:
            self._active_turns.pop(turn_id, None)
