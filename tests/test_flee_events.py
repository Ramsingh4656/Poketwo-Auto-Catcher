"""Offline tests for Poketwo flee-event detection and state machine recovery.

Run: python -m unittest tests.test_flee_events
  or: python tests/test_flee_events.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
import unittest
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bot"))

# Silence loggers
logging.getLogger("bot").setLevel(logging.CRITICAL)
logging.getLogger("pokemon_data").setLevel(logging.CRITICAL)

# Stub the predictor module
_stub = types.ModuleType("predictor")
class _StubModulePredictor:
    def __init__(self, *a, **k):
        self.loaded = False
_stub.PokemonPredictor = _StubModulePredictor
sys.modules["predictor"] = _stub

# Environment setup
os.environ.setdefault("CATCH_CHANNEL_ID", "111,222")
os.environ.setdefault("CNN_CONFIDENCE_THRESHOLD", "0.85")

import bot  # noqa: E402
from pokemon_data import resolve_authoritative_name  # noqa: E402

# Determinism
bot.DISTRACTION_CHANCE = 0.0
bot.MIN_DELAY = bot.MAX_DELAY = 0.0

POKETWO = bot.POKETWO_BOT_ID
P2 = bot.P2_ASSISTANT_ID
IDLE = bot.BotState.IDLE
IDENTIFYING = bot.BotState.IDENTIFYING
WAITING_FOR_HINT = bot.BotState.WAITING_FOR_HINT
WAITING_FOR_RESULT = bot.BotState.WAITING_FOR_RESULT

# Stub dependencies
_real_sleep = asyncio.sleep

async def _fast_sleep(secs=0, *a, **k):
    # Allow yielding control but don't pause
    await _real_sleep(0)

def run(coro, collapse_sleep=True):
    """Run one coroutine to completion with optional asyncio.sleep collapsing."""
    if collapse_sleep:
        asyncio.sleep = _fast_sleep
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.sleep = _real_sleep

async def pump(n=12):
    """Advance the event loop several ticks."""
    for _ in range(n):
        await _real_sleep(0)

class FakeAuthor:
    def __init__(self, aid):
        self.id = aid

class _FakeTyping:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False

class FakeChannel:
    def __init__(self, cid, name):
        self.id = cid
        self.name = name
        self.sent = []

    async def send(self, content):
        self.sent.append(content)

    def typing(self):
        return _FakeTyping()

class FakeEmbedImage:
    def __init__(self, url):
        self.url = url

class FakeEmbed:
    def __init__(self, title=None, description=None, image_url=None):
        self.title = title
        self.description = description
        self.image = FakeEmbedImage(image_url) if image_url else None
        self.thumbnail = None

class FakeMessage:
    def __init__(self, channel, author_id, content="", embeds=None):
        self.channel = channel
        self.author = FakeAuthor(author_id)
        self.content = content
        self.embeds = embeds or []

class StubPredictor:
    def __init__(self, loaded=True, best=None, top=None):
        self.loaded = loaded
        self._best = best
        self._top = top or []

    async def predict_best(self, image_bytes, min_confidence=0.85):
        await _real_sleep(0)
        return self._best

    async def predict(self, image_bytes, top_k=3):
        await _real_sleep(0)
        return self._top

class FakeBot(bot.PokeCatcherBot):
    @property
    def user(self):
        return FakeAuthor(999999)

    async def _download_spawn_image(self, embed, channel):
        return b"FAKE_IMAGE"

def make_bot(channel_ids, predictor, fake_image=b"IMG"):
    b = FakeBot.__new__(FakeBot)
    b.predictor = predictor
    b.stats = bot.Stats()
    b.logs = deque(maxlen=200)
    b.catch_channel_ids = set(channel_ids)
    b._sessions = {cid: bot.ChannelSession() for cid in channel_ids}
    b._http_session = None
    b._fake_image = fake_image
    return b

class FleeEventTests(unittest.TestCase):
    def test_standalone_flee_message_resets_state(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor())
            ch = FakeChannel(111, "general")
            
            # Start in WAITING_FOR_HINT
            b._sessions[111].state = WAITING_FOR_HINT
            b._sessions[111].spawn_id = 1
            
            # Send standalone flee content
            msg = FakeMessage(ch, POKETWO, content="The wild Pokémon fled!")
            await b.on_message(msg)
            return b
            
        b = run(scenario())
        self.assertEqual(b._sessions[111].state, IDLE)
        self.assertEqual(b._sessions[111].spawn_id, 2)
        self.assertEqual(b.stats.total_fled, 1)
        self.assertTrue(any("Previous spawn fled" in log for log in b.logs))

    def test_combined_flee_and_spawn_embed_processed(self) -> None:
        async def scenario():
            # CNN returns a confident Pikachu
            b = make_bot([111], StubPredictor(best=(resolve_authoritative_name("pikachu"), 0.95)))
            ch = FakeChannel(111, "general")
            
            # Start in WAITING_FOR_HINT
            b._sessions[111].state = WAITING_FOR_HINT
            b._sessions[111].spawn_id = 4
            
            # Combined message (flee title, new spawn embed)
            embed = FakeEmbed(
                title="Wild Skwovet fled. A new wild pokémon has appeared!",
                image_url="http://assets/pikachu.png"
            )
            msg = FakeMessage(ch, POKETWO, embeds=[embed])
            await b.on_message(msg)
            return b, ch
            
        b, ch = run(scenario())
        # The state should end up as IDLE because of the collapsed 10s catch timeout
        self.assertEqual(b._sessions[111].state, IDLE)
        self.assertEqual(b._sessions[111].spawn_id, 6) # Increment 1 for flee, 1 for new spawn
        self.assertEqual(b.stats.total_fled, 1)
        self.assertEqual(b.stats.total_cnn_correct, 1)
        # Catch command sent for Pikachu
        self.assertTrue(any("catch pikachu" in cmd for cmd in ch.sent))

    def test_per_channel_isolation(self) -> None:
        async def scenario():
            b = make_bot([111, 222], StubPredictor())
            cha = FakeChannel(111, "chan-a")
            chb = FakeChannel(222, "chan-b")
            
            # Both channels busy
            b._sessions[111].state = WAITING_FOR_HINT
            b._sessions[111].spawn_id = 5
            b._sessions[222].state = WAITING_FOR_HINT
            b._sessions[222].spawn_id = 10
            
            # Flee on channel A
            msg = FakeMessage(cha, POKETWO, content="Wild Pikachu fled.")
            await b.on_message(msg)
            return b

        b = run(scenario())
        # Channel 111 reset to IDLE
        self.assertEqual(b._sessions[111].state, IDLE)
        self.assertEqual(b._sessions[111].spawn_id, 6)
        
        # Channel 222 unaffected
        self.assertEqual(b._sessions[222].state, WAITING_FOR_HINT)
        self.assertEqual(b._sessions[222].spawn_id, 10)
        self.assertEqual(b.stats.total_fled, 1)

    def test_old_timeout_does_not_clobber_new_spawn(self) -> None:
        event30 = asyncio.Event()
        event10 = asyncio.Event()
        original_sleep = asyncio.sleep

        async def custom_sleep(secs, *a, **k):
            if secs == 30:
                await event30.wait()
            elif secs == 10:
                await event10.wait()
            else:
                await _real_sleep(0)

        async def scenario():
            b = make_bot([111], StubPredictor(best=None)) # CNN uncertain
            ch = FakeChannel(111, "general")
            
            # 1. Trigger spawn. CNN is uncertain, so it enters WAITING_FOR_HINT
            # and runs: await asyncio.sleep(30)
            task1 = asyncio.create_task(b.on_message(FakeMessage(
                ch, POKETWO, embeds=[FakeEmbed("A wild pokémon has appeared!", image_url="http://x/y.png")]
            )))
            await pump() # let task1 run up to the sleep(30)
            
            self.assertEqual(b._sessions[111].state, WAITING_FOR_HINT)
            first_spawn_id = b._sessions[111].spawn_id
            
            # 2. Previous spawn flees. Resets to IDLE and increments spawn_id
            await b.on_message(FakeMessage(ch, POKETWO, content="The wild Pokémon fled!"))
            self.assertEqual(b._sessions[111].state, IDLE)
            self.assertNotEqual(b._sessions[111].spawn_id, first_spawn_id)
            
            # 3. New spawn starts in the same channel, enters IDENTIFYING
            # (In this mock, CNN has high confidence so it catches and enters WAITING_FOR_RESULT)
            b.predictor = StubPredictor(best=(resolve_authoritative_name("pikachu"), 0.95))
            task2 = asyncio.create_task(b.on_message(FakeMessage(
                ch, POKETWO, embeds=[FakeEmbed("A wild pokémon has appeared!", image_url="http://x/z.png")]
            )))
            await pump()
            
            self.assertEqual(b._sessions[111].state, WAITING_FOR_RESULT)
            second_spawn_id = b._sessions[111].spawn_id
            
            # 4. Wait for the old timeout (task1) to complete (we advance task1)
            # Since sleep is NOT collapsed for 30s, we release event30
            event30.set()
            await task1
            
            # The state must remain WAITING_FOR_RESULT, not get clobbered back to IDLE by task1!
            self.assertEqual(b._sessions[111].state, WAITING_FOR_RESULT)
            self.assertEqual(b._sessions[111].spawn_id, second_spawn_id)
            
            # Clean up task2
            event10.set()
            await task2
            return b

        # Patch asyncio.sleep during this test
        asyncio.sleep = custom_sleep
        try:
            run(scenario(), collapse_sleep=False)
        finally:
            asyncio.sleep = original_sleep

if __name__ == "__main__":
    unittest.main()
