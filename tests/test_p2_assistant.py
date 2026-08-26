"""Offline behavioral tests for the P2 Assistant early-signal fast path.

These drive the REAL async handlers in bot.py (on_message / _handle_spawn /
_handle_p2_assistant / _handle_hint) with fake Discord objects. No Discord
login and no network: the predictor and the spawn-image download are stubbed,
and asyncio.sleep is collapsed to an instant yield so timing-based flows run
immediately while still letting concurrent tasks interleave.

Run:  python -m unittest tests.test_p2_assistant
  or: python tests/test_p2_assistant.py
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

# Keep the bot's own logger quiet: _log() appends to the in-memory deque BEFORE
# calling the logger, so assertions on bot.logs still work while stderr stays
# clean (and we dodge cp1252 mojibake on accented log strings under a Windows
# console).
logging.getLogger("bot").setLevel(logging.CRITICAL)
logging.getLogger("pokemon_data").setLevel(logging.CRITICAL)

# Stub the predictor module so importing bot never loads onnxruntime or a model.
_stub = types.ModuleType("predictor")


class _StubModulePredictor:
    def __init__(self, *a, **k):
        self.loaded = False


_stub.PokemonPredictor = _StubModulePredictor
sys.modules["predictor"] = _stub

# A valid channel so `import bot` doesn't SystemExit at module load.
os.environ.setdefault("CATCH_CHANNEL_ID", "111111111111111111")
os.environ.setdefault("CNN_CONFIDENCE_THRESHOLD", "0.85")

import bot  # noqa: E402
from pokemon_data import get_best_hint_match, resolve_authoritative_name  # noqa: E402

# Determinism: no random distraction branch, no artificial identify delay.
bot.DISTRACTION_CHANCE = 0.0
bot.MIN_DELAY = bot.MAX_DELAY = 0.0

POKETWO = bot.POKETWO_BOT_ID
P2 = bot.P2_ASSISTANT_ID
IDLE = bot.BotState.IDLE
IDENTIFYING = bot.BotState.IDENTIFYING
WAITING_FOR_HINT = bot.BotState.WAITING_FOR_HINT

# Expected authoritative resolutions, computed once from the live 936-label map.
EXP_SHROOM = resolve_authoritative_name("Shroomish")
EXP_PIKA = resolve_authoritative_name("pikachu")
EXP_HINT = get_best_hint_match("The pokémon is **pikachu**")


# ── asyncio.sleep collapsing ──────────────────────────────────────────────────
_real_sleep = asyncio.sleep


async def _fast_sleep(_secs=0, *a, **k):
    await _real_sleep(0)


def run(coro):
    """Run one coroutine to completion with asyncio.sleep collapsed to a yield."""
    asyncio.sleep = _fast_sleep
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.sleep = _real_sleep


async def pump(n=12):
    """Let the event loop advance parked tasks a few steps."""
    for _ in range(n):
        await _real_sleep(0)


# ── Fake Discord objects ─────────────────────────────────────────────────────
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
    def __init__(self, title=None, image_url=None):
        self.title = title
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
        self._best = best          # (name, conf) or None
        self._top = top or []

    async def predict_best(self, image_bytes, min_confidence=0.85):
        await _real_sleep(0)
        return self._best

    async def predict(self, image_bytes, top_k=3):
        await _real_sleep(0)
        return self._top


class BlockingPredictor:
    """A predictor whose predict_best() parks until release() is called.

    Uses an asyncio.Event (a genuine suspension our sleep patch never touches),
    letting a test hold a spawn handler inside CNN inference while it injects a
    competing P2 Assistant signal -- the real preemption race.
    """

    def __init__(self, best):
        self.loaded = True
        self._best = best
        self._gate = asyncio.Event()

    def release(self):
        self._gate.set()

    async def predict_best(self, image_bytes, min_confidence=0.85):
        await self._gate.wait()
        return self._best

    async def predict(self, image_bytes, top_k=3):
        return []


# discord.Client.user is a read-only property, so it can't be assigned on an
# instance. Shadow it with a property on the subclass returning a fixed fake.
_FAKE_USER = FakeAuthor(999999)


class FakeBot(bot.PokeCatcherBot):
    @property
    def user(self):
        return _FAKE_USER

    async def _download_spawn_image(self, embed, channel):
        return self._fake_image


def make_bot(channel_ids, predictor, fake_image=b"IMG"):
    """Build a FakeBot without invoking discord.Client.__init__ (no network)."""
    b = FakeBot.__new__(FakeBot)
    b.predictor = predictor
    b.stats = bot.Stats()
    b.logs = deque(maxlen=200)
    b.catch_channel_ids = set(channel_ids)
    b._sessions = {cid: bot.ChannelSession() for cid in channel_ids}
    b._http_session = None
    b._fake_image = fake_image
    return b


def catches(channel):
    """Catch commands sent to the channel (excludes the P2 'hint' request)."""
    return [s for s in channel.sent if " catch " in s]


def logs_text(b):
    return "\n".join(b.logs)


def spawn_message(channel):
    return FakeMessage(
        channel, POKETWO,
        embeds=[FakeEmbed("A wild pokémon has appeared!", "http://x/y.png")],
    )


class P2AssistantEarlySignalTests(unittest.TestCase):
    def test_authoritative_map_and_threshold(self) -> None:
        # The resolution-dependent tests assume the live 936-label map is loaded.
        self.assertIsNotNone(EXP_SHROOM)
        self.assertIsNotNone(EXP_PIKA)
        self.assertIsNotNone(EXP_HINT)
        self.assertEqual(bot.P2_ASSISTANT_AUTO_CONFIDENCE, 0.90)

    def test_early_high_confidence_auto_catches_fast(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True, best=(EXP_PIKA, 0.99)))
            ch = FakeChannel(111, "general")
            b._sessions[111].state = IDENTIFYING  # spawn just detected
            await b.on_message(FakeMessage(ch, P2, content="Shroomish: 98.229%"))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [f"<@{POKETWO}> catch {EXP_SHROOM}"])
        self.assertIn("source: P2 Assistant early auto-message", logs_text(b))
        self.assertEqual(b.stats.total_p2_assistant, 1)
        self.assertEqual(b.stats.total_cnn_correct, 0)  # CNN was skipped entirely

    def test_early_auto_also_catches_during_hint_wait(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True))
            ch = FakeChannel(111, "general")
            b._sessions[111].state = WAITING_FOR_HINT
            await b.on_message(FakeMessage(ch, P2, content="Shroomish: 97.0%"))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [f"<@{POKETWO}> catch {EXP_SHROOM}"])

    def test_low_confidence_auto_is_ignored(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True))
            ch = FakeChannel(111, "general")
            b._sessions[111].state = IDENTIFYING
            await b.on_message(FakeMessage(ch, P2, content="Shroomish: 45.0%"))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [])
        self.assertEqual(b._sessions[111].state, IDENTIFYING)  # normal flow intact
        self.assertEqual(b.stats.total_p2_assistant, 0)
        self.assertIn("below the 90% bar", logs_text(b))

    def test_unresolvable_auto_is_ignored(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True))
            ch = FakeChannel(111, "general")
            b._sessions[111].state = IDENTIFYING
            await b.on_message(FakeMessage(ch, P2, content="Notarealmon: 99.9%"))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [])
        self.assertEqual(b._sessions[111].state, IDENTIFYING)
        self.assertIn("Ignored unrecognized P2 Assistant candidate", logs_text(b))

    def test_early_auto_ignored_while_idle(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True))
            ch = FakeChannel(111, "general")  # session defaults to IDLE (no spawn)
            await b.on_message(FakeMessage(ch, P2, content="Shroomish: 99.0%"))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [])
        self.assertEqual(b._sessions[111].state, IDLE)
        self.assertIn("no spawn awaiting identification", logs_text(b))

    def test_cnn_confident_path_unchanged(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True, best=(EXP_PIKA, 0.97)))
            ch = FakeChannel(111, "general")
            await b.on_message(spawn_message(ch))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [f"<@{POKETWO}> catch {EXP_PIKA}"])
        self.assertIn("source: CNN", logs_text(b))
        self.assertEqual(b.stats.total_cnn_correct, 1)
        self.assertEqual(b.stats.total_p2_assistant, 0)

    def test_cnn_uncertain_enters_hint_wait(self) -> None:
        # With sleeps collapsed the 30s hint window closes instantly and the
        # handler returns to IDLE, but the hint request it sent (and the log it
        # wrote the moment it flipped to WAITING_FOR_HINT) prove it got there.
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True, best=None, top=[("a", 0.4), ("b", 0.3)]))
            ch = FakeChannel(111, "general")
            await b.on_message(spawn_message(ch))
            return b, ch

        b, ch = run(scenario())
        self.assertIn(f"<@{P2}> hint", ch.sent)
        self.assertIn("Waiting for hint from Poketwo", logs_text(b))
        self.assertEqual(catches(ch), [])  # nothing caught from the CNN path

    def test_poketwo_hint_catches(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True))
            ch = FakeChannel(111, "general")
            b._sessions[111].state = WAITING_FOR_HINT
            await b.on_message(FakeMessage(ch, POKETWO, content="The pokémon is **pikachu**"))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [f"<@{POKETWO}> catch {EXP_HINT}"])
        self.assertIn("source: Poketwo hint", logs_text(b))
        self.assertEqual(b.stats.total_hint_used, 1)
        self.assertEqual(b.stats.total_p2_assistant, 0)

    def test_post_hint_fallback_unchanged(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True))
            ch = FakeChannel(111, "general")
            b._sessions[111].state = WAITING_FOR_HINT
            await b.on_message(FakeMessage(ch, P2, content="Possible Pokémon: pikachu"))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [f"<@{POKETWO}> catch {EXP_PIKA}"])
        self.assertIn("source: P2 Assistant post-hint message", logs_text(b))

    def test_post_hint_ignored_outside_hint_wait(self) -> None:
        async def scenario():
            b = make_bot([111], StubPredictor(loaded=True))
            ch = FakeChannel(111, "general")
            b._sessions[111].state = IDENTIFYING
            await b.on_message(FakeMessage(ch, P2, content="Possible Pokémon: pikachu"))
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(catches(ch), [])
        self.assertEqual(b._sessions[111].state, IDENTIFYING)
        self.assertIn("not awaiting a hint", logs_text(b))

    def test_per_channel_isolation(self) -> None:
        async def scenario():
            b = make_bot([111, 222], StubPredictor(loaded=True))
            cha = FakeChannel(111, "chan-a")
            chb = FakeChannel(222, "chan-b")
            b._sessions[111].state = IDENTIFYING
            b._sessions[222].state = IDENTIFYING
            await b.on_message(FakeMessage(cha, P2, content="Shroomish: 98.5%"))
            return b, cha, chb

        b, cha, chb = run(scenario())
        self.assertEqual(catches(cha), [f"<@{POKETWO}> catch {EXP_SHROOM}"])
        self.assertEqual(chb.sent, [])
        self.assertEqual(b._sessions[222].state, IDENTIFYING)
        self.assertIsNone(b._sessions[222].pending_pokemon)

    def test_high_conf_auto_preempts_running_cnn(self) -> None:
        # CNN would return 'ditto'; hold it inside inference, let the P2 auto-guess
        # ('Shroomish') win, then release CNN and prove ditto is never caught.
        captured = {}

        async def scenario():
            pred = BlockingPredictor(best=("ditto", 0.99))
            b = make_bot([111], pred)
            ch = FakeChannel(111, "general")
            task = asyncio.create_task(b.on_message(spawn_message(ch)))
            await pump()  # spawn handler parks inside predict_best, still IDENTIFYING
            captured["parked_state"] = b._sessions[111].state
            await b.on_message(FakeMessage(ch, P2, content="Shroomish: 98.0%"))
            pred.release()   # now let the stale CNN result come back
            await task       # guard must see state changed and stand down
            return b, ch

        b, ch = run(scenario())
        self.assertEqual(captured["parked_state"], IDENTIFYING)
        c = catches(ch)
        self.assertIn(f"<@{POKETWO}> catch {EXP_SHROOM}", c)
        self.assertNotIn(f"<@{POKETWO}> catch ditto", c)  # CNN result discarded
        self.assertEqual(len(c), 1)
        self.assertIn("catching early (skipping CNN/hint wait)", logs_text(b))
        self.assertIn("skipping CNN result", logs_text(b))


if __name__ == "__main__":
    unittest.main()
