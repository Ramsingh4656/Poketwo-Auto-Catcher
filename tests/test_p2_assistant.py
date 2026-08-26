from __future__ import annotations

import asyncio
import inspect
import os
import sys
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("CATCH_CHANNEL_ID", "123456789012345678")
os.environ.setdefault("P2_ASSISTANT_ID", "854233015475109888")
os.environ.setdefault("CNN_CONFIDENCE_THRESHOLD", "0.85")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "bot"))

from bot import (  # noqa: E402
    BotState,
    POKETWO_BOT_ID,
    P2_ASSISTANT_ID,
    PokeCatcherBot,
)


class FakeChannel:
    def __init__(self, channel_id: int = 123456789012345678):
        self.id = channel_id
        self.sent: list[str] = []

    async def send(self, content: str):
        self.sent.append(content)


class P2AssistantFlowTests(unittest.IsolatedAsyncioTestCase):
    def make_uninitialized_bot(self) -> PokeCatcherBot:
        bot = object.__new__(PokeCatcherBot)
        bot.catch_channel_id = 123456789012345678
        bot.state = BotState.IDLE
        bot.logs = deque(maxlen=200)
        bot.stats = SimpleNamespace(
            total_p2_assistant=0,
            total_skipped=0,
            total_hint_used=0,
        )
        bot._spawn_channel_id = None
        bot._spawn_message_id = None
        bot._pending_pokemon = None
        bot._attempt_catch = AsyncMock()
        return bot

    async def test_hint_request_mentions_poketwo_only(self):
        bot = self.make_uninitialized_bot()
        channel = FakeChannel()

        await bot._request_poketwo_hint(channel)

        self.assertEqual(channel.sent, [f"<@{POKETWO_BOT_ID}> hint"])
        self.assertNotIn(str(P2_ASSISTANT_ID), channel.sent[0])
        self.assertFalse(hasattr(PokeCatcherBot, "_request_p2_hint"))
        source = inspect.getsource(PokeCatcherBot)
        self.assertNotIn("Requesting hint from P2 Assistant", source)

    async def test_uncertain_spawn_enters_hint_wait_and_requests_poketwo(self):
        bot = self.make_uninitialized_bot()
        channel = FakeChannel()
        message = SimpleNamespace(channel=channel, embeds=[])
        embed = SimpleNamespace(image=None, thumbnail=None)
        bot._download_spawn_image = AsyncMock(return_value=None)

        async def no_sleep(_seconds):
            return None

        with patch.object(asyncio, "sleep", new=no_sleep):
            await bot._handle_spawn(message, embed)

        self.assertEqual(channel.sent, [f"<@{POKETWO_BOT_ID}> hint"])
        self.assertNotIn(str(P2_ASSISTANT_ID), channel.sent[0])

    async def test_high_confidence_unprompted_p2_guess_catches_during_identification(self):
        bot = self.make_uninitialized_bot()
        bot.state = BotState.IDENTIFYING
        message = SimpleNamespace(
            channel=FakeChannel(),
            author=SimpleNamespace(id=P2_ASSISTANT_ID),
            content="Pikachu: 90%",
        )

        await bot.on_message(message)

        bot._attempt_catch.assert_awaited_once_with(message.channel, "pikachu")
        self.assertEqual(bot.stats.total_p2_assistant, 1)

    async def test_below_90_percent_p2_auto_guess_is_passive_and_does_not_catch(self):
        bot = self.make_uninitialized_bot()
        bot.state = BotState.IDENTIFYING
        message = SimpleNamespace(
            channel=FakeChannel(),
            author=SimpleNamespace(id=P2_ASSISTANT_ID),
            content="Pikachu: 89.99%",
        )

        await bot.on_message(message)

        bot._attempt_catch.assert_not_awaited()
        self.assertEqual(bot.state, BotState.IDENTIFYING)

    async def test_post_hint_p2_possible_pokemon_is_accepted_during_hint_wait(self):
        bot = self.make_uninitialized_bot()
        bot.state = BotState.WAITING_FOR_HINT
        message = SimpleNamespace(
            channel=FakeChannel(),
            author=SimpleNamespace(id=P2_ASSISTANT_ID),
            content="Possible Pokémon: Pikachu",
        )

        await bot.on_message(message)

        bot._attempt_catch.assert_awaited_once_with(message.channel, "pikachu")
        self.assertEqual(bot.stats.total_p2_assistant, 1)

    async def test_p2_messages_are_ignored_without_an_active_spawn(self):
        bot = self.make_uninitialized_bot()
        message = SimpleNamespace(
            channel=FakeChannel(),
            author=SimpleNamespace(id=P2_ASSISTANT_ID),
            content="Pikachu: 99%",
        )

        await bot.on_message(message)

        bot._attempt_catch.assert_not_awaited()
        self.assertEqual(bot.state, BotState.IDLE)

    async def test_poketwo_hint_is_still_processed_by_authoritative_resolver(self):
        bot = self.make_uninitialized_bot()
        bot.state = BotState.WAITING_FOR_HINT
        message = SimpleNamespace(
            channel=FakeChannel(),
            author=SimpleNamespace(id=POKETWO_BOT_ID),
            content="The pokémon is **p i k a c h u**.",
            embeds=[],
        )

        await bot.on_message(message)

        bot._attempt_catch.assert_awaited_once_with(message.channel, "pikachu")
        self.assertEqual(bot.stats.total_hint_used, 1)


if __name__ == "__main__":
    unittest.main()
