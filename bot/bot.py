"""
bot.py — Discord selfbot that detects Poketwo spawns and catches Pokemon.
Uses CNN model as primary identifier; falls back to hint-matching.
WARNING: Selfbots violate Discord ToS — use at your own risk.
"""

import os, re, random, asyncio, logging, time
from enum import Enum, auto
from typing import Optional
from collections import deque

import aiohttp
import discord

from predictor import PokemonPredictor
from pokemon_data import get_best_hint_match

logger = logging.getLogger("bot")

POKETWO_BOT_ID = 716390085896962058

# ═══════════════════════════════════════════════════════════════════════════════
# SET YOUR CATCH CHANNEL ID HERE — the bot will ONLY catch in this channel.
# To find a channel ID: right-click the channel in Discord → Copy Channel ID
# Set to None to catch in ALL channels (not recommended).
# You can also set via environment variable: CATCH_CHANNEL_ID=123456789
# ═══════════════════════════════════════════════════════════════════════════════
CATCH_CHANNEL_ID = int(os.getenv("CATCH_CHANNEL_ID", "0")) or None

MIN_DELAY, MAX_DELAY = 2.0, 5.0
DISTRACTION_CHANCE = 0.05
DISTRACTION_DELAY = (3.0, 8.0)
CNN_CONFIDENCE_THRESHOLD = 0.70

SPAWN_RE = re.compile(r"A wild pok[eé]mon has appeared!", re.IGNORECASE)
HINT_RE = re.compile(r"The pok[eé]mon is \*\*.+\*\*", re.IGNORECASE)


class BotState(Enum):
    IDLE = auto()
    IDENTIFYING = auto()
    WAITING_FOR_HINT = auto()
    WAITING_FOR_RESULT = auto()


class Stats:
    def __init__(self):
        self.total_caught = 0
        self.total_cnn_correct = 0
        self.total_hint_used = 0
        self.total_skipped = 0
        self.start_time = time.time()

    def to_dict(self):
        return {
            "total_caught": self.total_caught,
            "cnn_catches": self.total_cnn_correct,
            "hint_catches": self.total_hint_used,
            "skipped": self.total_skipped,
            "uptime_seconds": int(time.time() - self.start_time),
        }


class PokeCatcherBot(discord.Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictor = PokemonPredictor()
        self.state = BotState.IDLE
        self.stats = Stats()
        self.logs = deque(maxlen=200)
        self._spawn_channel_id = None
        self._spawn_message_id = None
        self._pending_pokemon = None

        # Channel restriction
        self.catch_channel_id = CATCH_CHANNEL_ID

    def _log(self, msg, level="info"):
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        getattr(logger, level, logger.info)(msg)

    async def on_ready(self):
        self._log(f"Logged in as {self.user} (ID: {self.user.id})")
        self._log(f"Model loaded: {self.predictor.loaded}")
        if self.catch_channel_id:
            self._log(f"Catching ONLY in channel: {self.catch_channel_id}")
        else:
            self._log("WARNING: No channel set — catching in ALL channels!")

    async def on_message(self, message):
        # Only process messages from Poketwo
        if message.author.id != POKETWO_BOT_ID:
            return
        # Only process messages in the designated catch channel
        if self.catch_channel_id and message.channel.id != self.catch_channel_id:
            return

        if message.embeds:
            for embed in message.embeds:
                if embed.title and SPAWN_RE.search(embed.title):
                    await self._handle_spawn(message, embed)
                    return

        if self.state == BotState.WAITING_FOR_HINT:
            if message.content and HINT_RE.search(message.content):
                await self._handle_hint(message)
                return

        if self.state == BotState.WAITING_FOR_RESULT:
            if message.content and self.user and str(self.user) in message.content:
                if "caught" in message.content.lower():
                    self.stats.total_caught += 1
                    self._log(f"Caught! {message.content}")
                else:
                    self._log(f"Catch failed: {message.content}")
                self.state = BotState.IDLE

    async def _handle_spawn(self, message, embed):
        if self.state != BotState.IDLE:
            self._log("Spawn detected but bot is busy — skipping.", "warning")
            self.stats.total_skipped += 1
            return

        self.state = BotState.IDENTIFYING
        self._spawn_channel_id = message.channel.id
        self._log(f"Spawn detected in #{message.channel}")

        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        if random.random() < DISTRACTION_CHANCE:
            extra = random.uniform(*DISTRACTION_DELAY)
            self._log(f"Distraction delay +{extra:.1f}s")
            delay += extra
        self._log(f"Waiting {delay:.1f}s before identifying...")
        await asyncio.sleep(delay)

        image_bytes = await self._download_spawn_image(embed)
        if image_bytes and self.predictor.loaded:
            result = await self.predictor.predict_best(image_bytes, min_confidence=CNN_CONFIDENCE_THRESHOLD)
            if result:
                name, conf = result
                self._log(f"CNN prediction: {name} ({conf:.1%})")
                await self._attempt_catch(message.channel, name)
                self.stats.total_cnn_correct += 1
                return
            else:
                top = await self.predictor.predict(image_bytes, top_k=3)
                top_str = ", ".join(f"{n} ({c:.1%})" for n, c in top)
                self._log(f"CNN uncertain — top: {top_str}")

        self._log("Waiting for hint from Poketwo...")
        self.state = BotState.WAITING_FOR_HINT
        await asyncio.sleep(30)
        if self.state == BotState.WAITING_FOR_HINT:
            self._log("Hint timeout — returning to IDLE.", "warning")
            self.stats.total_skipped += 1
            self.state = BotState.IDLE

    async def _handle_hint(self, message):
        self._log(f"Hint received: {message.content}")
        best = get_best_hint_match(message.content)
        if best:
            self._log(f"Hint matched: {best}")
            await asyncio.sleep(random.uniform(1.0, 3.0))
            await self._attempt_catch(message.channel, best)
            self.stats.total_hint_used += 1
        else:
            self._log("No hint match found — skipping.", "warning")
            self.stats.total_skipped += 1
            self.state = BotState.IDLE

    async def _attempt_catch(self, channel, pokemon_name):
        self.state = BotState.WAITING_FOR_RESULT
        self._pending_pokemon = pokemon_name
        try:
            async with channel.typing():
                await asyncio.sleep(random.uniform(0.3, 1.2))
        except Exception:
            pass
        # Always ping Poketwo bot with @mention to catch
        catch_cmd = f"<@{POKETWO_BOT_ID}> catch {pokemon_name}"
        self._log(f"Sending: {catch_cmd}")
        await channel.send(catch_cmd)
        await asyncio.sleep(10)
        if self.state == BotState.WAITING_FOR_RESULT:
            self._log("No catch confirmation received.", "warning")
            self.state = BotState.IDLE

    async def _download_spawn_image(self, embed):
        url = None
        if embed.image and embed.image.url:
            url = embed.image.url
        elif embed.thumbnail and embed.thumbnail.url:
            url = embed.thumbnail.url
        if not url:
            self._log("No image URL in spawn embed.", "warning")
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        self._log(f"Downloaded spawn image ({len(data)} bytes)")
                        return data
                    self._log(f"Image download HTTP {resp.status}", "warning")
        except Exception as exc:
            self._log(f"Image download failed: {exc}", "warning")
        return None


_bot_instance = None

def get_bot():
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = PokeCatcherBot()
    return _bot_instance

async def run_bot(token):
    bot = get_bot()
    await bot.start(token)
