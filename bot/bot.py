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
from pokemon_data import get_best_hint_match, resolve_authoritative_name

logger = logging.getLogger("bot")

POKETWO_BOT_ID = 716390085896962058
_DEFAULT_P2_ASSISTANT_ID = 854233015475109888
_raw_p2_assistant_id = os.getenv("P2_ASSISTANT_ID", "").strip()
P2_ASSISTANT_ENABLED = False
if not _raw_p2_assistant_id:
    # Keep the documented default target available, but require explicit opt-in
    # so an unset optional integration remains completely inactive.
    P2_ASSISTANT_ID = _DEFAULT_P2_ASSISTANT_ID
    logger.info("P2 Assistant fallback disabled; P2_ASSISTANT_ID is not set.")
elif _raw_p2_assistant_id.isdigit() and int(_raw_p2_assistant_id) > 0:
    P2_ASSISTANT_ID = int(_raw_p2_assistant_id)
    P2_ASSISTANT_ENABLED = True
else:
    P2_ASSISTANT_ID = None
    logger.warning(
        "P2_ASSISTANT_ID is invalid; P2 Assistant feature disabled. "
        "Set it to a valid numeric user ID or leave it unset."
    )

_P2_SCORE_RE = re.compile(r"^\s*([^:\n]+?)\s*:\s*(\d+(?:\.\d+)?)%\s*$", re.IGNORECASE)
_P2_NAME_RE = re.compile(r"^\s*Possible Pokémon:\s*(.+?)\s*$", re.IGNORECASE)
P2_AUTO_GUESS_MIN_CONFIDENCE = 0.90

# ═══════════════════════════════════════════════════════════════════════════════
# SET YOUR CATCH CHANNEL ID HERE — the bot will ONLY catch in this channel.
# To find a channel ID: right-click the channel in Discord → Copy Channel ID
# Blank or invalid values are rejected at startup to prevent all-channel catching.
# You can also set via environment variable: CATCH_CHANNEL_ID=123456789
# ═══════════════════════════════════════════════════════════════════════════════
_raw_channel_id = os.getenv("CATCH_CHANNEL_ID", "").strip()
_CHANNEL_ERROR = (
    "CATCH_CHANNEL_ID is required and must be set to a valid numeric channel ID "
    "— refusing to start to avoid catching in all channels"
)
if not _raw_channel_id.isdigit() or int(_raw_channel_id) <= 0:
    logger.error(_CHANNEL_ERROR)
    raise SystemExit(1)
CATCH_CHANNEL_ID = int(_raw_channel_id)

MIN_DELAY, MAX_DELAY = 2.0, 5.0
DISTRACTION_CHANCE = 0.05
DISTRACTION_DELAY = (3.0, 8.0)
# Conservative interim gate: the published 0.30 metric was calibrated on the
# training/evaluation pipeline, not directly on the deployed ONNX session.
# Override with CNN_CONFIDENCE_THRESHOLD after Stage 6 calibration.
_raw_threshold = os.getenv("CNN_CONFIDENCE_THRESHOLD", "0.85").strip()
_THRESHOLD_ERROR = (
    "CNN_CONFIDENCE_THRESHOLD must be a number between 0 and 1, "
    f"got: {_raw_threshold!r}"
)
try:
    CNN_CONFIDENCE_THRESHOLD = float(_raw_threshold)
except ValueError:
    logger.error(_THRESHOLD_ERROR)
    raise SystemExit(1)
if not 0 <= CNN_CONFIDENCE_THRESHOLD <= 1:
    logger.error(_THRESHOLD_ERROR)
    raise SystemExit(1)

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
        self.total_p2_assistant = 0
        self.total_skipped = 0
        self.start_time = time.time()

    def to_dict(self):
        return {
            "total_caught": self.total_caught,
            "cnn_catches": self.total_cnn_correct,
            "hint_catches": self.total_hint_used,
            "p2_assistant_catches": self.total_p2_assistant,
            "skipped": self.total_skipped,
            "uptime_seconds": int(time.time() - self.start_time),
        }


class PokeCatcherBot(discord.Client):
    def __init__(self, **kwargs):
        # discord.py-self (selfbot fork) does NOT use Intents — that's a
        # bot-only feature.  Only set intents if using regular discord.py.
        if hasattr(discord, "Intents"):
            intents = discord.Intents.default()
            intents.message_content = True
            intents.messages = True
            intents.guilds = True
            kwargs["intents"] = intents
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

        # Reusable aiohttp session (avoid creating one per download)
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a reusable aiohttp session, creating one if needed."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._http_session

    def _log(self, msg, level="info"):
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        getattr(logger, level, logger.info)(msg)

    async def on_ready(self):
        self._log(f"Logged in as {self.user} (ID: {self.user.id})")
        self._log(f"Model loaded: {self.predictor.loaded}")
        if self.catch_channel_id:
            self._log(f"Catching ONLY in channel: {self.catch_channel_id}")
        if P2_ASSISTANT_ENABLED:
            self._log(f"P2 Assistant fallback enabled for user: {P2_ASSISTANT_ID}")
        else:
            self._log("P2 Assistant fallback disabled.")

    @staticmethod
    def _parse_p2_candidate(content):
        """Parse the first strict candidate line from P2 Assistant."""
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            score_match = _P2_SCORE_RE.fullmatch(line)
            if score_match:
                raw_name, score = score_match.groups()
                name = resolve_authoritative_name(raw_name)
                return (name, float(score) / 100.0) if name else None

            name_match = _P2_NAME_RE.fullmatch(line)
            if name_match:
                name = resolve_authoritative_name(name_match.group(1))
                return (name, None) if name else None
        return None

    async def _request_poketwo_hint(self, channel):
        """Ask Pokétwo for a hint; P2 Assistant remains a passive listener only."""
        hint_cmd = f"<@{POKETWO_BOT_ID}> hint"
        self._log(f"Requesting hint from Pokétwo: {hint_cmd}")
        try:
            await channel.send(hint_cmd)
        except discord.HTTPException as exc:
            self._log(f"Failed to request Pokétwo hint: {exc}", "error")

    async def _handle_p2_assistant(self, message):
        """Passively consume valid P2 guesses; never ask P2 for a hint."""
        if self.state not in (BotState.IDENTIFYING, BotState.WAITING_FOR_HINT):
            self._log("Ignored P2 Assistant candidate because no spawn is awaiting identification.", "debug")
            return

        candidate = self._parse_p2_candidate(message.content or "")
        if candidate is None:
            self._log("Ignored unrecognized P2 Assistant candidate.", "warning")
            return

        pokemon_name, assistant_confidence = candidate
        is_post_hint_candidate = assistant_confidence is None
        is_high_confidence_auto_guess = (
            assistant_confidence is not None
            and assistant_confidence >= P2_AUTO_GUESS_MIN_CONFIDENCE
        )
        if self.state == BotState.IDENTIFYING and not is_high_confidence_auto_guess:
            self._log(
                "Ignored P2 Assistant auto-guess below 90% confidence; continuing CNN identification.",
                "debug",
            )
            return
        if self.state == BotState.WAITING_FOR_HINT and not (
            is_post_hint_candidate or is_high_confidence_auto_guess
        ):
            self._log("Ignored P2 Assistant auto-guess below 90% confidence.", "debug")
            return

        suffix = f" ({assistant_confidence:.1%})" if assistant_confidence is not None else " (post-hint)"
        self._log(f"P2 Assistant candidate: {pokemon_name}{suffix}")
        await self._attempt_catch(message.channel, pokemon_name)
        self.stats.total_p2_assistant += 1

    async def on_message(self, message):
        # Only process messages in the designated catch channel.
        if self.catch_channel_id and message.channel.id != self.catch_channel_id:
            return

        # P2 Assistant is an additive fallback signal; it never replaces Poketwo.
        if P2_ASSISTANT_ENABLED and P2_ASSISTANT_ID is not None and message.author.id == P2_ASSISTANT_ID:
            await self._handle_p2_assistant(message)
            return

        # Only process the remaining messages from Poketwo.
        if message.author.id != POKETWO_BOT_ID:
            return

        # Check for spawn embed
        if message.embeds:
            for embed in message.embeds:
                if embed.title and SPAWN_RE.search(embed.title):
                    await self._handle_spawn(message, embed)
                    return

        # Check for hint message
        if self.state == BotState.WAITING_FOR_HINT:
            if message.content and HINT_RE.search(message.content):
                await self._handle_hint(message)
                return

        # Check for catch confirmation
        if self.state == BotState.WAITING_FOR_RESULT:
            if message.content and self.user:
                # Poketwo mentions user by ID or @mention in catch results
                user_id_str = str(self.user.id)
                user_mention = f"<@{self.user.id}>"
                if user_id_str in message.content or user_mention in message.content:
                    if "caught" in message.content.lower():
                        self.stats.total_caught += 1
                        self._log(f"✓ Caught! {message.content}")
                    else:
                        self._log(f"✗ Catch failed: {message.content}")
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

        # If state changed while sleeping (e.g. another handler reset it), abort
        if self.state != BotState.IDENTIFYING:
            return

        image_bytes = await self._download_spawn_image(embed)
        if image_bytes and self.predictor.loaded:
            try:
                result = await self.predictor.predict_best(
                    image_bytes, min_confidence=CNN_CONFIDENCE_THRESHOLD
                )
            except Exception as exc:
                self._log(f"CNN prediction error: {exc}", "error")
                result = None

            if result:
                name, conf = result
                self._log(f"CNN prediction: {name} ({conf:.1%})")
                await self._attempt_catch(message.channel, name)
                self.stats.total_cnn_correct += 1
                return
            else:
                try:
                    top = await self.predictor.predict(image_bytes, top_k=3)
                    top_str = ", ".join(f"{n} ({c:.1%})" for n, c in top)
                    self._log(f"CNN uncertain — top: {top_str}")
                except Exception as exc:
                    self._log(f"CNN top-k error: {exc}", "error")

        self._log("Waiting for hint from Poketwo...")
        self.state = BotState.WAITING_FOR_HINT
        await self._request_poketwo_hint(message.channel)

        # Wait for hint with timeout — use a task so hint handler can cancel it
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
        try:
            await channel.send(catch_cmd)
        except discord.HTTPException as exc:
            self._log(f"Failed to send catch command: {exc}", "error")
            self.state = BotState.IDLE
            return

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
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    self._log(f"Downloaded spawn image ({len(data)} bytes)")
                    return data
                self._log(f"Image download HTTP {resp.status}", "warning")
        except Exception as exc:
            self._log(f"Image download failed: {exc}", "warning")
        return None

    async def close(self):
        """Clean up resources before closing."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        await super().close()


_bot_instance = None

def get_bot():
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = PokeCatcherBot()
    return _bot_instance

async def run_bot(token):
    bot = get_bot()
    await bot.start(token)
