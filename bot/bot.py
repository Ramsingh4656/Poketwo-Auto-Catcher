"""
bot.py — Discord selfbot that detects Poketwo spawns and catches Pokemon.
Uses CNN model as primary identifier; falls back to hint-matching.
WARNING: Selfbots violate Discord ToS — use at your own risk.
"""

from __future__ import annotations

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

# ── P2 Assistant (built-in, always-on secondary hint signal) ──────────────────
# The P2 Assistant bot ID is a fixed constant, NOT a configurable option. The
# integration is always active: it simply has no effect in servers where that
# bot is absent, because the bot only ever reacts to messages P2 Assistant
# actually sends. Its candidate names are still resolved through the
# authoritative 936-label mapping and rejected when unknown/ambiguous, and it
# never overrides Poketwo — it only fills in while the bot is awaiting a hint.
P2_ASSISTANT_ID = 854233015475109888

_P2_SCORE_RE = re.compile(r"^\s*([^:\n]+?)\s*:\s*(\d+(?:\.\d+)?)%\s*$", re.IGNORECASE)
_P2_NAME_RE = re.compile(r"^\s*Possible Pokémon:\s*(.+?)\s*$", re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════════
# SET YOUR CATCH CHANNEL ID(S) HERE — the bot will ONLY catch in these channels.
# To find a channel ID: right-click the channel in Discord → Copy Channel ID
# Accepts a single ID or a comma-separated list, e.g.
#   CATCH_CHANNEL_ID=123456789012345678
#   CATCH_CHANNEL_ID=123456789012345678,987654321098765432
# Blank or invalid values are rejected at startup to prevent all-channel catching.
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_catch_channel_ids(raw: str) -> list[int]:
    """Parse CATCH_CHANNEL_ID into a de-duplicated list of positive channel IDs.

    Accepts a single ID or a comma-separated list. Fails closed by raising
    ValueError — with a message naming the offending entries — when the value is
    missing/blank or any entry is not a positive integer, so the bot can never
    silently fall back to catching in every channel.
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError(
            "CATCH_CHANNEL_ID is required and must be one or more numeric channel "
            "IDs (comma-separated) — refusing to start to avoid catching in all channels"
        )
    ids: list[int] = []
    invalid: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if token.isdigit() and int(token) > 0:
            value = int(token)
            if value not in ids:
                ids.append(value)
        else:
            invalid.append(part)
    if invalid:
        raise ValueError(
            "CATCH_CHANNEL_ID contains invalid channel ID(s): "
            + ", ".join(repr(p) for p in invalid)
            + " — every entry must be a positive numeric channel ID"
        )
    return ids


try:
    CATCH_CHANNEL_IDS = _parse_catch_channel_ids(os.getenv("CATCH_CHANNEL_ID", ""))
except ValueError as exc:
    logger.error("%s", exc)
    raise SystemExit(1)

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


class ChannelSession:
    """Per-channel catch state.

    Each configured channel gets its own state machine and pending target so
    concurrent spawns in different channels never interfere with one another.
    """

    __slots__ = ("state", "pending_pokemon")

    def __init__(self):
        self.state = BotState.IDLE
        self.pending_pokemon = None


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
        self.stats = Stats()
        self.logs = deque(maxlen=200)

        # Channel restriction — one or more channels, each with its own session
        # so concurrent spawns in different channels don't clobber each other.
        self.catch_channel_ids = set(CATCH_CHANNEL_IDS)
        self._sessions = {cid: ChannelSession() for cid in self.catch_channel_ids}

        # Reusable aiohttp session (avoid creating one per download)
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a reusable aiohttp session, creating one if needed."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._http_session

    def _session(self, channel_id) -> ChannelSession:
        """Return the per-channel session, creating one on demand as a safety net."""
        session = self._sessions.get(channel_id)
        if session is None:
            session = ChannelSession()
            self._sessions[channel_id] = session
        return session

    @staticmethod
    def _chan(channel) -> str:
        """Human-readable channel label for logs, e.g. '[#general:123]'."""
        name = getattr(channel, "name", None)
        return f"[#{name}:{channel.id}]" if name else f"[channel {channel.id}]"

    def channel_state_names(self) -> dict:
        """Map channel ID (str) → current state name, for the dashboard."""
        return {str(cid): sess.state.name for cid, sess in self._sessions.items()}

    def _log(self, msg, level="info"):
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        getattr(logger, level, logger.info)(msg)

    async def on_ready(self):
        self._log(f"Logged in as {self.user} (ID: {self.user.id})")
        self._log(f"Model loaded: {self.predictor.loaded}")
        channels = ", ".join(str(c) for c in sorted(self.catch_channel_ids))
        self._log(f"Catching ONLY in channel(s): {channels}")
        self._log(
            f"P2 Assistant fallback active (built-in, ID {P2_ASSISTANT_ID}); "
            "no effect in servers where that bot is absent."
        )

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

    async def _request_p2_hint(self, channel):
        """Ask the built-in P2 Assistant for a hint exactly once."""
        hint_cmd = f"<@{P2_ASSISTANT_ID}> hint"
        self._log(f"{self._chan(channel)} Requesting hint from P2 Assistant: {hint_cmd}")
        try:
            await channel.send(hint_cmd)
        except discord.HTTPException as exc:
            self._log(f"{self._chan(channel)} Failed to request P2 Assistant hint: {exc}", "error")

    async def _handle_p2_assistant(self, message, session):
        """Use only a strict, authoritative P2 candidate while awaiting a hint."""
        channel = message.channel
        if session.state != BotState.WAITING_FOR_HINT:
            self._log(
                f"{self._chan(channel)} Ignored P2 Assistant candidate because this "
                "channel is not awaiting a hint.",
                "debug",
            )
            return

        candidate = self._parse_p2_candidate(message.content or "")
        if candidate is None:
            self._log(f"{self._chan(channel)} Ignored unrecognized P2 Assistant candidate.", "warning")
            return

        pokemon_name, assistant_confidence = candidate
        suffix = f" ({assistant_confidence:.1%})" if assistant_confidence is not None else ""
        self._log(f"{self._chan(channel)} P2 Assistant candidate: {pokemon_name}{suffix}")
        await self._attempt_catch(channel, pokemon_name, session)
        self.stats.total_p2_assistant += 1

    async def on_message(self, message):
        # Only process messages in a configured catch channel.
        channel_id = message.channel.id
        if channel_id not in self.catch_channel_ids:
            return
        session = self._session(channel_id)

        # P2 Assistant is an additive fallback signal; it never replaces Poketwo.
        if message.author.id == P2_ASSISTANT_ID:
            await self._handle_p2_assistant(message, session)
            return

        # Only process the remaining messages from Poketwo.
        if message.author.id != POKETWO_BOT_ID:
            return

        # Check for spawn embed
        if message.embeds:
            for embed in message.embeds:
                if embed.title and SPAWN_RE.search(embed.title):
                    await self._handle_spawn(message, embed, session)
                    return

        # Check for hint message
        if session.state == BotState.WAITING_FOR_HINT:
            if message.content and HINT_RE.search(message.content):
                await self._handle_hint(message, session)
                return

        # Check for catch confirmation
        if session.state == BotState.WAITING_FOR_RESULT:
            if message.content and self.user:
                # Poketwo mentions user by ID or @mention in catch results
                user_id_str = str(self.user.id)
                user_mention = f"<@{self.user.id}>"
                if user_id_str in message.content or user_mention in message.content:
                    if "caught" in message.content.lower():
                        self.stats.total_caught += 1
                        self._log(f"{self._chan(message.channel)} ✓ Caught! {message.content}")
                    else:
                        self._log(f"{self._chan(message.channel)} ✗ Catch failed: {message.content}")
                    session.state = BotState.IDLE

    async def _handle_spawn(self, message, embed, session):
        channel = message.channel
        if session.state != BotState.IDLE:
            self._log(f"{self._chan(channel)} Spawn detected but this channel is busy — skipping.", "warning")
            self.stats.total_skipped += 1
            return

        session.state = BotState.IDENTIFYING
        self._log(f"{self._chan(channel)} Spawn detected")

        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        if random.random() < DISTRACTION_CHANCE:
            extra = random.uniform(*DISTRACTION_DELAY)
            self._log(f"{self._chan(channel)} Distraction delay +{extra:.1f}s")
            delay += extra
        self._log(f"{self._chan(channel)} Waiting {delay:.1f}s before identifying...")
        await asyncio.sleep(delay)

        # If state changed while sleeping (e.g. another handler reset it), abort
        if session.state != BotState.IDENTIFYING:
            return

        image_bytes = await self._download_spawn_image(embed, channel)
        if image_bytes and self.predictor.loaded:
            try:
                result = await self.predictor.predict_best(
                    image_bytes, min_confidence=CNN_CONFIDENCE_THRESHOLD
                )
            except Exception as exc:
                self._log(f"{self._chan(channel)} CNN prediction error: {exc}", "error")
                result = None

            if result:
                name, conf = result
                self._log(f"{self._chan(channel)} CNN prediction: {name} ({conf:.1%})")
                await self._attempt_catch(channel, name, session)
                self.stats.total_cnn_correct += 1
                return
            else:
                try:
                    top = await self.predictor.predict(image_bytes, top_k=3)
                    top_str = ", ".join(f"{n} ({c:.1%})" for n, c in top)
                    self._log(f"{self._chan(channel)} CNN uncertain — top: {top_str}")
                except Exception as exc:
                    self._log(f"{self._chan(channel)} CNN top-k error: {exc}", "error")

        self._log(f"{self._chan(channel)} Waiting for hint from Poketwo...")
        session.state = BotState.WAITING_FOR_HINT
        await self._request_p2_hint(channel)

        # Wait for hint with timeout — the hint handler flips state before this fires
        await asyncio.sleep(30)
        if session.state == BotState.WAITING_FOR_HINT:
            self._log(f"{self._chan(channel)} Hint timeout — returning to IDLE.", "warning")
            self.stats.total_skipped += 1
            session.state = BotState.IDLE

    async def _handle_hint(self, message, session):
        channel = message.channel
        self._log(f"{self._chan(channel)} Hint received: {message.content}")
        best = get_best_hint_match(message.content)
        if best:
            self._log(f"{self._chan(channel)} Hint matched: {best}")
            await asyncio.sleep(random.uniform(1.0, 3.0))
            await self._attempt_catch(channel, best, session)
            self.stats.total_hint_used += 1
        else:
            self._log(f"{self._chan(channel)} No hint match found — skipping.", "warning")
            self.stats.total_skipped += 1
            session.state = BotState.IDLE

    async def _attempt_catch(self, channel, pokemon_name, session):
        session.state = BotState.WAITING_FOR_RESULT
        session.pending_pokemon = pokemon_name
        try:
            async with channel.typing():
                await asyncio.sleep(random.uniform(0.3, 1.2))
        except Exception:
            pass
        # Always ping Poketwo bot with @mention to catch
        catch_cmd = f"<@{POKETWO_BOT_ID}> catch {pokemon_name}"
        self._log(f"{self._chan(channel)} Sending: {catch_cmd}")
        try:
            await channel.send(catch_cmd)
        except discord.HTTPException as exc:
            self._log(f"{self._chan(channel)} Failed to send catch command: {exc}", "error")
            session.state = BotState.IDLE
            return

        await asyncio.sleep(10)
        if session.state == BotState.WAITING_FOR_RESULT:
            self._log(f"{self._chan(channel)} No catch confirmation received.", "warning")
            session.state = BotState.IDLE

    async def _download_spawn_image(self, embed, channel):
        url = None
        if embed.image and embed.image.url:
            url = embed.image.url
        elif embed.thumbnail and embed.thumbnail.url:
            url = embed.thumbnail.url
        if not url:
            self._log(f"{self._chan(channel)} No image URL in spawn embed.", "warning")
            return None
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    self._log(f"{self._chan(channel)} Downloaded spawn image ({len(data)} bytes)")
                    return data
                self._log(f"{self._chan(channel)} Image download HTTP {resp.status}", "warning")
        except Exception as exc:
            self._log(f"{self._chan(channel)} Image download failed: {exc}", "warning")
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
