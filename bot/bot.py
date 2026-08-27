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
from pokemon_data import get_best_hint_match, resolve_authoritative_name, is_text_only_name

logger = logging.getLogger("bot")

POKETWO_BOT_ID = 716390085896962058

# ── P2 Assistant (built-in, always-on secondary hint source) ──────────────────
# The P2 Assistant bot ID is a fixed constant, NOT a configurable option. The
# integration is always active: it simply has no effect in servers where that
# bot is absent, because the bot only ever reacts to messages P2 Assistant
# actually sends. P2 Assistant emits two useful, unprompted messages:
#   1. "<Name>: <confidence>%"  — an automatic guess the instant a spawn appears
#      (often before our own CNN finishes). Used as an EARLY fast-path catch when
#      the confidence clears P2_ASSISTANT_AUTO_CONFIDENCE (see below).
#   2. "Possible Pokémon: <name>" — a post-hint guess, used as a fallback while
#      awaiting Poketwo's hint.
# In every case the name is resolved through the authoritative 936-label mapping
# and rejected when unknown/ambiguous, and P2 never overrides a Poketwo hint the
# bot has already matched.
P2_ASSISTANT_ID = 854233015475109888

# Minimum self-reported confidence for P2 Assistant's automatic "<Name>: <conf>%"
# guess to trigger an immediate catch, skipping the CNN wait and the hint window.
# Deliberately stricter than CNN_CONFIDENCE_THRESHOLD: we're trusting a
# third-party signal we can't verify against the spawn image ourselves. Hardcoded
# (not an env var) to keep this narrow shortcut simple and predictable — tweak it
# right here if the trade-off needs adjusting.
P2_ASSISTANT_AUTO_CONFIDENCE = 0.90

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

SPAWN_RE = re.compile(r"A\s+(?:new\s+)?wild\s+pok[eé]mon\s+has\s+appeared!", re.IGNORECASE)
HINT_RE = re.compile(r"The pok[eé]mon is \*\*.+\*\*", re.IGNORECASE)
FLEE_RE = re.compile(r"(?:The wild pok[eé]mon fled!|Wild\s+([^\n]+?)\s+fled\.?)", re.IGNORECASE)


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

    __slots__ = ("state", "pending_pokemon", "spawn_id")

    def __init__(self):
        self.state = BotState.IDLE
        self.pending_pokemon = None
        self.spawn_id = 0


class Stats:
    def __init__(self):
        self.total_caught = 0
        self.total_cnn_correct = 0
        self.total_hint_used = 0
        self.total_p2_assistant = 0
        self.total_skipped = 0
        self.total_fled = 0
        self.start_time = time.time()

    def to_dict(self):
        return {
            "total_caught": self.total_caught,
            "cnn_catches": self.total_cnn_correct,
            "hint_catches": self.total_hint_used,
            "p2_assistant_catches": self.total_p2_assistant,
            "skipped": self.total_skipped,
            "total_fled": self.total_fled,
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
            f"P2 Assistant active (built-in, ID {P2_ASSISTANT_ID}): early auto-guess "
            f"catches at >= {P2_ASSISTANT_AUTO_CONFIDENCE:.0%}, plus post-hint fallback; "
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

    async def _request_poketwo_hint(self, channel):
        """Ask Poketwo for a hint exactly once."""
        hint_cmd = f"<@{POKETWO_BOT_ID}> hint"
        self._log(f"{self._chan(channel)} Requesting hint from Poketwo: {hint_cmd}")
        try:
            await channel.send(hint_cmd)
        except discord.HTTPException as exc:
            self._log(f"{self._chan(channel)} Failed to request Poketwo hint: {exc}", "error")

    async def _handle_p2_assistant(self, message, session):
        """Act on P2 Assistant's two unprompted message formats, per channel.

        * ``<Name>: <confidence>%`` — posted automatically the instant a spawn
          appears (often before our CNN finishes). Treated as an *early* signal:
          from spawn detection onward (IDENTIFYING or WAITING_FOR_HINT), if the
          confidence is at/above ``P2_ASSISTANT_AUTO_CONFIDENCE`` and the name
          resolves to a single authoritative label, we catch immediately —
          skipping the CNN wait and the hint window. Low-confidence or unresolved
          auto-messages are ignored so the normal CNN+hint flow proceeds exactly
          as before.
        * ``Possible Pokémon: <name>`` — posted after a hint. Unchanged fallback:
          acted on only while WAITING_FOR_HINT.

        All state is read/written on the per-channel *session*, so a P2 message
        in one channel never affects another.
        """
        channel = message.channel

        # Only meaningful once a spawn has been detected in THIS channel; ignore
        # P2 chatter while idle or already resolving a catch.
        if session.state not in (BotState.IDENTIFYING, BotState.WAITING_FOR_HINT):
            self._log(
                f"{self._chan(channel)} Ignored P2 Assistant message — no spawn "
                "awaiting identification in this channel.",
                "debug",
            )
            return

        candidate = self._parse_p2_candidate(message.content or "")
        if candidate is None:
            self._log(f"{self._chan(channel)} Ignored unrecognized P2 Assistant candidate.", "warning")
            return

        pokemon_name, assistant_confidence = candidate

        # ── Post-hint "Possible Pokémon: <name>" (no confidence) — existing path ──
        if assistant_confidence is None:
            if session.state != BotState.WAITING_FOR_HINT:
                self._log(
                    f"{self._chan(channel)} Ignored P2 Assistant post-hint candidate "
                    "— this channel is not awaiting a hint.",
                    "debug",
                )
                return
            self._log(f"{self._chan(channel)} P2 Assistant post-hint candidate: {pokemon_name}")
            await self._attempt_catch(channel, pokemon_name, session, "P2 Assistant post-hint message")
            self.stats.total_p2_assistant += 1
            return

        # ── Early automatic "<Name>: <confidence>%" — fast path ──
        if assistant_confidence < P2_ASSISTANT_AUTO_CONFIDENCE:
            self._log(
                f"{self._chan(channel)} P2 Assistant auto-guess {pokemon_name} "
                f"({assistant_confidence:.1%}) below the {P2_ASSISTANT_AUTO_CONFIDENCE:.0%} "
                "bar — ignoring; continuing normal CNN/hint flow."
            )
            return

        self._log(
            f"{self._chan(channel)} P2 Assistant auto-guess {pokemon_name} "
            f"({assistant_confidence:.1%}) meets the {P2_ASSISTANT_AUTO_CONFIDENCE:.0%} "
            "bar — catching early (skipping CNN/hint wait)."
        )
        await self._attempt_catch(channel, pokemon_name, session, "P2 Assistant early auto-message")
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

        # Check if the message contains a flee notification
        is_flee = False
        if message.content and FLEE_RE.search(message.content):
            is_flee = True
        elif message.embeds:
            for embed in message.embeds:
                embed_title = getattr(embed, "title", None)
                embed_desc = getattr(embed, "description", None)
                if (embed_title and FLEE_RE.search(embed_title)) or (embed_desc and FLEE_RE.search(embed_desc)):
                    is_flee = True
                    break

        if is_flee:
            if session.state != BotState.IDLE:
                self._log(
                    f"{self._chan(message.channel)} Previous spawn fled — resetting state from {session.state.name} to IDLE."
                )
            else:
                self._log(f"{self._chan(message.channel)} Flee notice received — state already IDLE.", "debug")
            session.state = BotState.IDLE
            session.spawn_id += 1
            self.stats.total_fled += 1

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
        session.spawn_id += 1
        current_spawn_id = session.spawn_id
        self._log(f"{self._chan(channel)} Spawn detected")

        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        if random.random() < DISTRACTION_CHANCE:
            extra = random.uniform(*DISTRACTION_DELAY)
            self._log(f"{self._chan(channel)} Distraction delay +{extra:.1f}s")
            delay += extra
        self._log(f"{self._chan(channel)} Waiting {delay:.1f}s before identifying...")
        await asyncio.sleep(delay)

        # If state changed while sleeping (e.g. another handler reset it), abort
        if session.spawn_id != current_spawn_id or session.state != BotState.IDENTIFYING:
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

            # A high-confidence P2 Assistant auto-guess may have already claimed
            # this spawn while we were downloading/inferring — stand down rather
            # than double-catch.
            if session.spawn_id != current_spawn_id or session.state != BotState.IDENTIFYING:
                self._log(
                    f"{self._chan(channel)} Spawn already claimed or reset during identification — skipping CNN result.",
                    "debug",
                )
                return

            if result:
                name, conf = result
                self._log(f"{self._chan(channel)} CNN prediction: {name} ({conf:.1%})")
                await self._attempt_catch(channel, name, session, "CNN")
                self.stats.total_cnn_correct += 1
                return
            else:
                try:
                    top = await self.predictor.predict(image_bytes, top_k=3)
                    top_str = ", ".join(f"{n} ({c:.1%})" for n, c in top)
                    self._log(f"{self._chan(channel)} CNN uncertain — top: {top_str}")
                except Exception as exc:
                    self._log(f"{self._chan(channel)} CNN top-k error: {exc}", "error")

        # If a P2 Assistant auto-guess claimed this spawn during image download or
        # CNN inference, it is already being caught — don't clobber that state by
        # entering the hint wait.
        if session.spawn_id != current_spawn_id or session.state != BotState.IDENTIFYING:
            self._log(
                f"{self._chan(channel)} Spawn already claimed or reset during identification — not entering hint wait.",
                "debug",
            )
            return

        self._log(f"{self._chan(channel)} Waiting for hint from Poketwo...")
        session.state = BotState.WAITING_FOR_HINT
        await self._request_poketwo_hint(channel)

        # Wait for hint with timeout — the hint handler flips state before this fires
        await asyncio.sleep(30)
        if session.spawn_id == current_spawn_id and session.state == BotState.WAITING_FOR_HINT:
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
            await self._attempt_catch(channel, best, session, "Poketwo hint")
            self.stats.total_hint_used += 1
        else:
            self._log(f"{self._chan(channel)} No hint match found — skipping.", "warning")
            self.stats.total_skipped += 1
            session.state = BotState.IDLE

    async def _attempt_catch(self, channel, pokemon_name, session, source):
        session.state = BotState.WAITING_FOR_RESULT
        session.pending_pokemon = pokemon_name
        current_spawn_id = session.spawn_id
        if is_text_only_name(pokemon_name):
            self._log(f"{self._chan(channel)} [TEXT-ONLY CATCH] Catching {pokemon_name} (source: {source})")
        else:
            self._log(f"{self._chan(channel)} Catching {pokemon_name} (source: {source})")
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
            if session.spawn_id == current_spawn_id:
                session.state = BotState.IDLE
            return

        await asyncio.sleep(10)
        if session.spawn_id == current_spawn_id and session.state == BotState.WAITING_FOR_RESULT:
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
