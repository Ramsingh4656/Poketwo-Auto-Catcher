"""
main.py — Entry point: starts Flask dashboard and optionally auto-starts the bot.

Environment variables:
  USER_TOKEN       — Discord user token (required)
  PORT             — Dashboard port (default 5000)
  AUTOSTART        — Set to "1" or "true" to start the bot automatically
  CATCH_CHANNEL_ID — Channel ID where bot catches Pokemon (required)
"""

# ── Load .env FIRST — before any os.getenv() call ─────────────────────────────
import os
import sys

try:
    from dotenv import load_dotenv
    # Explicitly resolve .env path relative to THIS file, not CWD.
    # This ensures it works no matter where the user runs `python main.py` from.
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    _loaded = load_dotenv(_env_path)
except ImportError:
    _loaded = False
    _env_path = None

import logging

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-12s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# Log .env loading result so user can diagnose issues
if _env_path:
    if _loaded:
        logger.info("Loaded .env from: %s", _env_path)
    else:
        logger.warning(".env file not found at: %s", _env_path)
else:
    logger.warning("python-dotenv not installed — reading env vars from system only.")

# ── Validate & clean token ─────────────────────────────────────────────────────
_raw_token = os.getenv("USER_TOKEN", "")
# Aggressively clean token: strip whitespace, BOM, quotes, carriage returns
TOKEN = _raw_token.strip().strip("\ufeff").strip('"').strip("'").strip()
# Remove any remaining non-printable / invisible characters
TOKEN = "".join(ch for ch in TOKEN if ch.isprintable() and ch != " ")

# Diagnostic logging
if _raw_token != TOKEN:
    logger.warning("Token was cleaned — original had extra characters!")
    logger.warning("  Raw length: %d, Clean length: %d", len(_raw_token), len(TOKEN))
    # Show which characters were removed
    removed = set(_raw_token) - set(TOKEN) - {" "}
    if removed:
        logger.warning("  Removed chars: %s", [hex(ord(c)) for c in removed])

if not TOKEN:
    logger.error("═" * 60)
    logger.error("USER_TOKEN environment variable is not set!")
    logger.error("")
    logger.error("Fix: edit  bot/.env  and set your Discord user token:")
    logger.error("  USER_TOKEN=your_actual_token_here")
    logger.error("")
    logger.error("  IMPORTANT: Do NOT wrap the token in quotes!")
    logger.error("  WRONG:  USER_TOKEN=\"mytoken123\"")
    logger.error("  RIGHT:  USER_TOKEN=mytoken123")
    logger.error("")
    logger.error("Or set it in your shell before running:")
    logger.error("  set USER_TOKEN=your_token_here   (Windows CMD)")
    logger.error("  $env:USER_TOKEN='your_token'     (PowerShell)")
    logger.error("═" * 60)
    sys.exit(1)

# Basic structure check: Discord tokens have 3 parts separated by dots
_token_parts = TOKEN.split(".")
if len(_token_parts) != 3:
    logger.error("═" * 60)
    logger.error("TOKEN FORMAT ERROR: Discord tokens have 3 parts separated by dots.")
    logger.error("  Expected format:  XXXXXX.YYYYYY.ZZZZZZ")
    logger.error("  Your token has %d part(s).", len(_token_parts))
    logger.error("  Token length: %d chars", len(TOKEN))
    logger.error("  First 10 chars: %s...", TOKEN[:10])
    logger.error("")
    logger.error("  Make sure you copied the FULL token, not just part of it.")
    logger.error("═" * 60)
    sys.exit(1)

logger.info("Token format OK: 3 parts, %d total chars", len(TOKEN))

# Validate CATCH_CHANNEL_ID
_raw_channel = os.getenv("CATCH_CHANNEL_ID", "").strip()
if _raw_channel and not _raw_channel.isdigit():
    logger.error("CATCH_CHANNEL_ID must be a number, got: %r", _raw_channel)
    sys.exit(1)

PORT = int(os.getenv("PORT", "5000"))
AUTOSTART = os.getenv("AUTOSTART", "").lower() in ("1", "true", "yes")
# ── Verify correct discord library ─────────────────────────────────────────────
# discord.py-self (selfbot fork) sends tokens as-is.
# Regular discord.py sends tokens as "Bot <token>" which causes 401 for user tokens.
# If both are installed, the wrong one may load first.
import discord
import inspect

_discord_file = discord.__file__
_is_selfbot_fork = False
try:
    _http_src = inspect.getsource(discord.http.HTTPClient.request)
    # The selfbot fork does NOT add "Bot " prefix
    _is_selfbot_fork = "'Bot ' + self.token" not in _http_src
except Exception:
    _is_selfbot_fork = not hasattr(discord, "Intents")  # fallback check

if not _is_selfbot_fork:
    logger.error("=" * 60)
    logger.error("WRONG DISCORD LIBRARY DETECTED!")
    logger.error("")
    logger.error("  Loaded: discord.py %s", discord.__version__)
    logger.error("  From:   %s", _discord_file)
    logger.error("")
    logger.error("  This is regular discord.py (for bots), NOT discord.py-self")
    logger.error("  (for selfbots). It adds 'Bot ' prefix to your token,")
    logger.error("  causing 'Improper token' errors.")
    logger.error("")
    logger.error("  FIX: Run this command and restart:")
    logger.error("    pip uninstall discord.py -y")
    logger.error("=" * 60)

    # Try to auto-fix
    logger.info("Attempting automatic fix...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "discord.py", "-y"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info("Uninstalled regular discord.py. Restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    else:
        logger.error("Auto-fix failed. Please run manually:")
        logger.error("  pip uninstall discord.py -y")
        sys.exit(1)
else:
    logger.info("Discord library: discord.py-self %s (OK)", discord.__version__)

# ── Import bot + web ──────────────────────────────────────────────────────────
from bot import get_bot
from web import app, set_bot_ref, _start_bot_thread

bot = get_bot()
set_bot_ref(bot, TOKEN)

logger.info("Bot user token: ...%s (last 4 chars)", TOKEN[-4:] if len(TOKEN) >= 4 else "????")
logger.info("Catch channel: %s", bot.catch_channel_id or "ALL CHANNELS (not recommended)")
logger.info("Model loaded: %s", bot.predictor.loaded)

# ── Auto-start ─────────────────────────────────────────────────────────────────
if AUTOSTART:
    logger.info("AUTOSTART enabled — launching bot…")
    _start_bot_thread()

# ── Run Flask ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Dashboard: http://127.0.0.1:%d/dashboard", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
