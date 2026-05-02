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

# ── Validate token ─────────────────────────────────────────────────────────────
TOKEN = os.getenv("USER_TOKEN", "").strip()
if not TOKEN:
    logger.error("═" * 60)
    logger.error("USER_TOKEN environment variable is not set!")
    logger.error("")
    logger.error("Fix: edit  bot/.env  and set your Discord user token:")
    logger.error("  USER_TOKEN=your_actual_token_here")
    logger.error("")
    logger.error("Or set it in your shell before running:")
    logger.error("  set USER_TOKEN=your_token_here   (Windows CMD)")
    logger.error("  $env:USER_TOKEN='your_token'     (PowerShell)")
    logger.error("═" * 60)
    sys.exit(1)

# Validate CATCH_CHANNEL_ID
_raw_channel = os.getenv("CATCH_CHANNEL_ID", "").strip()
if _raw_channel and not _raw_channel.isdigit():
    logger.error("CATCH_CHANNEL_ID must be a number, got: %r", _raw_channel)
    sys.exit(1)

PORT = int(os.getenv("PORT", "5000"))
AUTOSTART = os.getenv("AUTOSTART", "").lower() in ("1", "true", "yes")

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
