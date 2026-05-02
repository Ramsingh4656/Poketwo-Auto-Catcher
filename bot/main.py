try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

"""
main.py — Entry point: starts Flask dashboard and optionally auto-starts the bot.

Environment variables:
  USER_TOKEN   — Discord user token (required)
  PORT         — Dashboard port (default 5000)
  AUTOSTART    — Set to "1" or "true" to start the bot automatically
  CATCH_CHANNEL_ID — Channel ID where bot catches Pokemon (required)
"""

import os
import sys
import logging

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-12s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Validate token ─────────────────────────────────────────────────────────────
TOKEN = os.getenv("USER_TOKEN", "").strip()
if not TOKEN:
    logger.error("USER_TOKEN environment variable is not set.")
    logger.error("Set it via:  set USER_TOKEN=your_token_here  (Windows)")
    sys.exit(1)

PORT = int(os.getenv("PORT", "5000"))
AUTOSTART = os.getenv("AUTOSTART", "").lower() in ("1", "true", "yes")

# ── Import bot + web ──────────────────────────────────────────────────────────
from bot import get_bot
from web import app, set_bot_ref, _start_bot_thread

bot = get_bot()
set_bot_ref(bot, TOKEN)

# ── Auto-start ─────────────────────────────────────────────────────────────────
if AUTOSTART:
    logger.info("AUTOSTART enabled — launching bot…")
    _start_bot_thread()

# ── Run Flask ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Dashboard: http://127.0.0.1:%d/dashboard", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
