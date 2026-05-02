# PokéCatcher — CNN-Powered Pokétwo Auto-Catcher

> ⚠️ **Disclaimer**: Discord selfbots violate Discord's Terms of Service. Use entirely at your own risk. Never share your user token.

A Discord selfbot that uses a pre-trained CNN model to automatically identify and catch Pokémon spawned by the [Pokétwo](https://poketwo.net/) bot.

---

## Architecture

```
poketwo-autocatcher/
├── model/                      # Pre-trained model files
│   ├── pokemon_cnn.keras       # Trained CNN model
│   ├── class_indices.json      # Class → index mapping
│   └── index_to_pokemon.json   # Index → Pokémon name mapping
├── bot/                        # Selfbot + dashboard
│   ├── main.py                 # Entry point (Flask + bot launcher)
│   ├── bot.py                  # Discord selfbot core logic
│   ├── predictor.py            # CNN model loader & inference
│   ├── pokemon_data.py         # 1200+ Pokémon names + hint parser
│   ├── web.py                  # Flask dashboard (dark UI)
│   └── requirements.txt        # Python dependencies
└── README.md
```

## How It Works

1. **Spawn Detection** — Monitors Pokétwo messages for spawn embeds
2. **Human-like Delay** — Waits 2–5 seconds (with 5% chance of extra "distraction" delay)
3. **CNN Prediction** — Downloads the spawn image and runs it through the trained model
4. **Confidence Check** — If confidence ≥ 70%, sends the catch command immediately
5. **Hint Fallback** — If CNN is uncertain, waits for Pokétwo's hint and pattern-matches against 1200+ names
6. **Catch** — Pings `@Poketwo` and sends `catch <name>` with typing simulation

## Setup

### 1. Install Dependencies

```bash
cd bot
pip install -r requirements.txt
```

### 2. Set Environment Variables

| Variable | Required | Description |
|---|---|---|
| `USER_TOKEN` | ✅ | Your Discord user token |
| `CATCH_CHANNEL_ID` | ✅ | Channel ID where the bot catches Pokémon |
| `PORT` | ❌ | Dashboard port (default: 5000) |
| `AUTOSTART` | ❌ | Set to `1` to auto-start the bot |

> 💡 **How to get a Channel ID:** Enable **Developer Mode** in Discord (Settings → Advanced → Developer Mode), then right-click any channel → **Copy Channel ID**.

**Windows:**
```cmd
set USER_TOKEN=your_token_here
set CATCH_CHANNEL_ID=123456789012345678
set AUTOSTART=1
```

**Linux/macOS:**
```bash
export USER_TOKEN=your_token_here
export CATCH_CHANNEL_ID=123456789012345678
export AUTOSTART=1
```

### 3. Run

```bash
cd bot
python main.py
```

Open `http://localhost:5000/dashboard` for the web dashboard.

## Dashboard Features

- **Start / Stop** the bot remotely
- **Live stats**: total caught, CNN catches, hint catches, skipped
- **Model status** indicator
- **Live logs** (auto-refreshes every 5s)
- **Dark theme** with glassmorphism design

## Stats Tracked

| Stat | Description |
|---|---|
| `total_caught` | Total Pokémon successfully caught |
| `cnn_catches` | Catches made via CNN prediction |
| `hint_catches` | Catches made via hint fallback |
| `skipped` | Spawns skipped (busy, timeout, no match) |
| `uptime` | Bot uptime since last start |

## Important Notes

- **Set `CATCH_CHANNEL_ID`** — the bot will ONLY catch in that specific channel
- The catch command always pings Pokétwo: `@Poketwo catch <name>` (bot ID: `716390085896962058`)
- The CNN model (`pokemon_cnn.keras`) must exist before running
- All delays are randomized to mimic human behavior
- CNN is the **primary** detection method; hints are the **fallback**
- The bot uses `discord.py-self` (no Intents required)
- Spawn images are downloaded and preprocessed to 128×128 before inference
