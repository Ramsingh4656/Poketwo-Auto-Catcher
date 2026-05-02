<div align="center">

<img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png" width="120px" />

# PokéCatcher
### CNN-Powered Pokétwo Auto-Catcher

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![discord.py-self](https://img.shields.io/badge/discord.py--self-2.1.0-5865f2?style=for-the-badge&logo=discord&logoColor=white)](https://github.com/dolfies/discord.py-self)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

A Discord selfbot that uses a **custom trained CNN model** to automatically identify and catch Pokémon spawned by the [Pokétwo](https://poketwo.net/) bot — with human-like delays and a live web dashboard.

[Features](#-features) · [How It Works](#-how-it-works) · [Setup](#-setup) · [Dashboard](#-dashboard) · [Training](#-model-training) · [FAQ](#-faq)

</div>

---

> ⚠️ **Disclaimer**
> Discord selfbots violate [Discord's Terms of Service](https://discord.com/terms). Using this tool may result in your account being banned. This project is for **educational purposes only** — use entirely at your own risk. **Never share your user token with anyone.**

---

## ✨ Features

- 🧠 **Custom CNN Model** — trained specifically on Pokétwo's CDN images, not a generic vision API
- 🎯 **95%+ Accuracy** — identifies all 1025+ Pokémon including regional and alternate forms
- 💡 **Smart Fallback** — if CNN confidence < 70%, automatically uses Pokétwo's hint system
- 🧍 **Human-like Behaviour** — randomized delays (2–5s), typing indicators, occasional "distraction" pauses
- 🌐 **Live Dashboard** — start/stop the bot, view stats, and read logs from your browser
- 📺 **Single Channel Only** — strictly watches one channel, ignores everything else
- ⚡ **Fast Inference** — ~10ms per image on CPU, ~2ms on GPU
- 🔄 **Zero API Cost** — model runs fully offline after training, no external API calls

---

## 🔧 How It Works

```
Pokétwo Spawn Message
        ↓
  Download Image
        ↓
  CNN Model Inference
        ↓
  Confidence ≥ 70%? ──── YES ──→ Send catch command ✅
        │
       NO
        ↓
  Request /hint from Pokétwo
        ↓
  Pattern match hint against 1200+ Pokémon names
        ↓
  Confidence ≥ 85%? ──── YES ──→ Send catch command ✅
        │
       NO
        ↓
  Skip spawn (too risky) ⏭️
```

**Human-like behaviour at every step:**
- 2.5–5.0 second delay before reacting to spawn
- 5% chance of an extra 3–8 second "distraction" pause
- Typing indicator shown before every message
- Typing duration scales with message length

---

## 📁 Project Structure

```
poketwo-autocatcher/
├── model/                       # Pre-trained model files (you provide these)
│   ├── pokemon_cnn.keras        # Trained CNN model
│   ├── class_indices.json       # Pokémon name → index
│   └── index_to_pokemon.json    # Index → Pokémon name
│
├── bot/                         # Selfbot + dashboard
│   ├── main.py                  # Entry point
│   ├── bot.py                   # Discord selfbot core logic
│   ├── predictor.py             # CNN model loader & inference
│   ├── pokemon_data.py          # 1200+ Pokémon names + hint parser
│   ├── web.py                   # Flask dashboard
│   └── requirements.txt
│
├── train/                       # CNN training pipeline
│   ├── download_dataset.py      # Download all Pokétwo CDN images
│   ├── augment.py               # Data augmentation
│   ├── train_model.py           # Build & train CNN
│   ├── test_model.py            # Test model with any image
│   └── requirements_train.txt
│
└── README.md
```

---

## 🚀 Setup

### Prerequisites

- Python 3.10 or higher
- A trained CNN model (see [Model Training](#-model-training) below)
- Your Discord user token
- The channel ID where Pokétwo spawns happen

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Ramsingh4656/Poketwo-Auto-Catcher.git
cd Poketwo-Auto-Catcher
```

---

### Step 2 — Install Bot Dependencies

```bash
cd bot
pip install -r requirements.txt
```

---

### Step 3 — Get Your Discord User Token

> ⚠️ Your user token gives full access to your Discord account. Never share it.

1. Open **Discord in your browser** (not the desktop app)
2. Press `F12` to open DevTools
3. Go to the **Network** tab
4. Press `Ctrl+R` to reload the page
5. Click any request going to `discord.com/api`
6. In the **Request Headers** section, find `authorization`
7. Copy that value — that is your `USER_TOKEN`

---

### Step 4 — Get Your Channel ID

1. Open Discord → **User Settings** → **Advanced**
2. Enable **Developer Mode**
3. Right-click the channel where Pokétwo spawns happen
4. Click **Copy Channel ID**

---

### Step 5 — Set Environment Variables

**Windows (Command Prompt):**
```cmd
set USER_TOKEN=your_token_here
set CATCH_CHANNEL_ID=123456789012345678
set AUTOSTART=1
```

**Windows (PowerShell):**
```powershell
$env:USER_TOKEN="your_token_here"
$env:CATCH_CHANNEL_ID="123456789012345678"
$env:AUTOSTART="1"
```

**Linux / macOS:**
```bash
export USER_TOKEN=your_token_here
export CATCH_CHANNEL_ID=123456789012345678
export AUTOSTART=1
```

| Variable | Required | Description |
|---|---|---|
| `USER_TOKEN` | ✅ | Your Discord user token |
| `CATCH_CHANNEL_ID` | ✅ | Channel ID to watch for spawns |
| `PORT` | ❌ | Dashboard port (default: `5000`) |
| `AUTOSTART` | ❌ | Set to `1` to auto-start the bot on launch |

---

### Step 7 — Run

```bash
cd bot
python main.py
```

Open your browser and go to:
```
http://localhost:5000/dashboard
```

---

## 🖥️ Dashboard

The web dashboard lets you control everything from your browser.

| Feature | Description |
|---|---|
| **Start / Stop** | Start or stop the bot with one click |
| **Model Status** | Shows whether CNN model is loaded |
| **Live Stats** | Total caught, CNN catches, hint catches, skipped |
| **Uptime** | How long the bot has been running |
| **Live Logs** | Auto-refreshes every 5 seconds, colour-coded by level |

**Log colours:**
- ⬜ Grey — INFO (normal events)
- 🟡 Yellow — WARN (non-critical issues)
- 🔴 Red — ERROR (failures)
- 🟢 Green — successful catches

---

## 📊 Stats Explained

| Stat | Description |
|---|---|
| `total_caught` | Total Pokémon caught this session |
| `cnn_catches` | Catches made using CNN prediction |
| `hint_catches` | Catches made using hint fallback |
| `skipped` | Spawns skipped (low confidence or no match) |
| `uptime` | Time since bot was last started |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| `discord.py-self` | Discord user account API (no Intents needed) |
| `TensorFlow / Keras` | CNN model training and inference |
| `Flask` | Web dashboard backend |
| `aiohttp` | Async image downloading |
| `Pillow` | Image preprocessing |
| `NumPy` | Array operations for inference |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repo
2. Create your branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

<div align="center">

Made with ❤️ by [Ram Singh](https://github.com/Ramsingh4656)

⭐ Star this repo if it helped you!

</div>
