<div align="center">

<img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png" width="120px" />

# PokéCatcher
### CNN-Powered Pokétwo Auto-Catcher

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX--Runtime-1.18+-005c99?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![discord.py-self](https://img.shields.io/badge/discord.py--self-latest-5865f2?style=for-the-badge&logo=discord&logoColor=white)](https://github.com/dolfies/discord.py-self)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

A Discord selfbot that uses an optimized **MobileNetV3-small ONNX model** to automatically identify and catch Pokémon spawned by the [Pokétwo](https://poketwo.net/) bot — with human-like delays, a smart text-hint fallback system, and a live web dashboard.

[Features](#-features) · [How It Works](#-how-it-works) · [Project Structure](#-project-structure) · [Setup](#-setup) · [AI Model](#-ai-model) · [Performance](#-accuracyperformance) · [Dashboard](#-dashboard) · [Tech Stack](#-tech-stack) · [Troubleshooting](#-troubleshooting) · [Contributing](#-contributing)

</div>

---

> ⚠️ **Disclaimer**
> Discord selfbots violate [Discord's Terms of Service](https://discord.com/terms). Using this tool may result in your account being banned. This project is for **educational purposes only** — use entirely at your own risk. **Never share your user token with anyone.**

---

## ✨ Features

- 🧠 **Optimized ONNX Model** — runs a 936-class MobileNetV3-small vision model locally on CPU with zero external API call overhead.
- 🎯 **High Accuracy & Fallback** — delivers over 98% accepted top-1 accuracy on high-confidence visual predictions.
- 💡 **Smart Fallback** — if the AI prediction confidence is below the threshold (`0.30`), the bot abstains from guessing and automatically parses Pokétwo's text hints.
- 🧍 **Human-like Behaviour** — randomized response delays (2–5s), typing indicators, and occasional 3–8s "distraction" pauses.
- 🌐 **Live Dashboard** — start/stop the selfbot, view catch metrics, and read live execution logs directly from your browser.
- 📺 **Single Channel Only** — strictly watches the designated `CATCH_CHANNEL_ID` channel and ignores all other noise.
- ⚡ **Ultra-Fast Inference** — processing and classification complete in ~8.76 ms per image on standard CPUs.

---

## 🔧 How It Works

```
Pokétwo Spawn Message
        ↓
  Download Image
        ↓
  Image Preprocessing (224x224, NCHW Normalized)
        ↓
  ONNX Model Inference (MobileNetV3-small CPU)
        ↓
  Confidence ≥ 0.30? ──── YES ──→ Send catch command ✅
        │
       NO
        ↓
  Wait for Hint from Pokétwo
        ↓
  Pattern Match Hint Against 936 Pokémon Names
        ↓
  Match Found? ────────── YES ──→ Send catch command ✅
        │
       NO
        ↓
  Skip Spawn (avoid risk) ⏭️
```

**Human-like behavior at every step:**
- A random delay of 2–5 seconds is applied before downloading the image or identifying.
- A 5% chance of an extra 3–8 second distraction delay is simulated to mock user distraction.
- Typing indicators are triggered inside the Discord channel before sending the catch message.

---

## 📁 Project Structure

```text
Poketwo-Auto-Catcher/
├── bot/
│   ├── bot.py                   # Main selfbot listener & handler
│   ├── main.py                  # Web dashboard launcher & entry point
│   ├── pokemon_data.py          # Hint regex matcher & name dictionary
│   ├── predictor.py             # ONNX Runtime model inference engine
│   ├── web.py                   # Flask dashboard routes & controls
│   ├── requirements.txt         # Runtime requirements list
│   └── .env.example             # Environment template file
├── model/
│   └── poketwo_detector_full/   # Active 936-class detector assets
│       ├── pokemon_detector.onnx       # Compiled ONNX model
│       ├── pokemon_detector.onnx.data  # Model weight tensors
│       ├── metadata.json               # Index-to-class labels list
│       └── deployment_config.json      # Preprocessing configuration
├── scripts/
│   ├── download_dataset.py      # Pokétwo official metadata downloader
│   ├── download_kaggle_full.py  # Kaggle spawn image downloader
│   ├── prepare_full_manifest.py # Train/Val/Test manifest compiler
│   ├── train.py                 # PyTorch model training entry point
│   ├── evaluate.py              # Test set evaluation suite
│   ├── export_model.py          # PyTorch to ONNX converter script
│   └── infer.py                 # Standalone CLI prediction tool
├── POKETWO_FULL_COVERAGE_REPORT.md  # Detailed technical benchmark report
├── requirements.txt             # Primary runtime setup file
└── requirements-training.txt    # Optional model retraining requirements
```

---

## 🚀 Setup

### Prerequisites
- **Python**: Version `3.9` through `3.12` installed.
- **RAM**: ~1 GB of available system memory.
- **Discord**: An active user account.

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ramsingh4656/Poketwo-Auto-Catcher.git
   cd Poketwo-Auto-Catcher
   ```

2. **Set up a virtual environment:**
   - **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux/macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install the dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   > 📌 **Note**: This bot requires `discord.py-self`. If regular `discord.py` is installed in your global space, run `pip uninstall discord.py -y` first to prevent token format errors.

4. **Configure environment variables:**
   Copy the configuration template:
   - **Windows:** `copy bot\.env.example bot\.env`
   - **Linux/macOS:** `cp bot/.env.example bot/.env`

   Open `bot/.env` and edit the values:
   ```env
   USER_TOKEN=your_actual_discord_user_token_here
   CATCH_CHANNEL_ID=your_target_channel_id_here
   AUTOSTART=false
   PORT=5000
   CNN_CONFIDENCE_THRESHOLD=0.30
   ```
   > ⚠️ **Important**: Do **not** enclose your token in quotes.

### ▶️ How to Run the Bot

Launch the dashboard and bot using:
```bash
python bot/main.py
```
Open your browser and navigate to `http://localhost:5000/dashboard` to view statistics and start/stop the bot.

---

## 🧠 AI Model

The active classification model uses a **MobileNetV3-small** backbone exported to **ONNX Runtime format**, optimizing inference speed for typical host CPUs.

- **Class Universe**: Covers **936 labels** mapped from the official Pokétwo catchable dataset.
- **Test Evidence**: **839 labels** are supported with independent real-render Discord spawn screenshots. The remaining **97 labels** represent rare alternative forms that have official reference artwork but no active real-render samples in the dataset.
- **Abstention Logic**: Predictions scoring below `0.30` confidence return `None`, defaulting the bot's catch flow to the safe regex hint matching subsystem.

---

## 📊 Accuracy/Performance

Evaluation metrics on independent real-render test partitions:

| Metric | Measured Value |
|---|---|
| **Overall Top-1 Accuracy** | **92.2825%** |
| **Overall Top-3 Accuracy** | **97.5566%** |
| **Overall Top-5 Accuracy** | **98.5101%** |
| **Accepted Top-1 Accuracy (Confidence $\ge 0.30$)** | **98.8871%** |
| **Mean Inference Time (CPU)** | **~8.76 ms** / image |
| **P95 Inference Time (CPU)** | **~9.36 ms** / image |
| **Model Size** | **~9.58 MB** |

*Note: No AI detector achieves 100% accuracy. The bot protects your account by pairing visual predictions with a regular expression hint fallback system.*

---

## 🌐 Dashboard

The built-in Flask web panel provides a visual control suite:
- **State Toggle**: Start and stop the selfbot capture loop with click controls.
- **Live Statistics**: Monitor Total Caught, CNN catches, Hint catches, Skipped counts, and Uptime.
- **Real-Time Logs**: Review predictions, delays, raw API logs, and catch notices.

---

## 🛠️ Tech Stack

- **Core Language**: [Python](https://python.org) (v3.9+)
- **Inference Engine**: [ONNX Runtime](https://onnxruntime.ai) (CPU execution provider)
- **Image Processing**: [Pillow](https://python-pillow.org) (Lanczos resizing & RGB scaling)
- **Selfbot Driver**: [discord.py-self](https://github.com/dolfies/discord.py-self) (Discord user-agent client fork)
- **Web Interface**: [Flask](https://flask.palletsprojects.com) (WSGI dashboard engine)

---

## 🔧 Troubleshooting

### "Improper Token" or 401 Error
- Ensure you do not have regular `discord.py` installed alongside `discord.py-self`. Run:
  ```bash
  pip uninstall discord.py -y
  ```
- Verify that your token in `bot/.env` does **not** have double or single quotes surrounding it.

### Find Discord Channel ID
- Go to Discord Settings $\rightarrow$ Advanced $\rightarrow$ Enable **Developer Mode**.
- Right-click the desired spawn channel $\rightarrow$ Click **Copy Channel ID**.

### Find Discord User Token
- Log into Discord on a web browser.
- Press `F12` to open Developer Tools $\rightarrow$ Go to the Console tab.
- Paste the following code and press Enter to print your token:
  ```javascript
  (webpackChunkdiscord_app.push([[''],{},e=>{m=[];for(let c in e.c)m.push(e.c[c])}]),m).find(m=>m?.exports?.default?.getToken!==void 0).exports.default.getToken()
  ```

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE). 
Pokémon assets and trademarks belong to their respective owners.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome!
1. Fork this repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m 'Add your feature description'`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

---

<div align="center">
PokéCatcher is built with ❤️ for education & research.
</div>
