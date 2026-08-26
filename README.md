<div align="center">

<img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png" width="120px" />

# PokéCatcher
### CNN-Powered Pokétwo Auto-Catcher

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX--Runtime-1.18+-005c99?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![discord.py-self](https://img.shields.io/badge/discord.py--self-latest-5865f2?style=for-the-badge&logo=discord&logoColor=white)](https://github.com/dolfies/discord.py-self)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

[Features](#-features) · [How It Works](#-how-it-works) · [Project Structure](#-project-structure) · [Setup](#-setup) · [AI Model](#-ai-model) · [Performance](#-accuracyperformance) · [Dashboard](#-dashboard) · [Tech Stack](#-tech-stack) · [Troubleshooting](#-troubleshooting) · [Contributing](#-contributing)

</div>

---

A CPU-oriented Pokétwo spawn-recognition bot with an ONNX image classifier, conservative confidence gating, and text-hint fallback.

> **Accuracy status:** The repository runs the verified Stage 5/7 **MobileNetV3-large direct-resize ONNX detector** as the live model. The results below are measured benchmark evidence, not an aspirational production-accuracy guarantee.

## Current status

| Item | Current state |
|---|---|
| Live bot model path | `model/poketwo_detector_full/pokemon_detector.onnx` |
| Live bot model | **MobileNetV3-large**, direct RGB resize |
| Closed-set label universe | 936 labels |
| Real-render evaluation coverage | 839 labels |
| Official-reference-only training classes | 97 labels |
| Runtime | ONNX Runtime on CPU |
| Live threshold | `CNN_CONFIDENCE_THRESHOLD`, default `0.85` |
| Configuration validation | `CATCH_CHANNEL_ID`, `PORT`, and `CNN_CONFIDENCE_THRESHOLD` fail closed with clear startup errors |
| Below-threshold behavior | Abstain from CNN catch and wait for Pokétwo's text hint |

## Measured results (locked test set)

Trained on a leakage-safe dataset of 35,574 records: 19,297 full real renders, 16,083 CC0 real renders, and 194 official-reference images for otherwise-missing classes. Split: 22,586 train / 6,533 validation / 6,455 locked held-out test.

The 97 official-reference-only labels have no held-out real-render support and cannot be validated by this benchmark.

| Locked-test metric | Result |
|---|---:|
| Top-1 accuracy | **84.79%** |
| Top-3 accuracy | **92.42%** |
| Top-5 accuracy | **94.53%** |
| Present-label macro F1 | **78.08%** |
| All-936 macro F1 | **69.98%** |
| Test images | 6,455 |

**95%+ practical accuracy was not achieved.** Top-5 retrieval should not be interpreted as automatic-catch accuracy.

## Confidence threshold and abstention

A prediction is accepted only when the top-1 softmax probability is at least `CNN_CONFIDENCE_THRESHOLD` (default `0.85`). Below that, the bot doesn't catch — it waits for Pokétwo's own text hint instead.

The `0.85` value was calibrated on validation data only (31.4% coverage, zero accepted errors), then checked once against the unseen locked test set:

| Threshold | Coverage | Abstention | Accuracy on accepted predictions | Accepted errors |
|---:|---:|---:|---:|---:|
| 0.85 | 26.94% | 73.06% | **99.94%** | 1 / 1,739 |

This is a selective operating point, not overall catch accuracy — the bot abstained on 4,716 of 6,455 test images.

## Image contract

```text
RGB, direct resize to 224×224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The live model achieved 100% top-1 decision agreement between PyTorch and ONNX Runtime on all 6,455 locked-test images. Measured single-image ONNX latency: 2.61 ms median, 3.01 ms P95 (four CPU threads; excludes image download/decode/preprocessing).

## Setup

### Prerequisites

Python 3.9 or newer (3.10/3.11 recommended). Windows, Linux, and macOS are supported. macOS users should confirm TensorFlow/ONNX Runtime wheels are available for their CPU architecture; the live detector itself only needs ONNX Runtime.

You'll also need Git, an internet connection, and a Discord account. This project automates a Discord **user account** via `discord.py-self` — that carries an unresolved Discord policy and account-risk concern. Review Discord's current terms before running it.

### 1. Clone the repository

```bash
git clone https://github.com/Ramsingh4656/Poketwo-Auto-Catcher.git
cd Poketwo-Auto-Catcher
```

The application entry point is `bot/main.py`, run from the repository root.

### 2. Install dependencies

```bash
python -m pip install -r bot/requirements.txt
```

This installs Flask, `discord.py-self`, TensorFlow, `aiohttp`, Pillow, NumPy, `python-dotenv`, and ONNX Runtime. The live path uses ONNX; TensorFlow is only a legacy fallback in `bot/predictor.py`.

Verify the install:

```bash
python -c "import discord, flask, onnxruntime, PIL, dotenv; print('dependencies: OK')"
```

### 3. Create the private environment file

The app reads `.env` from the **`bot/` directory**, not the repo root:

```bash
cp bot/.env.example bot/.env        # macOS/Linux
copy bot\.env.example bot\.env      # Windows Command Prompt
```

Edit `bot/.env`. There are exactly five variables:

| Variable | Required value and behavior |
|---|---|
| `USER_TOKEN` | Your Discord **user-account token**, required. Must be non-empty with three dot-separated parts. Never use a bot token or a `Bot ` prefix. |
| `CATCH_CHANNEL_ID` | Required numeric channel ID. A blank, zero, or non-numeric value stops startup rather than catching in all channels. |
| `CNN_CONFIDENCE_THRESHOLD` | Top-1 confidence required to catch. Keep the validated default `0.85`; lower values trade accuracy for coverage. |
| `AUTOSTART` | `false` (default) starts only the dashboard; the Discord client starts from there. `true`/`1`/`yes` starts it automatically. |
| `PORT` | Local Flask dashboard port. Default `5000`. |

```dotenv
USER_TOKEN=your_actual_discord_user_token
CATCH_CHANNEL_ID=123456789012345678
AUTOSTART=false
PORT=5000
CNN_CONFIDENCE_THRESHOLD=0.85
```

`.env` is gitignored, but double-check `git status` before committing anything. Never paste the token into an issue, chat, screenshot, or log. If exposed, treat it as compromised and revoke the session.

### 4. Get your user token safely

If you're authorized on your own account and accept the policy risk: open Discord in a browser, open developer tools' Network panel, inspect a request Discord makes, and read the request's authorization value. Never use token-grabber sites or third-party extensions. Don't share it, screenshot it, or commit it — only ever put it in `bot/.env`.

### 5. Start the application

```bash
python bot/main.py
```

This loads `bot/.env`, validates the token and channel ID, checks the selfbot library, loads `model/poketwo_detector_full/pokemon_detector.onnx` + `metadata.json`, and starts Flask.

With `AUTOSTART=false`, the dashboard starts but Discord doesn't log in until you press **Start Bot** there. With `AUTOSTART=true`, both start automatically.

### 6. Confirm it's healthy

Expect log lines like:

```text
Loaded .env from: .../Poketwo-Auto-Catcher/bot/.env
Token format OK: 3 parts, ... total chars
Discord library: discord.py-self <version> (OK)
Loaded Pokétwo ONNX detector (936 classes).
Bot user token: ...<last-4-chars>
Catch channel: <your numeric channel ID>
Model loaded: True
Dashboard: http://127.0.0.1:5000/dashboard
```

The token itself is never logged — only its last 4 characters. Open the dashboard, check `model_loaded: true`, then start the bot if `AUTOSTART=false`. On successful login it logs the account and `Catching ONLY in channel: <id>`.

**The dashboard is unauthenticated and binds to `0.0.0.0`.** Keep it on a trusted machine/network only — don't expose or port-forward it without adding authentication.

### 7. Common first-run issues

| Symptom | Meaning and fix |
|---|---|
| `USER_TOKEN environment variable is not set!` | `bot/.env` is missing, in the wrong folder, or empty. Re-copy the template and edit it. |
| `TOKEN FORMAT ERROR: Discord tokens have 3 parts separated by dots.` | Value is incomplete or a bot token. Use the full user-token value, no `Bot ` prefix. |
| `CATCH_CHANNEL_ID is required and must be set to a valid numeric channel ID — refusing to start to avoid catching in all channels` | Set a valid numeric channel ID. |
| `ModuleNotFoundError: No module named 'discord'` | Dependencies weren't installed — rerun step 2. |
| `WRONG DISCORD LIBRARY DETECTED!` | Regular `discord.py` got imported instead of `discord.py-self`. Reinstall from `bot/requirements.txt` in a clean environment. |
| `Failed to load ONNX detector; trying legacy Keras model.` | Check the `.onnx`/`metadata.json` files and the ONNX Runtime install. The Keras fallback is not the verified model. |
| `Loaded legacy TensorFlow detector; accuracy is not the verified ONNX model.` | The ONNX path failed. Fix that before relying on results. |
| `No usable Pokémon detector found; predictor disabled.` | Neither model pair loaded. Dashboard may still run, but inference is off. |
| `Model loaded: False` | Check the predictor error above this line. |
| `PORT must be a valid integer in the range 1-65535, got: '<value>'` | Set `PORT` to a valid, available integer. |
| Dashboard doesn't open | Confirm the process is running and the port isn't already in use. |
| Discord login error after local startup succeeds | Token may be invalid, expired, or restricted. Never send it to anyone for debugging. |

## Model and label limitations

This is a 936-class closed-set classifier. The 97 official-reference-only classes have no real-render validation/test evidence. Sparse per-class support and form-level confusion are real risks — observed confusions include Zygarde Core vs. Cell, Palafin vs. Finizen, and Squawkabilly plumage variants.

The model abstains rather than guesses when unsure. A hint is only accepted when the 936-label mapping yields a unique match. The separate 1,659-class hybrid TFLite/pHash experiment is **not** wired into `bot/` and must not be mixed with the 936-class ONNX mapping without retraining and re-exporting against the same label order.

## Security, policy, and operational limitations

This project automates a Discord **user account**, not a bot account — an unresolved Discord policy and account-risk concern. You're responsible for reviewing Discord's current terms.

The Flask dashboard has **no authentication**. Don't expose it to an untrusted network without adding auth, access control, and transport protection. Treat the Discord token as a high-impact secret — keep it only in the local, gitignored `.env` file.

## Evaluation and deployment status

Stage 7 locked-test evaluation is complete for the live model. MobileNetV3-large replaced the prior MobileNetV3-small in commit `1728caa5d2a2f37e1b8acf0f8058392011673ea2`, verified via post-swap load, sample-inference, preprocessing, and label-mapping checks. Any future model swap or config change should go through the same verification.

The repository still contains historical training/hybrid/experiment artifacts. A cleanup proposal exists but hasn't been applied — no destructive cleanup should happen until dependencies and rollback needs are reviewed.

**Deferred, not run:** ResNet18, ConvNeXt-Tiny, ViT-B/16 (CPU cost too high relative to expected benefit — see Stage 5 report for estimates).

## License and responsibility

See `LICENSE`. Use responsibly and keep all credentials private.
