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

> **Accuracy status:** The current repository contains the verified Stage 5/7 **MobileNetV3-large direct-resize ONNX detector**, now integrated as the live model. The measured results below apply to this live artifact; they are benchmark evidence, not an aspirational production-accuracy guarantee.

## Current status

| Item | Current state |
|---|---|
| Live bot model path | `model/poketwo_detector_full/pokemon_detector.onnx` |
| Live bot model in repository | **MobileNetV3-large**, according to `model/poketwo_detector_full/deployment_config.json` |
| Stage 5 accuracy candidate | **MobileNetV3-large, direct RGB resize — now live** |
| Closed-set label universe | 936 labels |
| Real-render evaluation coverage | 839 labels |
| Official-reference-only training classes | 97 labels |
| Runtime | ONNX Runtime on CPU |
| Live threshold | `CNN_CONFIDENCE_THRESHOLD`, default `0.85` in `bot/bot.py` |
| Configuration validation | `CATCH_CHANNEL_ID`, `PORT`, and `CNN_CONFIDENCE_THRESHOLD` fail closed with clear startup validation errors |
| Below-threshold behavior | Abstain from CNN catch and wait for Pokétwo’s text hint |
| Stage 7 status | Locked-test evaluation complete; winner integrated and smoke-tested |

## Measured Stage 5/7 winner results

The MobileNetV3-large direct-resize checkpoint was trained on a leakage-safe external dataset with 35,574 records: 19,297 full real renders, 16,083 CC0 real renders, and 194 official-reference images used for training-only coverage of otherwise missing classes. The split was 22,586 train, 6,533 validation, and 6,455 locked held-out test images.

The 936-class mapping is fixed. The test split contains real-render examples for 839 labels; the 97 official-reference-only labels have no held-out real-render support and cannot be validated by this benchmark.

| Locked-test metric | Measured result |
|---|---:|
| Top-1 accuracy | **84.78699%** |
| Top-3 accuracy | **92.42448%** |
| Top-5 accuracy | **94.53137%** |
| Present-label macro F1 | **78.07583%** |
| All-936 macro F1 | **69.98464%** |
| Test images | 6,455 |

**95%+ practical accuracy was not achieved.** Top-1 accuracy was 84.78699%, and even top-5 accuracy was 94.53137%. Top-5 retrieval should not be interpreted as automatic-catch accuracy.

## Confidence threshold and abstention

The live bot reads `CNN_CONFIDENCE_THRESHOLD` from the environment and defaults to `0.85`. A prediction is accepted only when the top-1 maximum softmax probability is at least that value. Otherwise, the bot does not catch from the CNN result and waits for the Pokétwo hint resolver.

The `0.85` value was selected using validation-only Stage 6 calibration. On validation, it produced 31.4% coverage with zero observed accepted errors. When applied once to the unseen locked test set, it produced:

| Threshold | Coverage | Abstention | Accuracy on accepted predictions | Accepted errors |
|---:|---:|---:|---:|---:|
| 0.85 | 26.94036% | 73.05964% | **99.94250%** | 1 / 1,739 |

The accepted-prediction result is **not overall catch accuracy**. It is a selective operating point: the bot abstained on 4,716 of 6,455 test images. The higher-coverage validation alternative of 0.75 is not currently selected as the default.

## Image contract

The deployed ONNX preprocessing contract is RGB direct resize to 224×224 followed by ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The verified Stage 5 winner is now the live artifact at `model/poketwo_detector_full/pokemon_detector.onnx`. It achieved 100% top-1 decision agreement with its PyTorch checkpoint on all 6,455 locked-test images. The measured single-image ONNX model latency was 2.6104 ms median and 3.0102 ms at P95 using four CPU threads; preprocessing, image download, and decoding are excluded.

## Setup

### Prerequisites

The project is intended for **Python 3.9 or newer**. Python 3.10 or 3.11 is a practical choice because the dependency file includes TensorFlow as a legacy-model fallback. Windows, Linux, and macOS are supported at the command-line level. macOS users should confirm that the pinned or resolved TensorFlow/ONNX Runtime wheels are available for their CPU architecture; the live detector itself uses ONNX Runtime on CPU.

You also need Git, an internet connection for installing Python packages and for the bot’s Discord/image traffic, and a Discord account that you understand may be subject to Discord’s current rules and account-policy restrictions. This repository automates a user account through `discord.py-self`; that selfbot approach carries an unresolved Discord policy and account-risk concern.

### 1. Clone the repository

Open a terminal, PowerShell window, or Command Prompt and clone the repository:

```bash
git clone https://github.com/Ramsingh4656/Poketwo-Auto-Catcher.git
cd Poketwo-Auto-Catcher
```

Run the application from this repository root with the command shown below. The application entry point is `bot/main.py`.

### 2. Create and activate a virtual environment

A virtual environment keeps this project’s packages separate from other Python projects.

On **Linux or macOS**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

On **Windows PowerShell**:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks activation because of its execution policy, either use Command Prompt or activate after allowing scripts for the current PowerShell process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On **Windows Command Prompt**:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python --version
```

After activation, upgrade `pip` in the virtual environment:

```bash
python -m pip install --upgrade pip
```

### 3. Install the project dependencies

Install the exact dependency set declared by the repository:

```bash
python -m pip install -r bot/requirements.txt
```

That file installs Flask, `discord.py-self`, TensorFlow, `aiohttp`, Pillow, NumPy, `python-dotenv`, and ONNX Runtime. The live path is the ONNX model; TensorFlow is retained because `bot/predictor.py` has a legacy Keras fallback.

You can perform a basic import check before configuring credentials:

```bash
python -c "import discord, flask, onnxruntime, PIL, dotenv; print('dependencies: OK')"
```

### 4. Create the private environment file

The application reads `.env` from the **`bot/` directory**, not from the repository root. Copy the provided template as follows.

On Linux or macOS:

```bash
cp bot/.env.example bot/.env
```

On Windows PowerShell:

```powershell
Copy-Item bot/.env.example bot/.env
```

On Windows Command Prompt:

```bat
copy bot\.env.example bot\.env
```

Edit `bot/.env`. The template currently contains exactly five variables; there are no additional variables required by `bot/main.py` or present in `bot/.env.example`.

| Variable | Required value and behavior |
|---|---|
| `USER_TOKEN` | Your Discord **user-account token**, required. The startup code validates that it is non-empty and has three dot-separated parts. Do not use a Discord developer-portal bot token and do not prefix the value with `Bot `. |
| `CATCH_CHANNEL_ID` | The numeric Discord channel ID where catches are allowed. This value is required and must be a positive numeric channel ID. A blank, zero, or non-numeric value stops startup with: `CATCH_CHANNEL_ID is required and must be set to a valid numeric channel ID — refusing to start to avoid catching in all channels`. |
| `CNN_CONFIDENCE_THRESHOLD` | The top-1 softmax confidence required before the CNN sends a catch command. Keep the validated default at `0.85`; lower values increase coverage but increase the risk of confidently wrong catches. |
| `AUTOSTART` | Set to `false` to start only the dashboard at launch and start the Discord client from the dashboard. Set to `true`, `1`, or `yes` to start the Discord client automatically. The default in the template is `false`. |
| `PORT` | Local Flask dashboard port. The default is `5000`; the dashboard URL is normally `http://127.0.0.1:5000/dashboard`. |

A typical private file is:

```dotenv
USER_TOKEN=your_actual_discord_user_token
CATCH_CHANNEL_ID=123456789012345678
AUTOSTART=false
PORT=5000
CNN_CONFIDENCE_THRESHOLD=0.85
```

The repository’s `.gitignore` ignores `.env` files, but still check `git status` before committing. `CATCH_CHANNEL_ID` is required; an empty or invalid value now stops startup rather than enabling all-channel catching. Never paste the token into an issue, chat, screenshot, shell transcript, log, or commit. If it is exposed, treat it as compromised and follow Discord’s available account-security or session-revocation procedures.

### 5. Obtain the user token safely and understand the policy risk

`USER_TOKEN` is a high-impact credential for a Discord user account. This project uses `discord.py-self`, so it expects a user token rather than a normal bot token. Automating a user account can violate Discord’s Terms of Service and can lead to account action; review Discord’s current rules before deciding whether to run it.

If you are authorized to use your own account and accept that policy risk, a factual way users commonly retrieve their own token is through the Discord web application’s browser developer tools: open the Network panel while signed in, inspect a request made by Discord, and locate the request’s authorization value. Do not use token-grabber websites, third-party extensions, or any service that asks you to paste the credential. Do not share the value with anyone, do not include it in a command copied into a public transcript, and do not commit it. Use only the value in the local ignored file `bot/.env`.

Do not confuse a developer-portal bot token with a user token. The project’s startup checks expect a three-part user-token shape and the selfbot library sends it as-is.

### 6. Start the application

From the repository root, with the virtual environment activated and `bot/.env` configured, run:

```bash
python bot/main.py
```

The script loads `bot/.env` relative to `bot/main.py`, validates the token and channel ID, checks that the selfbot library is loaded, constructs the bot, loads `model/poketwo_detector_full/pokemon_detector.onnx` with its adjacent `metadata.json`, and starts Flask.

With `AUTOSTART=false`, the dashboard starts but the Discord client does not log in until you press the dashboard’s **Start Bot** control. With `AUTOSTART=true`, the startup sequence also launches the Discord client automatically.

### 7. Confirm that startup is healthy

With a valid private `.env`, the first log messages should follow this pattern:

```text
Loaded .env from: .../Poketwo-Auto-Catcher/bot/.env
Token format OK: 3 parts, ... total chars
Discord library: discord.py-self <version> (OK)
Loaded Pokétwo ONNX detector (936 classes).
Bot user token: ...<last-4-chars> (last 4 chars)
Catch channel: <your numeric channel ID>
Model loaded: True
Dashboard: http://127.0.0.1:5000/dashboard
```

The exact timestamp, token length, library version, channel ID, and last four token characters vary. The token itself should never appear in logs; the application intentionally logs only its final four characters.

Open the dashboard at [`http://127.0.0.1:5000/dashboard`](http://127.0.0.1:5000/dashboard), replacing `5000` if you changed `PORT`. The dashboard status view should show `model_loaded: true` after the model is loaded. The `/dashboard/status` endpoint exposes the running state, model-loaded state, current bot state, recent logs, and counters. If `AUTOSTART=false`, use the dashboard’s start control after confirming the model loaded. When Discord login succeeds, the bot log should report the logged-in account and `Catching ONLY in channel: <id>`.

The dashboard is unauthenticated and the Flask process binds to `0.0.0.0`; keep it on a trusted machine/network and do not expose or port-forward it without adding authentication and transport protection.

### 8. Common first-run issues

| Symptom | Meaning and fix |
|---|---|
| `USER_TOKEN environment variable is not set!` | `bot/.env` is missing, is in the repository root instead of `bot/`, or still contains an empty/placeholder value. Copy the template again and edit `bot/.env`. |
| `TOKEN FORMAT ERROR: Discord tokens have 3 parts separated by dots.` | The value is incomplete, is a bot token rather than a user token, or includes the wrong text. Copy the full authorized user-token value; do not add `Bot `. |
| `CATCH_CHANNEL_ID is required and must be set to a valid numeric channel ID — refusing to start to avoid catching in all channels` | `CATCH_CHANNEL_ID` is missing, blank, zero, or non-numeric. Set it to the numeric Discord channel ID before restarting. The bot now fails closed instead of enabling all-channel catching. |
| `ModuleNotFoundError: No module named 'discord'` or a similar import error | The virtual environment is not active or `python -m pip install -r bot/requirements.txt` was not run in that environment. |
| `WRONG DISCORD LIBRARY DETECTED!` | Regular `discord.py` was imported instead of `discord.py-self`. Use a clean virtual environment and install `bot/requirements.txt`; the current code also attempts to uninstall `discord.py` and restart, which is another reason to avoid mixing environments. |
| `Failed to load ONNX detector; trying legacy Keras model.` | The ONNX file/metadata could not be loaded or ONNX Runtime failed. Check `model/poketwo_detector_full/pokemon_detector.onnx`, `model/poketwo_detector_full/metadata.json`, and the ONNX Runtime installation. Do not assume that a subsequent legacy TensorFlow load has the same accuracy or preprocessing contract. |
| `Loaded legacy TensorFlow detector; accuracy is not the verified ONNX model.` | The live ONNX path was bypassed and the legacy `model/pokemon_cnn.keras` fallback was loaded. Treat this as an unsuccessful deployment check and fix the ONNX problem. |
| `No usable Pokémon detector found; predictor disabled.` | Neither the live ONNX pair nor the legacy Keras pair could be loaded. The Flask dashboard may still start, but image inference is disabled. |
| `Model loaded: False` | The predictor did not load a usable detector. Review the preceding predictor error rather than starting the Discord client. |
| Dashboard does not open | Confirm that the process is still running, use the configured `PORT`, and check whether another process already occupies that port. |
| `PORT must be a valid integer in the range 1-65535, got: 'not-a-port'` | `PORT` must be an integer from 1 through 65535 and the selected port must be available to the current user. Replace the example value with the actual invalid value shown in your log. |
| Discord login/authentication error after local startup succeeds | The token may be invalid, expired, incomplete, or subject to account restrictions. Never send the token to a third party for debugging. |
## Model and label limitations

This is a 936-class closed-set classifier. The 97 classes with official-reference-only training images do not have real-render validation/test evidence in the Stage 5 dataset. Sparse support for many classes and form-level confusion remain significant risks. Observed locked-test confusion patterns include Zygarde core versus cell, Palafin versus Finizen, and Squawkabilly plumage variants.

The model may abstain rather than guess. A hint is accepted only when the authoritative 936-label mapping yields a unique match. The separate 1,659-class hybrid TFLite/pHash experiment is not wired into `bot/` and must not be mixed with the 936-class ONNX mapping without retraining and re-exporting against the same label order.

## Security, policy, and operational limitations

This project automates a Discord user account rather than using a conventional bot account. That creates an unresolved Discord selfbot policy and account-risk concern; users are responsible for reviewing Discord’s current terms and applicable policies.

The Flask dashboard is unauthenticated. It must not be exposed to an untrusted network without adding authentication, access control, and transport protection. Treat the Discord user token as a high-impact secret: keep it in the local ignored `.env` file and never place it in logs, screenshots, commits, or issue reports.

## Evaluation and deployment status

Stage 7 is complete for the live Stage 5 winner. The locked-test report, per-class breakdown, parity results, and cleanup proposal are maintained as audit artifacts. The Stage 5 winner replaced the prior live MobileNetV3-small model in commit `1728caa5d2a2f37e1b8acf0f8058392011673ea2` and passed post-swap loading, sample-inference, preprocessing, and exact 936-label mapping checks. Any future model integration, metadata replacement, or live configuration change requires a separate approved change.

The repository currently contains historical training, hybrid, and experiment artifacts. A cleanup proposal has been prepared but not applied. No destructive cleanup should be performed until dependencies and rollback requirements are reviewed.

## License and responsibility

See `LICENSE` for the repository license. Use the software responsibly and keep all credentials private.
