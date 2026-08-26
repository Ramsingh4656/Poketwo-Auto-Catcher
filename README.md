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

> **Accuracy status:** The repository runs the verified Stage 5/7 **MobileNetV3-large direct-resize ONNX detector** as the live model. The figures below are measured benchmark evidence, not an aspirational production-accuracy guarantee. The bot abstains rather than guesses whenever it is not confident.

## ✨ Features

- **On-device ONNX classifier** — MobileNetV3-large, 936-class closed set, running on CPU through ONNX Runtime. No GPU required.
- **Conservative confidence gate** — a spawn is auto-caught only when the top-1 softmax probability is at least `CNN_CONFIDENCE_THRESHOLD` (default `0.85`). Below that, the bot **abstains instead of guessing**.
- **Text-hint fallback** — when the classifier abstains, the bot waits for Pokétwo's own hint (`The pokémon is **...**`) and resolves it against the same 936-label universe, accepting only a unique match.
- **Optional P2 Assistant fallback** — an additive, opt-in hint source (`P2_ASSISTANT_ID`); it never overrides Pokétwo.
- **Single-channel safety** — the bot refuses to start without a valid `CATCH_CHANNEL_ID`, so it can never catch in every channel.
- **Human-like pacing** — randomized 2–5 s pre-catch delays, with an occasional longer "distraction" delay.
- **Fail-closed configuration** — token format, channel ID, port, and threshold are validated at startup with clear, actionable errors.
- **ONNX-only by default** — the legacy TensorFlow/Keras model is never selected automatically; it requires an explicit `ALLOW_LEGACY_FALLBACK=true` opt-in and is unverified.
- **Local Flask dashboard** — live stats, start/stop controls, and logs, with optional HTTP Basic Auth and localhost-only binding by default.
- **Secret hygiene** — the Discord token is read only from a gitignored `bot/.env`; only its last 4 characters are ever logged.

## 🧠 How It Works

The bot is a `discord.py-self` client that only reacts inside your configured channel. Its catch loop is a small state machine (`IDLE → IDENTIFYING → WAITING_FOR_HINT → WAITING_FOR_RESULT`):

1. **Detect** — it watches for Pokétwo's spawn embed (`A wild pokémon has appeared!`).
2. **Wait** — it sleeps a randomized human-like delay (2–5 s, occasionally longer) before acting.
3. **Identify** — it downloads the spawn image and runs the ONNX classifier. If the top-1 confidence is at or above the threshold, it sends `@Pokétwo catch <name>`.
4. **Fall back** — if the classifier is not confident, it abstains and waits for Pokétwo's text hint, resolves it against the 936-label mapping, and catches only on a unique match. If `P2_ASSISTANT_ID` is enabled, it also requests a P2 Assistant hint.
5. **Confirm** — it reads Pokétwo's response, records the result, and returns to idle. Hints and results have timeouts, so a missed message can't wedge the bot.

The image contract and model details are under [AI Model](#-ai-model).

## 📁 Project Structure

```text
Poketwo-Auto-Catcher/
├── bot/                         # Application (runtime)
│   ├── main.py                  # Entry point: env validation, dashboard, optional autostart
│   ├── bot.py                   # Selfbot: spawn detection, catch loop, state machine
│   ├── predictor.py             # ONNX predictor (+ opt-in legacy fallback)
│   ├── pokemon_data.py          # 936-label hint resolver / name normalization
│   ├── web.py                   # Flask dashboard (stats, controls, optional auth)
│   ├── requirements.txt         # Runtime dependencies
│   └── .env.example             # Configuration template (copy to bot/.env)
├── model/
│   └── poketwo_detector_full/   # Live model artifacts (do not edit)
│       ├── pokemon_detector.onnx
│       ├── metadata.json        # 936-label array + architecture
│       └── deployment_config.json
├── scripts/                     # Training, evaluation, export, and data-tooling scripts
├── reports/                     # Evaluation, coverage, and provenance outputs (JSON/CSV)
├── data/processed/              # Dataset manifests (no raw images committed)
├── tests/                       # e.g. test_hint_resolution.py
├── archive/                     # Historical experiments + experimental hybrid path (not wired in)
├── requirements.txt             # Includes bot/requirements.txt
├── requirements-training.txt    # Extra deps for training/export (PyTorch, etc.)
└── LICENSE
```

The only pieces needed to run the bot are `bot/` and `model/poketwo_detector_full/`. Everything under `scripts/`, `reports/`, `data/`, and `archive/` supports the training/evaluation history and is not required at runtime.

## 🚀 Setup

### Prerequisites

- **Python 3.9+** (3.10 / 3.11 recommended), plus **Git** and an internet connection.
- A **Discord account**. This project automates a Discord **user account** through `discord.py-self` — an unresolved Discord policy and account-risk concern. Review Discord's current terms before running it (see [Security, policy, and operational limitations](#-security-policy-and-operational-limitations)).
- Windows, Linux, and macOS are supported. The live detector only needs ONNX Runtime; macOS users should confirm ONNX Runtime wheels (and, if they opt into the legacy fallback, TensorFlow wheels) exist for their CPU architecture.

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

This installs Flask, `discord.py-self`, TensorFlow, `aiohttp`, Pillow, NumPy, `python-dotenv`, and ONNX Runtime. The live path uses ONNX Runtime; TensorFlow is only pulled in for the opt-in legacy fallback in `bot/predictor.py`.

Verify the install:

```bash
python -c "import discord, flask, onnxruntime, PIL, dotenv; print('dependencies: OK')"
```

### 3. Configure `bot/.env`

The app reads `.env` from the **`bot/` directory**, not the repo root. Copy the template:

**macOS / Linux**
```bash
cp bot/.env.example bot/.env
```

**Windows (Command Prompt)**
```bash
copy bot\.env.example bot\.env
```

Then edit `bot/.env`. There are **nine** supported variables — two required, the rest optional with safe defaults:

| Variable | Required? | Default | What it does |
|---|---|---|---|
| `USER_TOKEN` | **Required** | — | Your Discord **user-account token**. No `Bot ` prefix, no surrounding quotes. |
| `CATCH_CHANNEL_ID` | **Required** | — | Numeric ID of the one channel the bot may catch in. Startup fails if it is missing or non-numeric, so the bot can never catch in every channel. |
| `AUTOSTART` | Optional | `false` | `false` starts only the dashboard; you press **Start Bot** there. `true` / `1` / `yes` logs the Discord client in automatically. |
| `PORT` | Optional | `5000` | Local Flask dashboard port. Must be an integer in `1–65535`. |
| `CNN_CONFIDENCE_THRESHOLD` | Optional | `0.85` | Top-1 confidence required to auto-catch. Must be between `0` and `1`. Keep the validated `0.85`; lower values trade accuracy for coverage. |
| `DASHBOARD_HOST` | Optional | `127.0.0.1` | Dashboard bind address. Localhost-only by default. Change it only for deliberate network exposure — and set `DASHBOARD_PASSWORD` if you do. |
| `DASHBOARD_PASSWORD` | Optional | *(blank)* | Enables HTTP Basic Auth on every dashboard route when set. Blank is allowed for trusted localhost use, but startup logs a warning. Strongly recommended before any network exposure. |
| `P2_ASSISTANT_ID` | Optional | *(blank)* | Numeric user ID for the optional P2 Assistant hint fallback. Blank disables it. Set `854233015475109888` (or your server's P2 Assistant ID) to enable. An invalid value logs a warning and disables the feature without blocking startup. |
| `ALLOW_LEGACY_FALLBACK` | Optional | `false` | Degraded-mode opt-in. `false` means startup refuses to continue if the verified ONNX detector fails to load. Set `true` only for deliberate troubleshooting — the fallback is a **different** TensorFlow/Keras model with a **different 1,218-label** universe, and none of the accuracy figures below apply to it. |

A minimal working `bot/.env`:

```dotenv
USER_TOKEN=your_actual_discord_user_token
CATCH_CHANNEL_ID=123456789012345678
AUTOSTART=false
PORT=5000
CNN_CONFIDENCE_THRESHOLD=0.85
ALLOW_LEGACY_FALLBACK=false
DASHBOARD_PASSWORD=
DASHBOARD_HOST=127.0.0.1
P2_ASSISTANT_ID=
```

### 4. Get your Discord user token safely

If you are authorized on your own account and accept the policy risk: open Discord in a browser, open your browser's developer tools, go to the **Network** panel, inspect a request Discord makes, and read its `authorization` value. **Never** use token-grabber sites or third-party extensions. Don't share, screenshot, or commit the token — put it only in `bot/.env` (which is gitignored).

### 5. Start the application

```bash
python bot/main.py
```

This loads `bot/.env`, validates the token and channel ID, confirms the selfbot library, loads `model/poketwo_detector_full/pokemon_detector.onnx` + `metadata.json`, and starts Flask. If the verified ONNX detector cannot load, startup **fails** rather than silently switching models — unless you deliberately set `ALLOW_LEGACY_FALLBACK=true`.

- With `AUTOSTART=false` (default): the dashboard starts, but Discord doesn't log in until you press **Start Bot**.
- With `AUTOSTART=true`: both start automatically.

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

Open the dashboard, confirm **Model: Loaded ✓** (`model_loaded: true`), then press **Start Bot** if `AUTOSTART=false`. On successful login the bot logs the account and `Catching ONLY in channel: <id>`. The token itself is never logged — only its last 4 characters.

If something doesn't line up, see [Troubleshooting](#-troubleshooting).

## 🤖 AI Model

| Item | Current state |
|---|---|
| Live model path | `model/poketwo_detector_full/pokemon_detector.onnx` |
| Architecture | **MobileNetV3-large**, direct RGB resize |
| Closed-set label universe | 936 labels |
| Real-render evaluation coverage | 839 labels |
| Official-reference-only training classes | 97 labels |
| Runtime | ONNX Runtime on CPU |
| Live threshold | `CNN_CONFIDENCE_THRESHOLD`, default `0.85` |
| Below-threshold behavior | Abstain from the CNN catch and wait for Pokétwo's text hint |
| Legacy fallback policy | ONNX-only by default; TensorFlow/Keras fallback needs `ALLOW_LEGACY_FALLBACK=true` and is unverified |

### Image contract

```text
RGB, direct resize to 224×224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The live model achieved **100% top-1 decision agreement** between PyTorch and ONNX Runtime across all 6,455 locked-test images. Measured single-image ONNX latency: **2.61 ms median, 3.01 ms P95** (four CPU threads; excludes image download, decode, and preprocessing).

### Label universe

The 936 labels come from `model/poketwo_detector_full/metadata.json`, which is authoritative because the deployed model can only emit those class indices. The live hint resolver searches the **same** 936 labels and returns a name only when exactly one label matches the normalized hint; ambiguous or unmatched hints abstain. When the model is retrained or replaced, the mapping must be regenerated and the class-count check must pass before deployment. For the full rationale, see [LABEL_UNIVERSE.md](LABEL_UNIVERSE.md).

## 📊 Accuracy/Performance

All figures are measured evidence from the locked, held-out test set. **No 100% catch-accuracy claim is made.**

### Measured results (locked test set)

Trained on a leakage-controlled dataset of **35,574 records**: 19,297 full real renders, 16,083 CC0 real renders, and 194 official-reference images for otherwise-missing classes. Split: **22,586 train / 6,533 validation / 6,455 locked held-out test**. The 97 official-reference-only labels have no held-out real-render support and cannot be validated by this benchmark.

| Locked-test metric | Result |
|---|---:|
| Top-1 accuracy | **84.79%** |
| Top-3 accuracy | **92.42%** |
| Top-5 accuracy | **94.53%** |
| Present-label macro F1 | **78.08%** |
| All-936 macro F1 | **69.98%** |
| Test images | 6,455 |

**95%+ practical accuracy was not achieved.** Top-5 retrieval should not be interpreted as automatic-catch accuracy.

### Confidence threshold and abstention

A prediction is accepted only when the top-1 softmax probability is at least `CNN_CONFIDENCE_THRESHOLD` (default `0.85`). Below that, the bot doesn't catch — it waits for Pokétwo's text hint instead.

The `0.85` value was calibrated on validation data only (31.4% coverage, zero accepted errors), then checked **once** against the unseen locked test set:

| Threshold | Coverage | Abstention | Accuracy on accepted predictions | Accepted errors |
|---:|---:|---:|---:|---:|
| 0.85 | 26.94% | 73.06% | **99.94%** | 1 / 1,739 |

This is a **selective operating point**, not overall catch accuracy — the bot abstained on 4,716 of 6,455 test images. That is by design: it prefers the hint fallback over a low-confidence guess.

## 🖥️ Dashboard

`bot/web.py` serves a small Flask dashboard (default `http://127.0.0.1:5000/dashboard`) showing total caught, CNN catches, hint catches, skipped, and uptime, plus **Start / Stop** controls and a live log tail. Routes: `/dashboard`, `/dashboard/status`, `/dashboard/start`, `/dashboard/stop`, `/dashboard/settings`.

- **Binding:** localhost-only by default (`DASHBOARD_HOST=127.0.0.1`).
- **Auth:** setting `DASHBOARD_PASSWORD` turns on HTTP Basic Auth for every dashboard route (use any username plus the configured password). If it is blank, startup logs a clear warning that dashboard authentication is disabled.
- Keep the dashboard on localhost and set a strong password before any deliberate network exposure.

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Inference | **ONNX Runtime** (CPU), NumPy, Pillow |
| Discord client | **`discord.py-self`**, `aiohttp` |
| Dashboard | **Flask**, `python-dotenv` |
| Legacy fallback (opt-in only) | TensorFlow / Keras |
| Training & export (see `requirements-training.txt`) | PyTorch, torchvision, scikit-learn, `onnx`, `onnxscript`, `psutil` |

## 🧩 Troubleshooting

| Symptom | Meaning and fix |
|---|---|
| `USER_TOKEN environment variable is not set!` | `bot/.env` is missing, in the wrong folder, or empty. Re-copy the template and edit it. |
| `TOKEN FORMAT ERROR: Discord tokens have 3 parts separated by dots.` | The value is incomplete or a bot token. Use the full user-token value, with no `Bot ` prefix and no quotes. |
| `CATCH_CHANNEL_ID is required and must be set to a valid numeric channel ID — refusing to start to avoid catching in all channels` | Set a valid numeric channel ID. |
| `ModuleNotFoundError: No module named 'discord'` | Dependencies weren't installed — rerun step 2. |
| `WRONG DISCORD LIBRARY DETECTED!` | Regular `discord.py` was imported instead of `discord.py-self`. `main.py` tries to uninstall it automatically; if that fails, run `pip uninstall discord.py -y` and reinstall from `bot/requirements.txt` in a clean environment. |
| `ONNX detector failed to load: ... Refusing to start with an unverified/incompatible legacy model` | The ONNX artifact, metadata, ONNX Runtime install, or model contract is invalid. Fix the ONNX deployment; don't enable the fallback unless you intentionally accept degraded, unverified behavior. |
| `DEGRADED LEGACY MODE ACTIVE: ... This is NOT the verified 936-class ... model; none of the README accuracy figures apply.` | `ALLOW_LEGACY_FALLBACK=true` is set. The running detector is a different model with a different label universe — treat its results as unverified and turn the opt-in back off after troubleshooting. |
| `Model loaded: False` | Check the predictor startup error logged just above this line. |
| `PORT must be a valid integer in the range 1-65535, got: '<value>'` | Set `PORT` to a valid, available integer. |
| `P2_ASSISTANT_ID is invalid; P2 Assistant feature disabled.` | The optional P2 Assistant fallback is off. Leave the variable blank/unset, or set it to the Assistant's numeric user ID. |
| `Dashboard authentication required.` / HTTP `401` | `DASHBOARD_PASSWORD` is set. Open the dashboard with HTTP Basic Auth using any username and the configured password. |
| `Dashboard authentication is disabled; set DASHBOARD_PASSWORD ...` | `DASHBOARD_PASSWORD` is blank. Fine for trusted localhost use; set a password before deliberate network exposure. |
| Dashboard doesn't open | Confirm the process is running, the configured `DASHBOARD_HOST` is reachable, and the port isn't already in use. |
| Discord login error after local startup succeeds | The token may be invalid, expired, or restricted. Never send it to anyone for debugging. |

## ⚠️ Model and label limitations

This is a **936-class closed-set** classifier. The 97 official-reference-only classes have **no** real-render validation/test evidence and are not represented as independently validated high-accuracy classes. Sparse per-class support and form-level confusion are real risks — observed confusions include Zygarde Core vs. Cell, Palafin vs. Finizen, and Squawkabilly plumage variants.

The model abstains rather than guesses when unsure, and a hint is accepted only when the 936-label mapping yields a unique match. The separate 1,659-class hybrid TFLite/pHash experiment under `archive/` is **not** wired into `bot/` and must not be mixed with the 936-class ONNX mapping without retraining and re-exporting against the same label order.

## 🔒 Security, policy, and operational limitations

- This project automates a Discord **user account** (a "selfbot"), not a bot account — an unresolved Discord policy and account-risk concern. You are responsible for reviewing Discord's current terms; **use at your own risk.**
- Treat the Discord token as a high-impact secret. Keep it only in the local, gitignored `bot/.env`. Only its last 4 characters are ever logged.
- The Flask dashboard binds to `127.0.0.1` by default (`DASHBOARD_HOST`) and supports optional HTTP Basic Auth (`DASHBOARD_PASSWORD`). Set a strong password before any deliberate network exposure, and never expose the dashboard to an untrusted network.

## ✅ Evaluation and deployment status

Stage 7 locked-test evaluation is **complete** for the live model. MobileNetV3-large replaced the prior MobileNetV3-small in commit `1728caa5d2a2f37e1b8acf0f8058392011673ea2`, verified via post-swap load, sample-inference, preprocessing, and label-mapping checks. Any future model swap or config change should go through the same verification.

Historical training/experiment artifacts and the experimental hybrid pipeline now live under `archive/` and are independent of the live runtime in `bot/` and `model/poketwo_detector_full/`.

**Deferred, not run:** ResNet18, ConvNeXt-Tiny, ViT-B/16 (CPU cost judged too high relative to expected benefit).

## 🤝 Contributing

Issues and pull requests are welcome. A few ground rules keep the project honest:

- **Never overstate accuracy.** Keep every figure tied to measured, reproducible evaluation; don't round misleadingly or drop limitation disclosures.
- **Any model swap must pass verification** — load, sample-inference, preprocessing, and label-mapping/class-count checks — before it is deployed, and the mapping must be regenerated for a new label order.
- **Keep the runtime ONNX-only by default.** The legacy fallback stays opt-in.
- Don't commit secrets or datasets; `bot/.env`, raw data, and model checkpoints are gitignored.

## 📄 License and responsibility

Released under the MIT License — see [`LICENSE`](LICENSE). Use responsibly, follow the policies of any service you interact with, and keep all credentials private.
