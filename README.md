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

1. Install the dependencies from `bot/requirements.txt` and ensure ONNX Runtime is available.
2. Copy the environment template:

   ```bash
   cp bot/.env.example bot/.env
   ```

3. Set the Discord user token and target channel ID in `bot/.env`. Keep the file private and never commit it.
4. Review the model path and metadata in `model/poketwo_detector_full/`.
5. Start the application:

   ```bash
   python bot/main.py
   ```

The application loads `bot/.env` before importing the bot. A process restart is required after changing the threshold or other environment variables.

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
