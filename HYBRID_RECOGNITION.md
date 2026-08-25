# Hybrid Pokémon Recognition

This repository now includes an optional, standalone hybrid recognition pipeline under `scripts/hybrid/`. It is kept separate from the existing 936-class ONNX deployment so the established runtime path is not replaced unexpectedly.

## Contents

| Path | Purpose |
|---|---|
| `model/hybrid/pokemon_classifier_augmented.tflite` | Recommended 1,659-class augmented TFLite model |
| `model/hybrid/pokemon_classifier.tflite` | Earlier baseline TFLite model |
| `model/hybrid/index_to_pokemon.json` | Label mapping for the hybrid models |
| `scripts/hybrid/build_hashes.py` | Downloads official Pokétwo assets and builds `phash_db.json` |
| `scripts/hybrid/hybrid_inference.py` | URL-based pHash, TFLite, and hint-resolution engine |
| `scripts/hybrid/train_augmented.py` | Reproducible augmented trainer |
| `scripts/hybrid/evaluate_augmented.py` | Deterministic evaluator |
| `requirements-hybrid.txt` | Optional dependencies for these tools |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-hybrid.txt
```

## Build the hash database

From the repository root:

```bash
python scripts/hybrid/build_hashes.py \
  --output model/hybrid/phash_db.json
```

The builder reads the public Pokétwo CSV and downloads numeric image assets. For transparent images it indexes the raw RGB image and deterministic composites on Discord dark mode `#313338` and light mode `#FFFFFF`. Hash collisions are retained as lists rather than silently selecting a wrong label.

A bounded smoke test is available before a full build:

```bash
python scripts/hybrid/build_hashes.py \
  --limit 3 \
  --output /tmp/phash_smoke.json
```

## Run inference

```bash
python scripts/hybrid/hybrid_inference.py \
  "IMAGE_URL" \
  --model model/hybrid/pokemon_classifier_augmented.tflite \
  --mapping model/hybrid/index_to_pokemon.json \
  --hash-db model/hybrid/phash_db.json \
  --hint "_ i _ a _ h u"
```

The detector first checks pHash exact and low-distance matches. If there is no unique hash result, it runs the TFLite classifier and reports the top five. A supplied hint is accepted only when exactly one top-five candidate matches; otherwise the engine abstains instead of guessing.

## Benchmark and limitations

The augmented model scored 79.93% top-1 and 91.11% top-5 on the repository-asset benchmark, which includes canonical base/shiny assets. This is not a guaranteed production score for real Discord embeds. Real, verified Pokétwo spawn screenshots are required to measure and improve domain accuracy. Augmentation can simulate backgrounds and compression but cannot create unseen poses, forms, or source artwork.

The scripts are transport-agnostic recognition components. They do not log in to Discord, read messages, or send catch commands. The existing project README documents the separate selfbot behavior and its associated Discord Terms of Service risk.
