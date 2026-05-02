"""
Pokemon CNN Predictor — loads the pre-trained Keras model and performs inference.
"""

import os
import io
import json
import logging
import asyncio
import functools
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("predictor")

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent
_MODEL_DIR = _BASE_DIR / "model"

MODEL_PATH = _MODEL_DIR / "pokemon_cnn.keras"
INDEX_MAP_PATH = _MODEL_DIR / "index_to_pokemon.json"

# Image dimensions expected by the CNN
IMG_SIZE = (128, 128)


class PokemonPredictor:
    """Wraps the trained CNN model for single-image Pokémon prediction."""

    def __init__(self) -> None:
        self.model = None
        self.index_to_pokemon: dict = {}
        self.loaded: bool = False
        self._load()

    # ── Private helpers ────────────────────────────────────────────────────────
    def _load(self) -> None:
        """Load the Keras model and the index-to-name mapping."""
        if not MODEL_PATH.exists():
            logger.warning("Model file not found at %s — predictor disabled.", MODEL_PATH)
            return
        if not INDEX_MAP_PATH.exists():
            logger.warning("Index map not found at %s — predictor disabled.", INDEX_MAP_PATH)
            return

        try:
            # Delay TensorFlow import so the module can still be imported on
            # machines without TF (e.g. for testing the rest of the bot).
            import tensorflow as tf

            # Suppress excessive TF logging
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            tf.get_logger().setLevel("ERROR")

            logger.info("Loading model from %s …", MODEL_PATH)
            self.model = tf.keras.models.load_model(str(MODEL_PATH))
            logger.info("Model loaded successfully.")

            with open(INDEX_MAP_PATH, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            # Keys may be stringified ints → normalise
            self.index_to_pokemon = {int(k): v for k, v in raw.items()}
            logger.info("Loaded %d class labels.", len(self.index_to_pokemon))

            self.loaded = True
        except Exception:
            logger.exception("Failed to load model / index map.")
            self.loaded = False

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Resize + normalise raw bytes into a model-ready batch tensor."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)  # shape: (1, 128, 128, 3)

    def _predict_sync(
        self, image_bytes: bytes, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Synchronous prediction returning a list of (name, confidence)."""
        if not self.loaded or self.model is None:
            return []

        tensor = self._preprocess(image_bytes)
        preds = self.model.predict(tensor, verbose=0)[0]  # shape: (num_classes,)

        # Get top-k indices
        top_indices = preds.argsort()[-top_k:][::-1]
        results: List[Tuple[str, float]] = []
        for idx in top_indices:
            idx = int(idx)
            name = self.index_to_pokemon.get(idx, f"unknown_{idx}")
            # Convert internal name format to Pokétwo-friendly name
            # e.g. "Charizard_Mega_X" → "Mega Charizard X", underscores → spaces
            name = self._format_name(name)
            confidence = float(preds[idx])
            results.append((name, confidence))
        return results

    @staticmethod
    def _format_name(raw_name: str) -> str:
        """Convert model label to a Pokétwo-compatible catch name.

        Poketwo generally uses the canonical name. For forms like
        ``Charizard_Mega_X`` → ``Mega Charizard X``, etc.
        Underscores become spaces; certain suffixes are handled.
        """
        name = raw_name.replace("_", " ")

        # Handle special form prefixes / suffixes
        # Mega forms: "Charizard Mega X" → "Mega Charizard X"
        if " Mega " in name:
            parts = name.split(" Mega ")
            name = f"Mega {parts[0]} {parts[1]}".strip()
        elif name.endswith(" Mega"):
            name = f"Mega {name[:-5]}".strip()

        # Gmax forms: "Charizard Gmax" → "Gigantamax Charizard"
        if name.endswith(" Gmax"):
            name = name[:-5].strip()

        # Alola forms: "Raichu Alola" → "Alolan Raichu"
        if name.endswith(" Alola"):
            name = f"Alolan {name[:-6]}".strip()

        # Galar forms: "Ponyta Galar" → "Galarian Ponyta"
        if name.endswith(" Galar"):
            name = f"Galarian {name[:-6]}".strip()

        # Hisuian forms
        if name.endswith(" Hisuian"):
            name = f"Hisuian {name[:-8]}".strip()

        # Paldea forms
        if "Paldea" in name:
            parts = name.split(" Paldea")
            name = f"Paldean {parts[0]}".strip()

        return name

    # ── Public async API ───────────────────────────────────────────────────────
    async def predict(
        self, image_bytes: bytes, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top-k predictions as ``[(name, confidence), …]``.

        Runs the model on a thread executor so the event loop stays responsive.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self._predict_sync, image_bytes, top_k)
        )

    async def predict_best(
        self, image_bytes: bytes, min_confidence: float = 0.70
    ) -> Optional[Tuple[str, float]]:
        """Return the best prediction only if it meets *min_confidence*.

        Returns ``(name, confidence)`` or ``None``.
        """
        results = await self.predict(image_bytes, top_k=1)
        if results and results[0][1] >= min_confidence:
            return results[0]
        return None
