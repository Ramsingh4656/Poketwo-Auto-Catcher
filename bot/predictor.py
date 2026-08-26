"""CPU image predictor for Pokétwo spawn images.

The primary detector is the verified 936-class ONNX Runtime export in
``model/poketwo_detector_full/``. The earlier Keras model is never selected
automatically; it is available only when ``ALLOW_LEGACY_FALLBACK=true`` is set
as an explicit, degraded-mode opt-in.
"""

from __future__ import annotations

import asyncio
import functools
import io
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("predictor")
_BASE_DIR = Path(__file__).resolve().parent.parent
_NEW_MODEL_DIR = _BASE_DIR / "model" / "poketwo_detector_full"
_ONNX_PATH = _NEW_MODEL_DIR / "pokemon_detector.onnx"
_METADATA_PATH = _NEW_MODEL_DIR / "metadata.json"
_LEGACY_MODEL_PATH = _BASE_DIR / "model" / "pokemon_cnn.keras"
_LEGACY_INDEX_PATH = _BASE_DIR / "model" / "index_to_pokemon.json"
_LEGACY_CLASS_INDICES_PATH = _BASE_DIR / "model" / "class_indices.json"
IMG_SIZE = (224, 224)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_ALLOW_LEGACY_FALLBACK = os.getenv("ALLOW_LEGACY_FALLBACK", "").strip().lower() in (
    "1", "true", "yes", "on"
)


class PokemonPredictor:
    """Wrap the deployed model and return Pokétwo-compatible names."""

    def __init__(self) -> None:
        self.session = None
        self.model = None
        self.index_to_pokemon: dict[int, str] = {}
        self.loaded = False
        self.backend = "none"
        self._load()

    def _load(self) -> None:
        onnx_error = None
        if not _ONNX_PATH.exists() or not _METADATA_PATH.exists():
            missing = []
            if not _ONNX_PATH.exists():
                missing.append(str(_ONNX_PATH))
            if not _METADATA_PATH.exists():
                missing.append(str(_METADATA_PATH))
            onnx_error = "required artifact(s) missing: " + ", ".join(missing)
        else:
            try:
                import onnxruntime as ort
                metadata = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
                self.index_to_pokemon = {i: name for i, name in enumerate(metadata["labels"])}
                self.session = ort.InferenceSession(
                    str(_ONNX_PATH),
                    providers=["CPUExecutionProvider"],
                    sess_options=self._session_options(),
                )
                output_classes = self.session.get_outputs()[0].shape[-1]
                if output_classes != len(self.index_to_pokemon):
                    raise ValueError(f"ONNX classes={output_classes}, labels={len(self.index_to_pokemon)}")
                self.loaded = True
                self.backend = "onnxruntime"
                logger.info("Loaded Pokétwo ONNX detector (%d classes).", len(self.index_to_pokemon))
                return
            except Exception as exc:
                onnx_error = f"{type(exc).__name__}: {exc}"
                logger.exception("ONNX detector load failed.")

        failure = (
            f"ONNX detector failed to load: {onnx_error}. "
            "Refusing to start with an unverified/incompatible legacy model — "
            "fix the ONNX artifact or explicitly enable the legacy fallback."
        )
        if not _ALLOW_LEGACY_FALLBACK:
            raise RuntimeError(failure)

        logger.critical(
            "ALLOW_LEGACY_FALLBACK is enabled. Attempting degraded legacy mode; "
            "this is a DIFFERENT model with a DIFFERENT 1,218-label universe, "
            "not the verified 936-class MobileNetV3-large. None of the README "
            "accuracy figures apply to this fallback."
        )
        if _LEGACY_MODEL_PATH.exists() and _LEGACY_INDEX_PATH.exists():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(_LEGACY_MODEL_PATH))
                model_classes = int(self.model.output_shape[-1])
                mapping_path = (
                    _LEGACY_CLASS_INDICES_PATH
                    if _LEGACY_CLASS_INDICES_PATH.exists()
                    else _LEGACY_INDEX_PATH
                )
                raw = json.loads(mapping_path.read_text(encoding="utf-8"))
                if mapping_path == _LEGACY_CLASS_INDICES_PATH:
                    self.index_to_pokemon = {int(index): name for name, index in raw.items()}
                else:
                    self.index_to_pokemon = {int(index): name for index, name in raw.items()}
                if model_classes != len(self.index_to_pokemon):
                    raise ValueError(
                        f"legacy model classes={model_classes}, labels={len(self.index_to_pokemon)} "
                        f"from {mapping_path.name}"
                    )
                self.loaded = True
                self.backend = "tensorflow"
                logger.critical(
                    "DEGRADED LEGACY MODE ACTIVE: TensorFlow/Keras model loaded with "
                    "%d labels. This is a DIFFERENT model with a DIFFERENT label universe, "
                    "not the verified 936-class MobileNetV3-large. None of the README "
                    "accuracy figures apply.",
                    len(self.index_to_pokemon),
                )
                return
            except Exception as exc:
                logger.exception("Legacy TensorFlow fallback also failed to load.")
                raise RuntimeError(f"{failure} Legacy fallback also failed: {exc}") from exc

        raise RuntimeError(
            f"{failure} ALLOW_LEGACY_FALLBACK=true was set, but the legacy model "
            "or mapping is missing."
        )

    @staticmethod
    def _session_options():
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

    def _predict_sync(self, image_bytes: bytes, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.loaded:
            return []
        tensor = self._preprocess(image_bytes)
        if self.backend == "onnxruntime":
            logits = self.session.run([self.session.get_outputs()[0].name], {"images": tensor})[0][0]
            logits = logits - np.max(logits)
            exp = np.exp(logits)
            preds = exp / exp.sum()
        else:
            preds = self.model.predict(np.transpose(tensor, (0, 2, 3, 1)), verbose=0)[0]
        top_k = min(max(1, top_k), len(preds))
        indices = np.argsort(preds)[-top_k:][::-1]
        return [(self._format_name(self.index_to_pokemon.get(int(i), f"unknown_{i}")), float(preds[i])) for i in indices]

    @staticmethod
    def _format_name(raw_name: str) -> str:
        name = raw_name.replace("_", " ")
        if " Mega " in name:
            parts = name.split(" Mega "); name = f"Mega {parts[0]} {parts[1]}".strip()
        elif name.endswith(" Mega"):
            name = f"Mega {name[:-5]}".strip()
        if name.endswith(" Gmax"): name = name[:-5].strip()
        if name.endswith(" Alola"): name = f"Alolan {name[:-6]}".strip()
        if name.endswith(" Galar"): name = f"Galarian {name[:-6]}".strip()
        if name.endswith(" Hisuian"): name = f"Hisuian {name[:-8]}".strip()
        if "Paldea" in name:
            parts = name.split(" Paldea"); name = f"Paldean {parts[0]}".strip()
        return name

    async def predict(self, image_bytes: bytes, top_k: int = 5) -> List[Tuple[str, float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(self._predict_sync, image_bytes, top_k))

    async def predict_best(self, image_bytes: bytes, min_confidence: float = 0.85) -> Optional[Tuple[str, float]]:
        results = await self.predict(image_bytes, top_k=1)
        if results and results[0][1] >= min_confidence:
            return results[0]
        return None
