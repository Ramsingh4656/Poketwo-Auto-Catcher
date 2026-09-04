#!/usr/bin/env python3
"""Hybrid Pokétwo image detector.

This module is intentionally transport-agnostic: it accepts an image URL and
an optional hint string, and returns structured predictions. It does not log in
to Discord, read messages, or send catch commands.
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import imagehash
import numpy as np
import requests
from PIL import Image, ImageOps


DEFAULT_HASH_DISTANCE = 4
DEFAULT_TOP_K = 5
DEFAULT_IMAGE_SIZE = (224, 224)


def _ascii_alnum(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch.lower() for ch in normalized if ch.isalnum())


def _hint_pattern(hint_string: str) -> str:
    """Convert `_ i _ a _ h u` or `p i k a c h u` into a full-match regex."""
    compact = "".join(ch.lower() for ch in hint_string if ch.isalnum() or ch in "_?")
    if not compact:
        return r"(?!)"
    pieces = []
    for char in compact:
        if char in "_?":
            pieces.append(r".")
        else:
            pieces.append(re.escape(_ascii_alnum(char)))
    return "^" + "".join(pieces) + "$"


def resolve_hint(hint_string: str, label_universe: Iterable[str]) -> str | None:
    """Return a name only when exactly one complete-universe label matches.

    The resolver intentionally does not depend on CNN rank. It accepts a
    conservative result only when the hint matches exactly one label in the
    supplied model mapping; ambiguous or unknown hints return ``None``.
    """
    pattern = re.compile(_hint_pattern(hint_string))
    candidates: list[str] = []
    for name in label_universe:
        if not isinstance(name, str):
            continue
        if pattern.fullmatch(_ascii_alnum(name)) and name not in candidates:
            candidates.append(name)
    return candidates[0] if len(candidates) == 1 else None


def _names_from_db_value(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [name for name in value if isinstance(name, str)]


class HybridDetector:
    def __init__(
        self,
        model_path: str | Path = "pokemon_classifier_augmented.tflite",
        mapping_path: str | Path = "index_to_pokemon.json",
        hash_db_path: str | Path = "phash_db.json",
        hash_distance: int = DEFAULT_HASH_DISTANCE,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        request_timeout: float = 20.0,
    ):
        import tensorflow as tf

        self.hash_distance = hash_distance
        self.image_size = image_size
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "poketwo-local-hybrid-detector/1.0"})
        self.hash_db: dict[str, str | list[str]] = json.loads(Path(hash_db_path).read_text(encoding="utf-8"))
        self.mapping: dict[str, str] = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        self.hashes = [(imagehash.hex_to_hash(key), value) for key, value in self.hash_db.items()]
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_info = self.interpreter.get_input_details()[0]
        self.output_info = self.interpreter.get_output_details()[0]

    @staticmethod
    def _open(data: bytes) -> Image.Image:
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")

    def _download(self, image_url: str) -> bytes:
        response = self.session.get(image_url, timeout=self.request_timeout)
        response.raise_for_status()
        if not response.content:
            raise ValueError("Downloaded image is empty")
        return response.content

    def _hash_lookup(self, image: Image.Image) -> dict[str, Any] | None:
        query = imagehash.phash(image)
        exact = self.hash_db.get(str(query))
        if exact is not None:
            names = _names_from_db_value(exact)
            if len(names) == 1:
                return {"name": names[0], "distance": 0, "method": "phash_exact"}

        nearest_distance: int | None = None
        nearest_names: set[str] = set()
        for stored_hash, value in self.hashes:
            distance = query - stored_hash
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_names = set(_names_from_db_value(value))
            elif distance == nearest_distance:
                nearest_names.update(_names_from_db_value(value))
        if nearest_distance is not None and nearest_distance <= self.hash_distance and len(nearest_names) == 1:
            return {
                "name": next(iter(nearest_names)),
                "distance": int(nearest_distance),
                "method": "phash_near",
            }
        return None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        canvas = Image.new("RGB", self.image_size, (0, 0, 0))
        contained = ImageOps.contain(image, self.image_size, method=Image.Resampling.LANCZOS)
        canvas.paste(contained, ((self.image_size[0] - contained.width) // 2, (self.image_size[1] - contained.height) // 2))
        array = np.asarray(canvas, dtype=np.float32)[None, ...]
        dtype = self.input_info["dtype"]
        if dtype == np.float32:
            return array
        scale, zero = self.input_info["quantization"]
        if not scale:
            return array.astype(dtype)
        info = np.iinfo(dtype)
        return np.round(array / scale + zero).clip(info.min, info.max).astype(dtype)

    def _cnn_top5(self, image: Image.Image) -> list[dict[str, Any]]:
        tensor = self._preprocess(image)
        self.interpreter.set_tensor(self.input_info["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_info["index"])
        scale, zero = self.output_info["quantization"]
        probabilities = raw.astype(np.float32)
        if scale:
            probabilities = (probabilities - zero) * scale
        probabilities = probabilities[0]
        # The exported classifier has a softmax head. Only apply softmax when a
        # backend returns logits instead of probabilities.
        if np.any(probabilities < 0.0) or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-3):
            probabilities = np.exp(probabilities - probabilities.max())
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = probabilities / probabilities.sum()
        indices = np.argsort(probabilities)[::-1][:DEFAULT_TOP_K]
        return [
            {
                "index": int(index),
                "name": self.mapping.get(str(int(index)), str(int(index))),
                "confidence": float(probabilities[index]),
            }
            for index in indices
        ]

    def detect(
        self,
        image_url: str,
        hint_string: str | None = None,
        min_confidence: float = 0.70,
        min_margin: float = 0.15,
    ) -> dict[str, Any]:
        """Run pHash, then TFLite, then optional hint cross-reference."""
        data = self._download(image_url)
        image = self._open(data)
        hash_result = self._hash_lookup(image)
        if hash_result is not None:
            return {"name": hash_result["name"], "accepted": True, **hash_result, "top_5": []}

        top_5 = self._cnn_top5(image)
        # Search the complete experimental TFLite mapping, not only CNN top-5.
        hinted = resolve_hint(hint_string, self.mapping.values()) if hint_string else None
        if hinted is not None:
            return {
                "name": hinted,
                "accepted": True,
                "method": "hint_plus_cnn",
                "hint": hint_string,
                "top_5": top_5,
            }

        top1 = top_5[0]
        top2 = top_5[1] if len(top_5) > 1 else {"confidence": 0.0}
        margin = float(top1["confidence"] - top2["confidence"])
        accepted = top1["confidence"] >= min_confidence and margin >= min_margin
        return {
            "name": top1["name"] if accepted else None,
            "accepted": accepted,
            "method": "cnn",
            "confidence": top1["confidence"],
            "margin": margin,
            "hint": hint_string,
            "top_5": top_5,
        }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_url")
    parser.add_argument("--hint", default=None)
    parser.add_argument("--model", default="pokemon_classifier_augmented.tflite")
    parser.add_argument("--mapping", default="index_to_pokemon.json")
    parser.add_argument("--hash-db", default="phash_db.json")
    parser.add_argument("--hash-distance", type=int, default=DEFAULT_HASH_DISTANCE)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--min-margin", type=float, default=0.15)
    args = parser.parse_args()
    detector = HybridDetector(args.model, args.mapping, args.hash_db, args.hash_distance)
    print(json.dumps(detector.detect(args.image_url, args.hint, args.min_confidence, args.min_margin), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
