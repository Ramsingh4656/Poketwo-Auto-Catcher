from __future__ import annotations

import io
import importlib
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bot'))


def main() -> None:
    predictor_module = importlib.import_module('predictor')
    predictor = predictor_module.PokemonPredictor()
    image = Image.new('RGB', (224, 224), (48, 48, 48))
    buf = io.BytesIO()
    image.save(buf, format='JPEG')
    result = predictor._predict_sync(buf.getvalue(), top_k=5)
    assert predictor.loaded and predictor.backend == 'onnxruntime', (predictor.loaded, predictor.backend)
    assert len(result) == 5 and all(0.0 <= score <= 1.0 for _, score in result), result
    importlib.import_module('bot')
    print({'backend': predictor.backend, 'classes': len(predictor.index_to_pokemon), 'top_k': result, 'bot_import': 'ok'})


if __name__ == '__main__':
    main()
