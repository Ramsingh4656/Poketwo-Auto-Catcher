from pathlib import Path
from bot.predictor import PokemonPredictor

sample = next(Path('data/raw/kaggle/extracted/test').glob('*/*.jpg'))
predictor = PokemonPredictor()
results = predictor._predict_sync(sample.read_bytes(), top_k=5)
print({'sample': str(sample), 'loaded': predictor.loaded, 'backend': predictor.backend, 'results': results})
assert predictor.loaded
assert predictor.backend == 'onnxruntime'
assert len(results) == 5
assert all(0.0 <= confidence <= 1.0 for _, confidence in results)
