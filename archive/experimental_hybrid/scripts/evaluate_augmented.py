from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path('/home/ubuntu/poketwo_autotrain')
DATASET = ROOT / 'dataset'
ART = ROOT / 'augmented_artifacts'
IMG = 224
BATCH = 64
mapping = json.loads((ROOT / 'class_indices.json').read_text(encoding='utf-8'))
class_names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
model = tf.keras.models.load_model(ART / 'pokemon_classifier_augmented.keras', compile=False)

paths, labels = [], []
for name in class_names:
    for path in sorted((DATASET / 'val' / name).glob('*.png')):
        paths.append(path)
        labels.append(mapping[name])


def render(path: Path, bg: tuple[int, int, int]) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert('RGBA')
        base = Image.new('RGBA', im.size, bg + (255,))
        base.alpha_composite(im)
        base = base.convert('RGB')
        contained = ImageOps.contain(base, (IMG, IMG), method=Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (IMG, IMG), bg)
        canvas.paste(contained, ((IMG - contained.width) // 2, (IMG - contained.height) // 2))
        return np.asarray(canvas, dtype=np.float32)


def score(bg: tuple[int, int, int]):
    pred, true = [], []
    for start in range(0, len(paths), BATCH):
        batch_paths = paths[start:start + BATCH]
        x = np.stack([render(path, bg) for path in batch_paths], axis=0)
        p = model.predict(x, verbose=0)
        pred.append(p)
        true.extend(labels[start:start + BATCH])
    p = np.concatenate(pred, axis=0)
    y = np.asarray(true)
    order = np.argsort(p, axis=1)[:, ::-1]
    top1 = float(np.mean(order[:, 0] == y))
    top5 = float(np.mean(np.any(order[:, :5] == y[:, None], axis=1)))
    conf = p[np.arange(len(y)), order[:, 0]]
    margin = conf - p[np.arange(len(y)), order[:, 1]]
    return {'samples': len(y), 'top1_accuracy': top1, 'top5_accuracy': top5, 'mean_confidence': float(conf.mean()), 'mean_margin': float(margin.mean())}

summary = {
    'classes': len(class_names),
    'validation_images': len(paths),
    'dark_mode_313338': score((49, 51, 56)),
    'light_mode_ffffff': score((255, 255, 255)),
}
summary['combined'] = {
    'samples': summary['dark_mode_313338']['samples'] + summary['light_mode_ffffff']['samples'],
    'top1_accuracy': (summary['dark_mode_313338']['top1_accuracy'] + summary['light_mode_ffffff']['top1_accuracy']) / 2,
    'top5_accuracy': (summary['dark_mode_313338']['top5_accuracy'] + summary['light_mode_ffffff']['top5_accuracy']) / 2,
}
summary['warning'] = 'This is still a repository-asset validation benchmark, not a real Pokétwo Discord screenshot benchmark.'
(ART / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
