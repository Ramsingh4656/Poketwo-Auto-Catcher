from __future__ import annotations
import json, zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
model=root/'model/pokemon_cnn.keras'
index=root/'model/index_to_pokemon.json'
print('model_exists', model.exists(), 'bytes', model.stat().st_size if model.exists() else None)
labels=json.loads(index.read_text()) if index.exists() else {}
print('index_labels', len(labels), 'first', list(labels.items())[:5], 'last', list(labels.items())[-5:])
if model.exists():
    try:
        with zipfile.ZipFile(model) as z:
            print('keras_files', z.namelist())
            cfg=json.loads(z.read('config.json'))
            model_cfg=cfg.get('config',{})
            print('keras_module', cfg.get('module'), 'class_name', cfg.get('class_name'))
            print('compile_config', cfg.get('compile_config'))
            layers=model_cfg.get('layers', [])
            print('layer_sequence', [layer.get('class_name') for layer in layers])
            for layer in layers:
                if layer.get('class_name') in ('InputLayer','Dense','Flatten','Rescaling'):
                    print('layer', layer.get('class_name'), layer.get('config'))
    except Exception as exc: print('keras_inspection_error', repr(exc))
