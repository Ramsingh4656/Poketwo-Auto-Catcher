from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
SEED = 20260820


def dhash(im: Image.Image, size: int = 16) -> int:
    gray = im.convert('L').resize((size + 1, size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.ravel(): value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int: return (a ^ b).bit_count()


def relpath(path: Path, root: Path) -> str:
    # Preserve repository-local symlink paths in the manifest; resolving the
    # symlink would leak the external workspace path into the portable package.
    return path.absolute().relative_to(root.absolute()).as_posix()


def inspect(path: Path):
    raw = path.read_bytes()
    with Image.open(path) as im:
        im.load()
        return {
            'sha256': hashlib.sha256(raw).hexdigest(), 'dhash': dhash(im),
            'width': im.width, 'height': im.height, 'mode': im.mode,
        }


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, required=True)
    ap.add_argument('--official-root', type=Path, required=True)
    ap.add_argument('--kaggle-root', type=Path, required=True)
    ap.add_argument('--pokemon-csv', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.pokemon_csv.open(encoding='utf-8-sig', newline='') as f:
        metadata = {r['slug'].lower(): r for r in csv.DictReader(f) if r.get('enabled') == '1' and r.get('catchable') == '1'}
    labels = sorted(metadata)
    records: list[dict] = []
    errors = []

    # Every official catchable record contributes real Pokétwo reference assets.
    for slug in labels:
        r = metadata[slug]; pid = r['id']
        for shiny in (0, 1):
            path = args.official_root / ('shiny' if shiny else 'images') / f'{pid}.png'
            try: info = inspect(path)
            except Exception as e:
                errors.append({'path': str(path), 'error': repr(e)}); continue
            records.append({
                'path': relpath(path, args.repo_root), 'label': slug, 'class_id': int(pid),
                'split': 'train', 'split_original': 'train', 'source_type': 'official_reference',
                'source_group': f'official_{pid}', 'shiny': shiny, 'source_path': str(path), **info,
            })

    # The Apache source provides independent Discord-like captured renders. Filenames 1..23
    # are deterministic source observations; the split is by source file and then checked for
    # cross-split exact/perceptual contamination.
    for slug in labels:
        folder = args.kaggle_root / slug
        if not folder.is_dir(): continue
        imgs = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}], key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
        n = len(imgs)
        for i, path in enumerate(imgs):
            split = 'train' if i < max(1, int(n * 0.70)) else ('val' if i < max(2, int(n * 0.85)) else 'test')
            try: info = inspect(path)
            except Exception as e:
                errors.append({'path': str(path), 'error': repr(e)}); continue
            records.append({
                'path': relpath(path, args.repo_root), 'label': slug, 'class_id': int(metadata[slug]['id']),
                'split': split, 'split_original': split, 'source_type': 'kaggle_real_render',
                'source_group': f'kaggle_{slug}', 'shiny': 0, 'source_path': str(path), **info,
            })

    # Drop exact duplicates across any split; then conservatively drop perceptual duplicates
    # only when they cross validation/test boundaries. Official references are not eligible
    # for the independent real-render test set and remain training-only.
    by_hash = defaultdict(list)
    for rec in records: by_hash[rec['sha256']].append(rec)
    dropped = []
    drop_ids = set()
    for vals in by_hash.values():
        if len(vals) > 1 and len({x['split'] for x in vals}) > 1:
            for x in vals: drop_ids.add(id(x)); dropped.append({'path': x['path'], 'reason': 'exact_duplicate_cross_split'})
    remain = [x for x in records if id(x) not in drop_ids]
    real = [x for x in remain if x['source_type'] == 'kaggle_real_render']
    real_by_label = defaultdict(list)
    for item in real: real_by_label[item['label']].append(item)
    for items in real_by_label.values():
        for i, left in enumerate(items):
            if left['split'] == 'train': continue
            for right in items[i + 1:]:
                if left['split'] == right['split']: continue
                if left['width'] == right['width'] and left['height'] == right['height'] and hamming(left['dhash'], right['dhash']) <= 4:
                    drop_ids.add(id(left)); drop_ids.add(id(right))
                    dropped.extend([{'path': left['path'], 'reason': 'cross_split_perceptual_duplicate'}, {'path': right['path'], 'reason': 'cross_split_perceptual_duplicate'}])
    clean = [x for x in remain if id(x) not in drop_ids]
    for x in clean: x.pop('dhash', None); x.pop('source_path', None)
    fields = ['path','label','class_id','split','split_original','source_type','source_group','shiny','sha256','width','height','mode']
    write_csv(args.output_dir / 'full_manifest.csv', clean, fields)
    write_csv(args.output_dir / 'full_dropped_samples.csv', dropped, ['path','reason'])
    coverage = {}
    for slug in labels:
        vals = [x for x in clean if x['label'] == slug]
        coverage[slug] = {
            'metadata': metadata[slug], 'total': len(vals),
            'official_reference_train': sum(x['source_type']=='official_reference' for x in vals),
            'kaggle_real_render': sum(x['source_type']=='kaggle_real_render' for x in vals),
            'real_train': sum(x['source_type']=='kaggle_real_render' and x['split']=='train' for x in vals),
            'real_val': sum(x['source_type']=='kaggle_real_render' and x['split']=='val' for x in vals),
            'real_test': sum(x['source_type']=='kaggle_real_render' and x['split']=='test' for x in vals),
        }
    summary = {
        'classes': len(labels), 'records': len(clean), 'errors': errors, 'dropped': len(dropped),
        'by_split': dict(Counter(x['split'] for x in clean)),
        'by_source': dict(Counter(x['source_type'] for x in clean)),
        'real_render_test_classes': sum(v['real_test'] > 0 for v in coverage.values()),
        'reference_only_classes': sum(v['kaggle_real_render'] == 0 for v in coverage.values()),
        'coverage': coverage,
    }
    (args.output_dir / 'full_coverage_manifest_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['classes','records','dropped','by_split','by_source','real_render_test_classes','reference_only_classes','errors']}, indent=2))


if __name__ == '__main__': main()
