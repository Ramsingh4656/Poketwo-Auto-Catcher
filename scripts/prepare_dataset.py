from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

AUGMENT_MARKERS = ('rot', 'flip', 'gray', 'contrast', 'bright', 'blur', 'sharp', 'persp', 'noise', 'jpeg', 'crop', 'scale')


def source_group(path: Path) -> str:
    match = re.match(r'(\d+)', path.stem)
    return match.group(1) if match else path.stem


def dhash(im: Image.Image, size: int = 16) -> int:
    gray = im.convert('L').resize((size + 1, size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def is_original(path: Path) -> bool:
    stem = path.stem.lower()
    if stem.endswith('_none'):
        return True
    return not any(f'_{marker}' in stem for marker in AUGMENT_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--captured-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--captured-path-prefix', default='data/raw')
    parser.add_argument('--official-path-prefix', default='data/raw/poketwo-data')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for split in ('train', 'val', 'test'):
        for path in sorted((args.captured_root / split).glob('*/*')):
            if not path.is_file() or (split == 'train' and not is_original(path)):
                continue
            label = path.parent.name
            try:
                raw = path.read_bytes()
                with Image.open(path) as im:
                    im.load()
                    candidates.append({
                        'path': str(Path(args.captured_path_prefix) / path.resolve().relative_to(args.captured_root.parent.parent.resolve())),
                        'relative_path': str(path.resolve().relative_to(args.captured_root.parent.parent.resolve())),
                        'split_original': split,
                        'label': label,
                        'source_group': source_group(path),
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'dhash': dhash(im),
                        'width': im.width,
                        'height': im.height,
                        'mode': im.mode,
                    })
            except Exception:
                pass

    by_hash = defaultdict(list)
    by_source = defaultdict(list)
    for rec in candidates:
        by_hash[rec['sha256']].append(rec)
        by_source[rec['source_group']].append(rec)
    drop_ids = set()
    reasons = defaultdict(list)
    for group in by_hash.values():
        if len(group) > 1:
            for rec in group:
                drop_ids.add(id(rec)); reasons[rec['path']].append('exact_duplicate')

    # Remove source groups that cross the supplied splits. This prevents a
    # captured original and a derivative/duplicate from crossing boundaries.
    for group, items in by_source.items():
        if len({x['split_original'] for x in items}) > 1:
            for rec in items:
                drop_ids.add(id(rec)); reasons[rec['path']].append('source_group_cross_split')

    # Conservative cross-split perceptual duplicate check among remaining data.
    for i, left in enumerate(candidates):
        if id(left) in drop_ids:
            continue
        for right in candidates[i + 1:]:
            if id(right) in drop_ids or left['split_original'] == right['split_original']:
                continue
            if left['width'] == right['width'] and left['height'] == right['height'] and hamming(left['dhash'], right['dhash']) <= 8:
                drop_ids.add(id(left)); drop_ids.add(id(right))
                reasons[left['path']].append('cross_split_near_duplicate')
                reasons[right['path']].append('cross_split_near_duplicate')

    cleaned = []
    for rec in candidates:
        if id(rec) in drop_ids:
            continue
        cleaned.append({k: v for k, v in rec.items() if k != 'dhash'})

    manifest_path = args.output_dir / 'captured_manifest.csv'
    fields = ['path', 'relative_path', 'split_original', 'label', 'source_group', 'sha256', 'width', 'height', 'mode']
    with manifest_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(cleaned)

    reason_path = args.output_dir / 'dropped_samples.csv'
    with reason_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['path', 'reasons'])
        for path, vals in sorted(reasons.items()): writer.writerow([path, '|'.join(sorted(set(vals)))])

    # Official manifest uses catchable Pokétwo image assets, while preserving
    # source metadata and visual form identity for the hybrid recognizer.
    with (args.official_root / 'csv' / 'pokemon.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = {int(r['id']): r for r in csv.DictReader(f)}
    official_fields = ['path', 'pokemon_id', 'pokemon_name', 'slug', 'form', 'shiny', 'enabled', 'catchable', 'sha256', 'width', 'height', 'mode']
    official_rows = []
    for shiny in (False, True):
        rel = 'shiny' if shiny else 'images'
        for path in sorted((args.official_root / rel).glob('*.png')):
            try:
                raw = path.read_bytes()
                with Image.open(path) as im: im.load()
                r = rows.get(int(path.stem), {})
                if r.get('catchable') != '1' or r.get('enabled') != '1': continue
                official_rows.append({'path': str(Path(args.official_path_prefix) / rel / path.name), 'pokemon_id': int(path.stem), 'pokemon_name': r.get('name.en', ''), 'slug': r.get('slug', ''), 'form': r.get('is_form', ''), 'shiny': int(shiny), 'enabled': r.get('enabled', ''), 'catchable': r.get('catchable', ''), 'sha256': hashlib.sha256(raw).hexdigest(), 'width': im.width, 'height': im.height, 'mode': im.mode})
            except Exception:
                pass
    official_manifest = args.output_dir / 'official_manifest.csv'
    with official_manifest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=official_fields); writer.writeheader(); writer.writerows(official_rows)

    from collections import Counter
    print('captured_candidates', len(candidates))
    print('captured_clean', len(cleaned))
    print('captured_dropped', len(candidates) - len(cleaned))
    print('drop_reasons', dict(Counter(reason for vals in reasons.values() for reason in set(vals))))
    print('clean_by_split', dict(Counter(x['split_original'] for x in cleaned)))
    print('clean_classes', len({x['label'] for x in cleaned}))
    print('official_manifest', len(official_rows))

if __name__ == '__main__':
    main()
