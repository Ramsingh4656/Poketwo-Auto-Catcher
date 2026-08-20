from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dhash(image: Image.Image, size: int = 16) -> int:
    gray = image.convert('L').resize((size + 1, size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def source_group(path: Path) -> str:
    # Captured dataset augmentation filenames use timestamp_suffix; use the
    # timestamp as the immutable source identity for split-leak checks.
    return re.split(r'_(?:rot|flip|gray|contrast|bright|blur|sharp|persp|noise|jpeg|crop|scale)', path.stem, maxsplit=1)[0]


def audit_captured(root: Path) -> dict:
    records = []
    errors = []
    exact = defaultdict(list)
    by_source = defaultdict(list)
    split_names = ('train', 'val', 'test')
    class_counts = {split: Counter() for split in split_names}
    for split in split_names:
        for path in sorted((root / split).glob('*/*')):
            if not path.is_file():
                continue
            label = path.parent.name
            try:
                raw = path.read_bytes()
                with Image.open(path) as im:
                    im.load()
                    rec = {
                        'path': str(path),
                        'split': split,
                        'label': label,
                        'source_group': source_group(path),
                        'sha256': sha256_bytes(raw),
                        'dhash': dhash(im),
                        'width': im.width,
                        'height': im.height,
                        'mode': im.mode,
                        'format': im.format,
                        'bytes': len(raw),
                    }
                records.append(rec)
                exact[rec['sha256']].append(rec['path'])
                by_source[rec['source_group']].append(rec)
                class_counts[split][label] += 1
            except Exception as exc:
                errors.append({'path': str(path), 'error': repr(exc)})

    split_source_overlap = {}
    for group, items in by_source.items():
        splits = sorted({x['split'] for x in items})
        if len(splits) > 1:
            split_source_overlap[group] = splits

    exact_groups = [paths for paths in exact.values() if len(paths) > 1]
    # Perceptual duplicate check is intentionally conservative: a low dHash
    # distance indicates a possible duplicate, not a label correction.
    cross_split_near = []
    buckets = defaultdict(list)
    for rec in records:
        key = (rec['width'], rec['height'], rec['dhash'] >> 8)
        buckets[key].append(rec)
    for candidates in buckets.values():
        for i, left in enumerate(candidates):
            for right in candidates[i + 1:]:
                if left['split'] != right['split'] and hamming(left['dhash'], right['dhash']) <= 8:
                    cross_split_near.append({
                        'left': left['path'], 'right': right['path'],
                        'distance': hamming(left['dhash'], right['dhash']),
                    })

    all_classes = sorted(set().union(*(set(c) for c in class_counts.values())))
    report = {
        'root': str(root),
        'record_count': len(records),
        'errors': errors,
        'class_count': len(all_classes),
        'classes': all_classes,
        'class_counts': {split: dict(sorted(counter.items())) for split, counter in class_counts.items()},
        'missing_by_split': {split: sorted(set(all_classes) - set(class_counts[split])) for split in split_names},
        'exact_duplicate_groups': exact_groups,
        'exact_duplicate_file_count': sum(len(x) for x in exact_groups),
        'split_source_overlap': split_source_overlap,
        'cross_split_near_duplicate_candidates': cross_split_near,
        'dimensions': {f'{w}x{h}': n for (w, h), n in Counter((r['width'], r['height']) for r in records).items()},
        'modes': dict(Counter(r['mode'] for r in records)),
        'formats': dict(Counter(r['format'] for r in records)),
        'manifest': records,
    }
    return report


def audit_official(root: Path) -> dict:
    csv_path = root / 'csv' / 'pokemon.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    by_id = {int(r['id']): r for r in rows}
    output = {'metadata_rows': len(rows), 'directories': {}}
    for rel in ('images', 'shiny'):
        records = []
        errors = []
        exact = defaultdict(list)
        for path in sorted((root / rel).glob('*.png')):
            try:
                raw = path.read_bytes()
                with Image.open(path) as im:
                    im.load()
                    rec = {
                        'path': str(path),
                        'pokemon_id': int(path.stem),
                        'pokemon_name': by_id.get(int(path.stem), {}).get('name.en'),
                        'slug': by_id.get(int(path.stem), {}).get('slug'),
                        'enabled': by_id.get(int(path.stem), {}).get('enabled'),
                        'catchable': by_id.get(int(path.stem), {}).get('catchable'),
                        'form': by_id.get(int(path.stem), {}).get('is_form'),
                        'sha256': sha256_bytes(raw),
                        'width': im.width,
                        'height': im.height,
                        'mode': im.mode,
                        'format': im.format,
                        'bytes': len(raw),
                    }
                records.append(rec)
                exact[rec['sha256']].append(rec['path'])
            except Exception as exc:
                errors.append({'path': str(path), 'error': repr(exc)})
        groups = [x for x in exact.values() if len(x) > 1]
        output['directories'][rel] = {
            'record_count': len(records),
            'errors': errors,
            'duplicate_groups': groups,
            'duplicate_file_count': sum(len(x) for x in groups),
            'dimensions': {f'{w}x{h}': n for (w, h), n in Counter((r['width'], r['height']) for r in records).items()},
            'modes': dict(Counter(r['mode'] for r in records)),
            'catchable_records': sum(r['catchable'] == '1' for r in records),
            'manifest': records,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--captured-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {
        'captured': audit_captured(args.captured_root),
        'official': audit_official(args.official_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = {
        'captured_records': report['captured']['record_count'],
        'captured_errors': len(report['captured']['errors']),
        'captured_classes': report['captured']['class_count'],
        'exact_duplicate_files': report['captured']['exact_duplicate_file_count'],
        'split_source_overlap_groups': len(report['captured']['split_source_overlap']),
        'cross_split_near_duplicate_candidates': len(report['captured']['cross_split_near_duplicate_candidates']),
        'official_images': report['official']['directories']['images']['record_count'],
        'official_image_errors': len(report['official']['directories']['images']['errors']),
        'official_shiny': report['official']['directories']['shiny']['record_count'],
    }
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
