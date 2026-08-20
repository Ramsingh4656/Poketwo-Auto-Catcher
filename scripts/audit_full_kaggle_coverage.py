from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--pokemon-csv', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.pokemon_csv.open(encoding='utf-8')))
    catch = [r for r in rows if r.get('enabled') == '1' and r.get('catchable') == '1']
    by_slug = {r['slug'].lower(): r for r in catch}
    folders = sorted([p for p in args.data_root.iterdir() if p.is_dir()])
    folder_names = {p.name.lower(): p for p in folders}
    matched = sorted(set(folder_names) & set(by_slug))
    missing = sorted(set(by_slug) - set(folder_names))
    unmatched = sorted(set(folder_names) - set(by_slug))
    per_class = {}
    bad_files = []
    hashes = Counter()
    for slug in matched:
        p = folder_names[slug]
        imgs = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}])
        good = 0
        dims = Counter()
        for img in imgs:
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    dims[f'{im.width}x{im.height}'] += 1
                good += 1
                h = hashlib.sha256(img.read_bytes()).hexdigest()
                hashes[h] += 1
            except Exception as e:
                bad_files.append({'path': str(img), 'error': repr(e)})
        per_class[slug] = {'count': len(imgs), 'valid': good, 'dimensions': dict(dims), 'metadata': by_slug[slug]}
    duplicate_hashes = {h: n for h, n in hashes.items() if n > 1}
    result = {
        'source': 'https://www.kaggle.com/datasets/dhruv2015/poketwo-datset-3',
        'dataset_root': str(args.data_root),
        'metadata_snapshot': str(args.pokemon_csv),
        'official_catchable_records': len(catch),
        'dataset_folder_count': len(folders),
        'matched_official_catchable_slugs': len(matched),
        'missing_official_catchable_slugs': len(missing),
        'unmatched_dataset_folders': len(unmatched),
        'matched_image_files': sum(v['count'] for v in per_class.values()),
        'valid_matched_image_files': sum(v['valid'] for v in per_class.values()),
        'invalid_files': len(bad_files),
        'exact_duplicate_hash_groups': len(duplicate_hashes),
        'missing_slugs': missing,
        'unmatched_folders': unmatched,
        'bad_files': bad_files,
        'per_class': per_class,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['official_catchable_records','dataset_folder_count','matched_official_catchable_slugs','missing_official_catchable_slugs','unmatched_dataset_folders','matched_image_files','valid_matched_image_files','invalid_files','exact_duplicate_hash_groups']}, indent=2))


if __name__ == '__main__':
    main()
