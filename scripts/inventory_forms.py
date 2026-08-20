from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def truth(value: str) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    with args.csv.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    catchable = [r for r in rows if truth(r.get('enabled', '')) and truth(r.get('catchable', ''))]
    enabled = [r for r in rows if truth(r.get('enabled', ''))]
    def has_form(r): return bool(r.get('is_form', '').strip())
    def has_mega(r): return any(r.get(k, '').strip() for k in ('evo.mega', 'evo.mega_x', 'evo.mega_y'))
    category = Counter()
    for r in catchable:
        if has_mega(r): category['mega-capable/base-or-mega-linked'] += 1
        if has_form(r): category['is_form'] += 1
        if truth(r.get('has_gender_differences', '')): category['gender-difference-capable'] += 1
        if 'alolan' in r.get('slug', '').lower() or 'galarian' in r.get('slug', '').lower() or 'hisuian' in r.get('slug', '').lower() or 'paldean' in r.get('slug', '').lower(): category['regional-slug'] += 1
    result = {
        'source_csv': str(args.csv),
        'all_rows': len(rows),
        'enabled_rows': len(enabled),
        'catchable_rows': len(catchable),
        'catchable_unique_slugs': len({r.get('slug', '') for r in catchable}),
        'catchable_unique_names': len({r.get('name.en', '') for r in catchable}),
        'category_counts_nonexclusive': dict(category),
        'catchable_rows': [
            {k: r.get(k, '') for k in ('id','dex_number','region','slug','enabled','catchable','has_gender_differences','name.en','evo.mega','evo.mega_x','evo.mega_y','is_form','form_item')}
            for r in catchable
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'catchable_rows'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
