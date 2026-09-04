#!/usr/bin/env python3
"""Build a pHash lookup database from the public Pokétwo data repository.

The output is a JSON object mapping a hexadecimal pHash to either one exact
Pokémon name or a list of names when two official assets collide at the chosen
hash size. Multiple alpha/background variants are indexed because a Discord
embed may render transparent artwork on a different background.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import imagehash
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CSV_URL = "https://raw.githubusercontent.com/poketwo/data/master/csv/pokemon.csv"
IMAGE_URL = "https://raw.githubusercontent.com/poketwo/data/master/images/{id}.png"
DARK = (49, 51, 56, 255)       # #313338
LIGHT = (255, 255, 255, 255)   # #FFFFFF


def session_with_retries() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16))
    session.headers.update({"User-Agent": "poketwo-local-phash-builder/1.0"})
    return session


def is_enabled(row: dict[str, str]) -> bool:
    value = row.get("enabled", "true").strip().lower()
    return value not in {"false", "0", "no", "disabled"}


def label_from_row(row: dict[str, str]) -> str:
    for key in ("name.en", "name", "name.en2", "slug"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"No English label found in CSV row: {row}")


def read_rows(session: requests.Session) -> list[dict[str, str]]:
    response = session.get(CSV_URL, timeout=30)
    response.raise_for_status()
    return list(csv.DictReader(io.StringIO(response.text)))


def alpha_variants(image: Image.Image) -> list[Image.Image]:
    """Return raw RGB plus deterministic dark/light alpha composites."""
    rgba = image.convert("RGBA")
    raw = rgba.convert("RGB")
    dark = Image.new("RGBA", rgba.size, DARK)
    dark.alpha_composite(rgba)
    light = Image.new("RGBA", rgba.size, LIGHT)
    light.alpha_composite(rgba)
    return [raw, dark.convert("RGB"), light.convert("RGB")]


def add_hash(db: dict[str, str | list[str]], hash_value: str, name: str) -> None:
    current = db.get(hash_value)
    if current is None:
        db[hash_value] = name
    elif isinstance(current, str) and current != name:
        db[hash_value] = sorted({current, name})
    elif isinstance(current, list) and name not in current:
        current.append(name)
        current.sort()


def build_database(output: Path, include_disabled: bool = False, hash_size: int = 8, delay: float = 0.0, limit: int | None = None) -> dict[str, Any]:
    session = session_with_retries()
    rows = read_rows(session)
    db: dict[str, str | list[str]] = {}
    processed = skipped = failed = 0
    eligible_seen = 0
    failures: list[dict[str, str]] = []

    for row in rows:
        if not include_disabled and not is_enabled(row):
            skipped += 1
            continue
        raw_id = (row.get("id") or "").strip()
        if not raw_id.isdigit():
            skipped += 1
            continue
        if limit is not None and eligible_seen >= limit:
            break
        eligible_seen += 1
        name = label_from_row(row)
        url = IMAGE_URL.format(id=raw_id)
        try:
            response = session.get(url, timeout=30)
            if response.status_code != 200 or not response.content:
                raise RuntimeError(f"HTTP {response.status_code}")
            with Image.open(io.BytesIO(response.content)) as image:
                image.load()
                for variant in alpha_variants(image):
                    add_hash(db, str(imagehash.phash(variant, hash_size=hash_size)), name)
            processed += 1
        except Exception as exc:  # keep the builder going when one asset is unavailable
            failed += 1
            failures.append({"id": raw_id, "name": name, "error": str(exc)})
        if delay:
            time.sleep(delay)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(db, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "csv_url": CSV_URL,
        "image_url_pattern": IMAGE_URL,
        "processed_rows": processed,
        "skipped_rows": skipped,
        "failed_rows": failed,
        "limit": limit,
        "hash_entries": len(db),
        "hash_size": hash_size,
        "variants_per_image": 3,
        "backgrounds": {"dark": "#313338", "light": "#FFFFFF"},
        "failures": failures,
    }
    output.with_name(output.stem + ".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("phash_db.json"))
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--hash-size", type=int, default=8, help="imagehash pHash size; 8 gives a 64-bit hash")
    parser.add_argument("--delay", type=float, default=0.0, help="optional delay between downloads")
    parser.add_argument("--limit", type=int, default=None, help="optional number of eligible rows; useful for smoke tests")
    args = parser.parse_args()
    if args.hash_size < 4:
        parser.error("--hash-size must be at least 4")
    try:
        report = build_database(args.output, args.include_disabled, args.hash_size, args.delay, args.limit)
    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["failed_rows"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
