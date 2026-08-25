#!/usr/bin/env python3
"""Inventory a Pokétwo source archive without extracting the raw dataset.

This tool is intentionally read-only. It writes only a JSON provenance report and
never copies images into the repository. Use it with the public Kaggle archives
referenced by the Stage 4 documentation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


def normalize_label(value: str) -> str:
    """Normalize archive folder names and manifest labels for comparison only."""
    value = value.lower().replace('♀', '-f').replace('♂', '-m')
    value = value.replace('%', ' percent ')
    return re.sub(r'[^a-z0-9]+', '-', value).strip('-')


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_info(data: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(data)) as image:
        image.load()
        return {
            'format': image.format,
            'mode': image.mode,
            'width': image.width,
            'height': image.height,
        }


def load_real_manifest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = csv.DictReader(handle)
        result = {}
        for row in rows:
            if row.get('source_type') != 'kaggle_real_render':
                continue
            result[(normalize_label(row['label']), Path(row['path']).name)] = row
        return result


def inspect_archive(archive: Path, manifest_path: Path | None, sample_count: int) -> dict[str, Any]:
    class_members: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    invalid_layout: list[str] = []
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            suffix = Path(member.filename).suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                continue
            parts = Path(member.filename).parts
            if len(parts) < 2:
                invalid_layout.append(member.filename)
                continue
            class_members[parts[0]].append(member)

        class_counts = Counter({name: len(items) for name, items in class_members.items()})
        sample_members = []
        for class_name in sorted(class_members):
            members = sorted(class_members[class_name], key=lambda item: item.filename)
            sample_members.extend(members[:1])
            if len(sample_members) >= sample_count:
                break
        samples = []
        for member in sample_members[:sample_count]:
            data = handle.read(member)
            try:
                info = image_info(data)
                samples.append({
                    'member': member.filename,
                    'bytes': member.file_size,
                    'sha256': sha256_bytes(data),
                    **info,
                })
            except Exception as exc:  # pragma: no cover - corrupt source diagnostic
                samples.append({
                    'member': member.filename,
                    'bytes': member.file_size,
                    'error': repr(exc),
                })

        result: dict[str, Any] = {
            'archive': archive.name,
            'archive_bytes': archive.stat().st_size,
            'archive_sha256': sha256_file(archive),
            'image_members': sum(class_counts.values()),
            'class_folders': len(class_counts),
            'images_per_class': {
                'min': min(class_counts.values()) if class_counts else 0,
                'max': max(class_counts.values()) if class_counts else 0,
                'all_counts_equal': len(set(class_counts.values())) <= 1,
            },
            'image_extensions': dict(Counter(Path(member.filename).suffix.lower() for items in class_members.values() for member in items)),
            'class_folder_examples': sorted(class_counts)[:20],
            'samples': samples,
            'invalid_image_layout_members': invalid_layout[:20],
        }

        if manifest_path:
            manifest = load_real_manifest(manifest_path)
            matching = 0
            hash_matches = 0
            manifest_labels = Counter(label for label, _ in manifest)
            archive_classes = {normalize_label(name) for name in class_members}
            for class_name, members in class_members.items():
                normalized = normalize_label(class_name)
                for member in members:
                    row = manifest.get((normalized, Path(member.filename).name))
                    if not row:
                        continue
                    matching += 1
                    data = handle.read(member)
                    if row.get('sha256') == sha256_bytes(data):
                        hash_matches += 1
            result['historical_manifest'] = {
                'path_name': manifest_path.name,
                'real_render_rows': len(manifest),
                'real_render_labels': len(manifest_labels),
                'real_render_rows_matching_archive_member': matching,
                'matching_member_hashes': hash_matches,
                'manifest_labels_not_in_archive': sorted(set(manifest_labels) - archive_classes),
                'archive_labels_not_in_manifest': sorted(archive_classes - set(manifest_labels)),
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--historical-manifest', type=Path)
    parser.add_argument('--sample-count', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect_archive(args.archive, args.historical_manifest, max(0, args.sample_count))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
