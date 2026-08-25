# Stage 4: Data Provenance and Preprocessing Evidence

Stage 4 recovered and validated the public source data referenced by the repository’s historical manifests. This stage did **not** train a model, alter live preprocessing, or commit raw archives or images. The portable inspector at `scripts/inspect_stage4_provenance.py` inventories a supplied archive without extracting the dataset and can reproduce the historical real-render hash check.

## 1. Recovered sources

Two public Kaggle sources were re-fetched outside the repository. Their licensing and provenance statements are recorded as listed by the source pages; the Apache source also contains an ownership/provenance note, so its contents are treated as validated source material rather than assumed to be independently authored.

| Source | License as listed | Archive size and SHA-256 | Observed contents | Role in Stage 4 |
|---|---|---:|---|---|
| [`spreadsheets600/poketwo-spawn-images-v1`][1] | CC0 1.0 / Public Domain | 381,269,985 bytes; `2fdb999fc1fd101386c55395f11407013ed8f9dfd2c7a28ea7d663838cf7ce43` | 21,954 JPEG images; 69 class folders nested under train/val/test; all 224×224 RGB; split counts 21,353 train, 301 val, 300 test | Legitimate square spawn-render source and preprocessing control set |
| [`dhruv2015/poketwo-datset-3`, version 2][2] | Apache License 2.0 | 8,006,210,062 bytes; `34033229d7475a39e32f19e6c32c4cb7bbd459ffc04de8c84625112b67e1d9ef` | 33,856 PNG images in 1,472 class folders; exactly 23 images per folder; all observed images 800×500 RGB | Original-source lead for the historical full-coverage real-render rows |

The CC0 page’s displayed counts contain a minor README/data-card inconsistency; the counts above are the exact files observed in the downloaded archive. The full archive was validated from its ZIP central directory and from decoded image samples without unpacking the complete 8 GB file.

## 2. Historical manifest validation

The repository’s `reports/full_coverage/full_manifest.csv` contains 21,169 rows. Its 19,297 rows marked `kaggle_real_render` correspond to 839 labels and are the rows attributed to the full Kaggle render source. Every one of those rows matched an archive member by normalized class folder and filename, and every one of the 19,297 recorded SHA-256 values matched the re-downloaded archive bytes.

| Validation item | Exact result |
|---|---:|
| Historical manifest rows | 21,169 |
| Official-reference rows | 1,872 |
| Historical full-source render rows | 19,297 |
| Full-source render labels | 839 |
| Render rows matching re-downloaded archive members | 19,297 / 19,297 |
| Matching member hashes | 19,297 / 19,297 |
| Full archive folders | 1,472 |
| Full archive folders outside the historical 839-label real-render subset | 633 |

This is strong provenance evidence that the historical real-render portion was not fabricated or irretrievably lost: it is recoverable from the cited public archive. The downloaded archive is broader than the deployed model’s target universe, so it must not be copied wholesale into the repository or treated as a ready-to-train 1,472-class target.

## 3. Deployed-label coverage

The active ONNX model has 936 labels. The historical full manifest provides independent 800×500 real-render images for 839 of them, or **89.6368%** of the deployed label universe. The remaining 97 deployed labels have official reference assets in the historical manifest but no matching full-source render folder there. This is a data-coverage observation, not an assertion that those 97 labels are uncatchable; the Stage 2 official-data analysis established the 936-label ceiling separately.

| Deployed target | Labels | Share of 936 |
|---|---:|---:|
| Independent full-source real-render coverage | 839 | 89.6368% |
| Reference-only in historical full manifest | 97 | 10.3632% |
| Total active ONNX universe | 936 | 100% |

The full archive’s 1,472 folders include many additional base species, regional forms, event forms, and other variants that are not in the active 936-label ONNX universe. The repository’s existing full manifest intentionally filtered that broader archive to the official enabled-and-catchable target labels.

## 4. Crop and resize evidence

The live predictor currently performs a direct RGB Lanczos resize to 224×224. The recovered full-source images are 800×500, so their aspect ratio is 1.6:1 while the model input is square. Direct resizing maps the 800-pixel width to 224 pixels and the 500-pixel height to 224 pixels; relative to preserving the height scale, the horizontal dimension is compressed to **0.625×**. This is a geometric distortion, not a crop.

An aspect-preserving contain operation would fit 800×500 into 224×224 as 224×140, leaving 84 pixels of vertical padding, approximately 42 pixels at the top and bottom. The recovered CC0 images are already 224×224, so direct resizing introduces no aspect-ratio distortion for that source.

Representative panels were rendered outside Git from full-source samples for Abomasnow, Pikachu, and Aegislash-Blade, and from a native-square CC0 sample. Visual review showed the same structural pattern across the full-source examples: direct resizing broadens or narrows the complete spawn scene, while contain-padding preserves the scene proportions. The square CC0 sample showed no corresponding geometric difference.

> **Decision:** Keep the live direct-resize path unchanged during Stage 4. The evidence justifies a controlled Stage 5 comparison of direct resize versus aspect-preserving contain-padding, but it does **not** prove an accuracy improvement and does not justify deploying a new preprocessing transform without a benchmark.

The debug panels and extracted samples remain in the external Stage 4 workspace. The repository contains only sanitized sample metadata in `reports/stage4_selected_sample_manifest.json`, not the image bytes.

## 5. Reproduction

To inspect a downloaded full archive and compare it with the historical real-render manifest, run:

```bash
python3 scripts/inspect_stage4_provenance.py \
  --archive /path/to/poketwo-datset-3-v2.zip \
  --historical-manifest reports/full_coverage/full_manifest.csv \
  --sample-count 20 \
  --output /tmp/stage4_provenance.json
```

The command is read-only with respect to the archive and repository. It writes a JSON report containing the archive checksum, image/class counts, decoded sample dimensions and modes, and—when a historical manifest is supplied—the exact member and hash-match counts.

## 6. Scope boundary

No Stage 5 training, model replacement, class-universe expansion, accuracy claim, or live crop change was performed. The next stage requires explicit approval before any training or benchmark pipeline is started.

## References

[1]: https://www.kaggle.com/datasets/spreadsheets600/poketwo-spawn-images-v1 "Kaggle: Pokétwo Spawn Images v1"
[2]: https://www.kaggle.com/datasets/dhruv2015/poketwo-datset-3 "Kaggle: Pokétwo Dataset 3"
[3]: https://github.com/poketwo/data "Pokétwo official data repository"
