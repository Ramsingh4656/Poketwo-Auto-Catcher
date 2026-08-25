from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf

ROOT = Path('/home/ubuntu/poketwo_autotrain')
DATASET = ROOT / 'dataset'
MAPPING = ROOT / 'class_indices.json'
OUT = ROOT / 'augmented_artifacts'
SEED = 20260820
IMG = 224
BATCH = 32


def load_index() -> tuple[list[str], dict[str, int]]:
    mapping = json.loads(MAPPING.read_text(encoding='utf-8'))
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    return names, mapping


def files_and_labels(split: str, class_names: list[str], mapping: dict[str, int]):
    paths, labels = [], []
    for name in class_names:
        for path in sorted((DATASET / split / name).glob('*.png')):
            paths.append(str(path))
            labels.append(mapping[name])
    if not paths:
        raise RuntimeError(f'No PNG files found in {DATASET / split}')
    return paths, labels


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def decode_rgba(path: tf.Tensor) -> tf.Tensor:
    data = tf.io.read_file(path)
    return tf.image.decode_png(data, channels=4)


def choose_background() -> tf.Tensor:
    # Exact Discord dark and light colors are deliberately present in the
    # sampling pool. The other choices add mild domain variation.
    palette = tf.constant([
        [49.0, 51.0, 56.0],    # Discord dark mode #313338
        [255.0, 255.0, 255.0],  # Discord/light embed white
        [38.0, 43.0, 48.0],
        [245.0, 246.0, 248.0],
        [72.0, 75.0, 82.0],
    ], dtype=tf.float32)
    choice = tf.random.uniform([], 0, tf.shape(palette)[0], dtype=tf.int32)
    return palette[choice] / 255.0


def composite_rgba(rgba: tf.Tensor) -> tf.Tensor:
    rgb = tf.cast(rgba[..., :3], tf.float32) / 255.0
    alpha = tf.cast(rgba[..., 3:4], tf.float32) / 255.0
    background = tf.reshape(choose_background(), [1, 1, 3])
    return rgb * alpha + background * (1.0 - alpha)


def scale_and_pad(img: tf.Tensor) -> tf.Tensor:
    scale = tf.random.uniform([], 0.70, 1.30)
    side = tf.cast(tf.round(scale * IMG), tf.int32)
    resized = tf.image.resize(img, [side, side], method='bilinear')

    def crop_large() -> tf.Tensor:
        return tf.image.random_crop(resized, [IMG, IMG, 3], seed=SEED)

    def pad_small() -> tf.Tensor:
        # Crop-and-pad keeps the image on a fixed canvas while preserving the
        # simulated zoom-out margin.
        return tf.image.resize_with_crop_or_pad(resized, IMG, IMG)

    return tf.cond(side >= IMG, crop_large, pad_small)


def blur_once(img: tf.Tensor) -> tf.Tensor:
    kernel = tf.constant([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], tf.float32) / 16.0
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    return tf.nn.depthwise_conv2d(img[None, ...], kernel, [1, 1, 1, 1], 'SAME')[0]


def jpeg_roundtrip(img: tf.Tensor) -> tf.Tensor:
    pixels = tf.cast(tf.clip_by_value(img * 255.0, 0.0, 255.0), tf.uint8)
    quality_index = tf.random.uniform([], 0, 4, dtype=tf.int32)
    def encode_at(quality: int) -> tf.Tensor:
        encoded = tf.image.encode_jpeg(pixels, quality=quality)
        return tf.cast(tf.image.decode_jpeg(encoded, channels=3), tf.float32) / 255.0
    return tf.switch_case(quality_index, branch_fns=[
        lambda: encode_at(55), lambda: encode_at(70), lambda: encode_at(85), lambda: encode_at(95)
    ])


def augment(path: tf.Tensor, label: tf.Tensor):
    img = composite_rgba(decode_rgba(path))
    img = scale_and_pad(img)
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, 0.80, 1.20)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.cond(tf.random.uniform([]) < 0.35, lambda: blur_once(img), lambda: img)
    img = tf.cond(tf.random.uniform([]) < 0.45, lambda: jpeg_roundtrip(img), lambda: img)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.025)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img * 255.0, label


def clean(path: tf.Tensor, label: tf.Tensor):
    img = composite_rgba(decode_rgba(path))
    img = tf.image.resize_with_pad(img, IMG, IMG)
    return img * 255.0, label


def make_dataset(split: str, class_names: list[str], mapping: dict[str, int], training: bool):
    paths, labels = files_and_labels(split, class_names, mapping)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(clean, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE), len(paths)


def top5(y_true, y_pred):
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG, IMG, 3), include_top=False, include_preprocessing=True, weights='imagenet'
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG, IMG, 3), name='image')
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='probabilities')(x)
    return tf.keras.Model(inputs, outputs, name='poketwo_augmented_mobilenetv3small'), base


def main() -> None:
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    class_names, mapping = load_index()
    train_ds, n_train = make_dataset('train', class_names, mapping, True)
    val_ds, n_val = make_dataset('val', class_names, mapping, False)
    model, base = build_model(len(class_names))
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='top1'), top5]
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss='sparse_categorical_crossentropy', metrics=metrics)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(OUT / 'best_frozen.keras', monitor='val_top1', mode='max', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ]
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=callbacks, verbose=2)

    base.trainable = True
    for layer in base.layers[:-35]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-5), loss='sparse_categorical_crossentropy', metrics=metrics)
    callbacks2 = [
        tf.keras.callbacks.ModelCheckpoint(OUT / 'best_finetuned.keras', monitor='val_top1', mode='max', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ]
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks2, verbose=2)

    evaluation = model.evaluate(val_ds, return_dict=True, verbose=0)
    model.save(OUT / 'pokemon_classifier_augmented.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    (OUT / 'pokemon_classifier_augmented.tflite').write_bytes(tflite)
    (OUT / 'index_to_pokemon.json').write_text((ROOT / 'index_to_pokemon.json').read_text(encoding='utf-8'), encoding='utf-8')
    (OUT / 'class_indices.json').write_text(json.dumps(mapping, indent=2) + '\n', encoding='utf-8')
    summary = {
        'model': 'fresh MobileNetV3Small ImageNet transfer-learning run',
        'classes': len(class_names), 'train_images': n_train, 'val_images': n_val,
        'augmentation': ['alpha composite', '#313338 dark background', '#FFFFFF light background', '0.70-1.30 scale/pad', 'JPEG quality 55-95', '3x3 Gaussian blur', 'pixel noise stddev 0.025', 'brightness/contrast/saturation'],
        'evaluation': {k: float(v) for k, v in evaluation.items()},
        'epochs_frozen': len(history1.history.get('loss', [])), 'epochs_finetuned': len(history2.history.get('loss', [])),
        'warning': 'Validation still contains one clean repository counterpart per class; augmentation improves invariance but cannot measure real Discord screenshot accuracy.',
    }
    (OUT / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
