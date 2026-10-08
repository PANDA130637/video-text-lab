"""Small-data manifests, deterministic video sampling, and feature caches."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .encoders import CAPTIONS, SEMANTIC_IDS, get_encoder

REQUIRED_COLUMNS = ("id", "group_id", "split", "video_path", "text", "semantic_id")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
EXTRACTOR_VERSION = "2"


def frame_paths(directory: Path) -> list[Path]:
    def key(path: Path):
        return [int(piece) if piece.isdigit() else piece.lower()
                for piece in re.split(r"(\d+)", path.name)]
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=key)


def resolve_video_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else Path(manifest_path).resolve().parent / path).resolve()


def validate_manifest(manifest_path: Path) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")
        records = [{str(key): (value or "").strip() for key, value in row.items() if key is not None}
                   for row in reader]
    if not records:
        raise ValueError("Manifest is empty.")
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    pair_splits: dict[str, str] = {}
    path_splits: dict[Path, str] = {}
    for row_number, record in enumerate(records, start=2):
        for column in REQUIRED_COLUMNS:
            if not record.get(column):
                raise ValueError(f"Manifest row {row_number}: {column} must not be empty.")
        identifier, split = record["id"], record["split"]
        if identifier in ids:
            raise ValueError(f"Duplicate id in manifest: {identifier}")
        ids.add(identifier)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r} for {identifier}; use train, val, or test.")
        group = record["group_id"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Group split leakage: {group!r} occurs in {group_splits[group]} and {split}.")
        group_splits[group] = split
        pair = record.get("pair_id")
        if pair:
            if pair in pair_splits and pair_splits[pair] != split:
                raise ValueError(f"Pair split leakage: {pair!r} spans splits.")
            pair_splits[pair] = split
        path = resolve_video_path(manifest_path, record["video_path"])
        if not path.exists():
            raise FileNotFoundError(f"Video input not found for {identifier}: {path}")
        if path.is_dir():
            if not frame_paths(path):
                raise ValueError(f"Frame directory has no supported images: {path}")
        elif path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video format: {path.suffix}; use a frame directory or video file.")
        if path in path_splits and path_splits[path] != split:
            raise ValueError(f"Video path split leakage: {path} spans splits.")
        path_splits[path] = split
    return records


def make_synthetic(output_dir: Path, groups: int = 80, frames: int = 8, seed: int = 42) -> Path:
    """Create controlled reverse pairs; these are NOT real-world benchmarks."""
    output_dir = Path(output_dir).resolve()
    if groups < 12:
        raise ValueError("Use at least 12 groups so all eight captions occur in train/val/test.")
    if frames < 2:
        raise ValueError("At least two frames are needed to represent direction.")
    settings = {"kind": "synthetic_control", "version": 1, "groups": groups, "frames": frames, "seed": seed}
    marker = output_dir / ".vtlab-synthetic.json"
    manifest = output_dir / "manifest.csv"
    if output_dir.exists() and any(output_dir.iterdir()):
        if marker.exists() and manifest.exists() and json.loads(marker.read_text(encoding="utf-8")) == settings:
            validate_manifest(manifest)
            return manifest
        raise FileExistsError(f"Refusing to overwrite existing contents in {output_dir}. Choose a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    split_by_group = {}
    for combo in range(4):
        group_ids = list(range(combo, groups, 4))
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_train = min(n - 2, max(1, int(n * 0.7)))
        n_val = min(n - n_train - 1, max(1, int(n * 0.15)))
        for index, group_id in enumerate(group_ids):
            split_by_group[group_id] = "train" if index < n_train else "val" if index < n_train + n_val else "test"
    records = []
    for group_index in range(groups):
        combo = group_index % 4
        group_id = f"scene_{group_index:04d}"
        color_name = "red" if combo < 2 else "blue"
        fill = (235, 25, 25) if color_name == "red" else (25, 25, 235)
        horizontal = combo % 2 == 0
        background = int(rng.integers(15, 65))
        perpendicular = int(rng.integers(17, 47))
        radius = int(rng.integers(3, 7))
        start, end = int(rng.integers(8, 17)), int(rng.integers(47, 56))
        positions = np.rint(np.linspace(start, end, frames)).astype(int)
        scene_frames = []
        for position in positions:
            frame = Image.new("RGB", (64, 64), (background,) * 3)
            x, y = (int(position), perpendicular) if horizontal else (perpendicular, int(position))
            ImageDraw.Draw(frame).ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
            scene_frames.append(frame)
        for direction in (0, 1):
            caption_index = combo * 2 + direction
            video_id = f"{group_id}_{'forward' if direction == 0 else 'reverse'}"
            relative_path = Path("frames") / video_id
            directory = output_dir / relative_path
            directory.mkdir(parents=True, exist_ok=True)
            ordered_frames = scene_frames if direction == 0 else list(reversed(scene_frames))
            for frame_index, frame in enumerate(ordered_frames):
                frame.save(directory / f"frame_{frame_index:04d}.png")
            records.append({
                "id": video_id, "group_id": group_id, "split": split_by_group[group_index],
                "video_path": relative_path.as_posix(), "text": CAPTIONS[caption_index],
                "semantic_id": SEMANTIC_IDS[caption_index], "pair_id": group_id,
                "source": "synthetic_control",
            })
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS) + ["pair_id", "source"])
        writer.writeheader()
        writer.writerows(records)
    marker.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    validate_manifest(manifest)
    return manifest


def _sample_indices(length: int, frames: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot sample an empty video.")
    return np.rint(np.linspace(0, length - 1, frames)).astype(np.int64)


def load_video_frames(path: Path, frames: int = 8) -> list[Image.Image]:
    path = Path(path)
    if frames < 2:
        raise ValueError("Use at least two sampled frames.")
    if path.is_dir():
        paths = frame_paths(path)
        result = []
        for index in _sample_indices(len(paths), frames):
            with Image.open(paths[int(index)]) as image:
                result.append(image.convert("RGB"))
        return result
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Reading video files needs cv2 in your existing Python environment. "
            "No dependency was installed. You can instead provide directories of PNG/JPEG frames."
        ) from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            capture = cv2.VideoCapture(str(path))
        indices = _sample_indices(count, frames)
        wanted = set(map(int, indices))
        sampled = {}
        for index in range(int(indices[-1]) + 1):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Video decoding stopped at frame {index}/{count}: {path}")
            if index in wanted:
                sampled[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return [sampled[int(index)].copy() for index in indices]
    finally:
        capture.release()


def _update_file_hash(digest, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def _fingerprint(manifest_path: Path, records: list[dict], settings: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    _update_file_hash(digest, manifest_path)
    seen = set()
    for record in records:
        path = resolve_video_path(manifest_path, record["video_path"])
        if path in seen:
            continue
        seen.add(path)
        # Include file boundaries/names as well as all bytes; no mtime-only cache.
        for entry in frame_paths(path) if path.is_dir() else [path]:
            digest.update(json.dumps([record["video_path"], entry.name, entry.stat().st_size]).encode("utf-8"))
            _update_file_hash(digest, entry)
    return digest.hexdigest()


def extract_features(manifest_path: Path, cache_path: Path, encoder: str = "color",
                     frames: int = 8, device: str = "cpu",
                     model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 16) -> dict:
    started = time.perf_counter()
    manifest_path, cache_path = Path(manifest_path).resolve(), Path(cache_path).resolve()
    if frames < 2 or batch_size < 1:
        raise ValueError("frames must be >= 2 and batch_size must be >= 1.")
    records = validate_manifest(manifest_path)
    is_synthetic = all(record.get("source") == "synthetic_control" for record in records)
    if encoder == "color" and not is_synthetic:
        raise ValueError("The handcrafted color encoder is restricted to source=synthetic_control. Use encoder='clip' for real data.")
    settings = {"extractor_version": EXTRACTOR_VERSION, "encoder": encoder, "frames": frames,
                "device": device, "model_name": model_name if encoder == "clip" else "handcrafted-dot-centroids-v1"}
    fingerprint = _fingerprint(manifest_path, records, settings)
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                provenance = json.loads(str(archive["provenance"].item()))
                if provenance.get("fingerprint") == fingerprint:
                    return {**provenance, "cache_path": str(cache_path), "reused": True,
                            "elapsed_seconds": round(time.perf_counter() - started, 4)}
        except (KeyError, ValueError, OSError):
            pass  # Interrupted/corrupt/old caches are safely replaced after successful extraction.
    feature_encoder = get_encoder(encoder, device, model_name)
    videos = []
    for record in records:
        images = load_video_frames(resolve_video_path(manifest_path, record["video_path"]), frames)
        features = [feature_encoder.encode_images(images[start:start + batch_size])
                    for start in range(0, len(images), batch_size)]
        videos.append(np.concatenate(features, axis=0))
    video_features = np.stack(videos).astype(np.float32)
    texts = [record["text"] for record in records]
    text_features = np.concatenate([
        feature_encoder.encode_texts(texts[start:start + batch_size])
        for start in range(0, len(texts), batch_size)
    ], axis=0).astype(np.float32)
    if not np.isfinite(video_features).all() or not np.isfinite(text_features).all():
        raise ValueError("Encoder produced non-finite features; no cache was saved.")
    provenance = {
        **settings, "source": "synthetic_control" if is_synthetic else "user_provided",
        "synthetic": is_synthetic, "fingerprint": fingerprint, "manifest_path": str(manifest_path),
        "videos": len(records), "video_shape": list(video_features.shape), "text_shape": list(text_features.shape),
        "extraction_seconds": round(time.perf_counter() - started, 4),
        "warning": "Synthetic control only; not evidence of real-video capability." if is_synthetic else "User-provided data; verify rights and split grouping.",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix=".npz", delete=False) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, video_features=video_features, text_features=text_features,
                                metadata=np.array(json.dumps(records, ensure_ascii=False)),
                                provenance=np.array(json.dumps(provenance, ensure_ascii=False)))
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {**provenance, "cache_path": str(cache_path), "reused": False,
            "elapsed_seconds": round(time.perf_counter() - started, 4)}
