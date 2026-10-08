"""Deterministic small-model training with validation-only checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .metrics import retrieval_metrics
from .models import build_model


def load_cache(cache_path: Path | str) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as archive:
        missing = {"video_features", "text_features", "metadata"} - set(archive.files)
        if missing:
            raise ValueError(f"Cache missing keys: {sorted(missing)}")
        video = np.asarray(archive["video_features"], dtype=np.float32)
        text = np.asarray(archive["text_features"], dtype=np.float32)
        records = json.loads(str(archive["metadata"].item()))
        provenance = (json.loads(str(archive["provenance"].item()))
                      if "provenance" in archive else {})
    if not isinstance(records, list) or not records:
        raise ValueError("Cache metadata must contain a nonempty JSON list")
    if not isinstance(provenance, dict):
        raise ValueError("Cache provenance must be a JSON object")
    if (video.ndim != 3 or text.ndim != 2 or len(video) != len(records)
            or len(text) != len(records) or min(video.shape) == 0 or min(text.shape) == 0):
        raise ValueError("Cache requires aligned video [N,T,Dv] and text [N,Dt] features")
    if not np.isfinite(video).all() or not np.isfinite(text).all():
        raise ValueError("Cache contains NaN or infinite features")
    required = {"id", "group_id", "split", "video_path", "text", "semantic_id"}
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    pair_splits: dict[str, str] = {}
    path_splits: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or required - record.keys():
            raise ValueError(f"Each cache record requires {sorted(required)}")
        if any(not str(record[field]).strip() for field in required):
            raise ValueError("Required cache record values must not be empty")
        identifier, group, split = str(record["id"]), str(record["group_id"]), record["split"]
        if identifier in ids:
            raise ValueError(f"Duplicate record id: {identifier}")
        ids.add(identifier)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Data leakage: group {group} spans multiple splits")
        group_splits[group] = split
        video_path = str(Path(record["video_path"]))
        if video_path in path_splits and path_splits[video_path] != split:
            raise ValueError(f"Data leakage: video path {video_path} spans multiple splits")
        path_splits[video_path] = split
        if record.get("pair_id"):
            pair = str(record["pair_id"])
            if pair in pair_splits and pair_splits[pair] != split:
                raise ValueError(f"Data leakage: pair {pair} spans multiple splits")
            pair_splits[pair] = split
    return dict(video_features=video, text_features=text, records=records,
                provenance=provenance)


def _subset(cache: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    indices = [index for index, record in enumerate(cache["records"]) if record["split"] == split]
    if not indices:
        raise ValueError(f"Cache contains no {split!r} records")
    return (cache["video_features"][indices], cache["text_features"][indices],
            [cache["records"][index] for index in indices])


def _device(device: str) -> torch.device:
    chosen = torch.device(device)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    if chosen.type == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise ValueError("XPU requested but unavailable; use --device cpu")
    return chosen


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def multi_positive_loss(video_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
                        labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Log probability of any correct semantic label, in both directions.

    Repeated captions/classes are positives, never artificial negative examples.
    A batch containing only one class carries no contrastive training signal.
    """
    scores = video_embeddings @ text_embeddings.T / temperature
    positive = labels[:, None] == labels[None, :]
    positive_scores = scores.masked_fill(~positive, float("-inf"))
    row_loss = torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)
    col_loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(positive_scores, dim=0)
    return (row_loss.mean() + col_loss.mean()) / 2


def _encode(model: torch.nn.Module, video: np.ndarray, text: np.ndarray,
            device: torch.device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    video_outputs, text_outputs = [], []
    with torch.inference_mode():
        for start in range(0, len(video), batch_size):
            outputs = model(torch.from_numpy(video[start:start + batch_size]).to(device),
                            torch.from_numpy(text[start:start + batch_size]).to(device))
            video_outputs.append(outputs[0].cpu().numpy())
            text_outputs.append(outputs[1].cpu().numpy())
    return np.concatenate(video_outputs), np.concatenate(text_outputs)


def _label_indices(records: list[dict]) -> torch.Tensor:
    labels = sorted({str(record["semantic_id"]) for record in records})
    mapping = {label: index for index, label in enumerate(labels)}
    return torch.tensor([mapping[str(record["semantic_id"])] for record in records], dtype=torch.long)


def _validation_loss(video: np.ndarray, text: np.ndarray, records: list[dict]) -> float:
    # Fixed chunks bound RAM; validation loss is only a checkpoint tie-breaker.
    labels = _label_indices(records)
    total = 0.0
    for start in range(0, len(records), 256):
        stop = start + 256
        loss = multi_positive_loss(torch.from_numpy(video[start:stop]),
                                   torch.from_numpy(text[start:stop]), labels[start:stop])
        total += float(loss) * len(video[start:stop])
    return total / len(records)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def train(cache_path: Path, output_dir: Path, model_name: str = "mean", seed: int = 42,
          epochs: int = 40, batch_size: int = 32, lr: float = 0.003,
          threads: int = 4, device: str = "cpu") -> dict[str, Any]:
    if epochs < 1 or batch_size < 2 or threads < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Require epochs >= 1, batch_size >= 2, threads >= 1 and finite lr > 0")
    started = time.perf_counter()
    torch.set_num_threads(threads)
    _seed(seed)
    chosen = _device(device)
    cache = load_cache(cache_path)
    train_video, train_text, train_records = _subset(cache, "train")
    val_video, val_text, val_records = _subset(cache, "val")
    if len({str(record["semantic_id"]) for record in train_records}) < 2:
        raise ValueError("Contrastive training requires at least two training semantic classes")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    model = build_model(model_name, train_video.shape[-1], train_text.shape[-1]).to(chosen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_video_tensor = torch.from_numpy(train_video)
    train_text_tensor = torch.from_numpy(train_text)
    train_labels = _label_indices(train_records)
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_key = (float("-inf"), float("-inf"))
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_records), generator=generator)
        loss_sum, samples = 0.0, 0
        for indices in permutation.split(batch_size):
            if len(indices) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            videos, texts = model(train_video_tensor[indices].to(chosen),
                                   train_text_tensor[indices].to(chosen))
            loss = multi_positive_loss(videos, texts, train_labels[indices].to(chosen))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            samples += len(indices)
        val_embeddings = _encode(model, val_video, val_text, chosen)
        validation = retrieval_metrics(*val_embeddings, val_records)["metrics"]
        val_loss = _validation_loss(*val_embeddings, val_records)
        if not np.isfinite(val_loss):
            raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch}")
        entry = dict(epoch=epoch, train_loss=loss_sum / samples, val_loss=val_loss,
                     val_mean_recall_at_1=validation["mean_recall_at_1"],
                     val_order_query_accuracy=validation["order_query_accuracy"],
                     elapsed_seconds=time.perf_counter() - started)
        history.append(entry)
        key = (validation["mean_recall_at_1"], -val_loss)
        if key > best_key:
            best_key, best_epoch = key, epoch
            torch.save(dict(format_version=1, config=model.config,
                            state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
                            seed=seed, epoch=epoch, selection_split="val",
                            selection_metric="mean_recall_at_1, then minimum val_loss",
                            val_metrics=validation, val_loss=val_loss,
                            provenance=cache["provenance"],
                            cache_path=str(Path(cache_path).resolve()),
                            train_config=dict(epochs=epochs, batch_size=batch_size, lr=lr,
                                              threads=threads, device=str(chosen))), checkpoint_path)
    report = dict(model=model_name, checkpoint=str(checkpoint_path.resolve()), seed=seed,
                  epochs=epochs, best_epoch=best_epoch, selection_split="val",
                  best_val_mean_recall_at_1=best_key[0], best_val_loss=-best_key[1],
                  train_records=len(train_records), val_records=len(val_records),
                  trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
                  device=str(chosen), threads=threads,
                  elapsed_seconds=time.perf_counter() - started,
                  provenance=cache["provenance"], test_used_for_selection=False)
    _json_write(output_dir / "history.json", history)
    _json_write(output_dir / "train_report.json", report)
    return report


def load_model(checkpoint_path: Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    chosen = _device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "state_dict"} <= checkpoint.keys():
        raise ValueError("Checkpoint missing config or state_dict")
    model = build_model(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(chosen).eval(), checkpoint


def validate_cache_compatibility(cache: dict[str, Any], checkpoint: dict[str, Any]) -> None:
    """Allow a new holdout dataset, but reject changed feature definitions."""
    config = checkpoint["config"]
    if (cache["video_features"].shape[-1] != config["video_dim"]
            or cache["text_features"].shape[-1] != config["text_dim"]):
        raise ValueError("Cache feature dimensions do not match the checkpoint")
    trained_on, incoming = checkpoint.get("provenance", {}), cache.get("provenance", {})
    for key in ("encoder", "model_name", "extractor_version", "frames"):
        if key in trained_on and key in incoming and trained_on[key] != incoming[key]:
            raise ValueError(f"Cache {key}={incoming[key]!r} differs from checkpoint {trained_on[key]!r}")


def evaluate(cache_path: Path, checkpoint_path: Path, split: str = "test",
             shuffle_frames: bool = False, seed: int = 42, device: str = "cpu") -> dict[str, Any]:
    cache = load_cache(cache_path)
    video, text, records = _subset(cache, split)
    model, checkpoint = load_model(checkpoint_path, device=device)
    validate_cache_compatibility(cache, checkpoint)
    config = checkpoint["config"]
    if shuffle_frames:
        # Diagnostic only: never use shuffled test results for model selection.
        rng = np.random.default_rng(seed)
        video = np.stack([frames[rng.permutation(len(frames))] for frames in video])
    embeddings = _encode(model, video, text, _device(device))
    result = retrieval_metrics(*embeddings, records)
    result.update(split=split, model=config["model_name"],
                  checkpoint=str(Path(checkpoint_path).resolve()), seed=seed,
                  shuffle_frames=shuffle_frames, provenance=cache["provenance"],
                  training_seed=checkpoint.get("seed"), selection_split=checkpoint.get("selection_split"))
    return result
