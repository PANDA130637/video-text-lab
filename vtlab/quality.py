"""Lightweight text alignment and temporal stability evaluation.

The evaluator is deliberately metric-oriented: it does not claim to replace
human preference tests or a learned video-quality model. It uses the same
frozen frame/text features as the retrieval experiment so that a generated
video, a real video, and a controlled perturbation can be compared with one
reproducible protocol.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _bootstrap_ci(values: list[float], seed: int, samples: int) -> list[float]:
    """Deterministic percentile bootstrap interval for a small test set."""
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _record_metrics(frames: np.ndarray, text: np.ndarray, record: dict[str, Any],
                    flicker_threshold: float) -> dict[str, Any]:
    frame_embeddings = _normalize(frames)
    text_embedding = _normalize(text[None])[0]
    alignment = frame_embeddings @ text_embedding
    adjacent = np.sum(frame_embeddings[1:] * frame_embeddings[:-1], axis=1)
    alignment_mean = float(np.mean(alignment))
    alignment_p10 = float(np.percentile(alignment, 10))
    adjacent_mean = float(np.mean(adjacent)) if len(adjacent) else 1.0
    adjacent_std = float(np.std(adjacent)) if len(adjacent) else 0.0
    flicker_rate = float(np.mean(adjacent < flicker_threshold)) if len(adjacent) else 0.0
    alignment_score = _clip01((alignment_mean + 1.0) / 2.0)
    stability_score = _clip01((adjacent_mean + 1.0) / 2.0)
    # This is an explicitly heuristic score, useful for ranking candidates.
    quality_score = 100.0 * (0.65 * alignment_score + 0.35 * stability_score)
    return {
        "id": str(record["id"]),
        "group_id": str(record["group_id"]),
        "split": str(record["split"]),
        "semantic_id": str(record["semantic_id"]),
        "text": str(record["text"]),
        "alignment_mean": alignment_mean,
        "alignment_p10": alignment_p10,
        "temporal_adjacent_cosine": adjacent_mean,
        "temporal_adjacent_std": adjacent_std,
        "flicker_rate": flicker_rate,
        "alignment_score": alignment_score,
        "stability_score": stability_score,
        "quality_score": float(quality_score),
    }


def evaluate_quality(video_features: np.ndarray, text_features: np.ndarray,
                     records: list[dict[str, Any]], split: str | None = None,
                     flicker_threshold: float = 0.85, bootstrap_samples: int = 2000,
                     seed: int = 2026) -> dict[str, Any]:
    """Evaluate aligned frame/text features and return per-video metrics.

    ``quality_score`` is a transparent heuristic (65% text alignment and 35%
    adjacent-frame stability), not a calibrated perceptual score.
    """
    video_features = np.asarray(video_features, dtype=np.float32)
    text_features = np.asarray(text_features, dtype=np.float32)
    if video_features.ndim != 3 or text_features.ndim != 2:
        raise ValueError("Quality evaluation requires video [N,T,D] and text [N,D] features")
    if len(video_features) != len(text_features) or len(records) != len(video_features):
        raise ValueError("Video, text, and records must have the same length")
    if not (0.0 < flicker_threshold <= 1.0):
        raise ValueError("flicker_threshold must be in (0, 1]")
    selected = [i for i, record in enumerate(records)
                if split is None or record["split"] == split]
    if not selected:
        raise ValueError(f"No records in split {split!r}")
    rows = [_record_metrics(video_features[i], text_features[i], records[i], flicker_threshold)
            for i in selected]
    summary = {
        "videos": len(rows),
        "quality_score_mean": float(np.mean([r["quality_score"] for r in rows])),
        "quality_score_std": float(np.std([r["quality_score"] for r in rows])),
        "alignment_mean": float(np.mean([r["alignment_mean"] for r in rows])),
        "temporal_adjacent_cosine": float(np.mean([r["temporal_adjacent_cosine"] for r in rows])),
        "flicker_rate": float(np.mean([r["flicker_rate"] for r in rows])),
        "quality_score_ci95": _bootstrap_ci([r["quality_score"] for r in rows], seed, bootstrap_samples),
        "alignment_mean_ci95": _bootstrap_ci([r["alignment_mean"] for r in rows], seed + 1, bootstrap_samples),
        "temporal_adjacent_cosine_ci95": _bootstrap_ci([r["temporal_adjacent_cosine"] for r in rows], seed + 2, bootstrap_samples),
    }
    by_semantic = {}
    for semantic_id in sorted({r["semantic_id"] for r in rows}):
        group = [r for r in rows if r["semantic_id"] == semantic_id]
        by_semantic[semantic_id] = {
            "videos": len(group),
            "quality_score_mean": float(np.mean([r["quality_score"] for r in group])),
            "quality_score_ci95": _bootstrap_ci([r["quality_score"] for r in group], seed, bootstrap_samples),
            "alignment_mean": float(np.mean([r["alignment_mean"] for r in group])),
            "temporal_adjacent_cosine": float(np.mean([r["temporal_adjacent_cosine"] for r in group])),
            "flicker_rate": float(np.mean([r["flicker_rate"] for r in group])),
        }
    return {
        "protocol": {
            "split": split or "all",
            "flicker_threshold": flicker_threshold,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "score_formula": "100 * (0.65 * alignment_score + 0.35 * stability_score)",
            "alignment_score": "clip((mean frame-text cosine + 1) / 2, 0, 1)",
            "stability_score": "clip((mean adjacent-frame cosine + 1) / 2, 0, 1)",
            "notice": "Heuristic diagnostic; not a human-calibrated perceptual quality score.",
        },
        "summary": summary,
        "by_semantic_id": by_semantic,
        "videos": rows,
    }


def perturbation_diagnostics(video_features: np.ndarray, text_features: np.ndarray,
                             records: list[dict[str, Any]], split: str = "test",
                             flicker_threshold: float = 0.85,
                             seed: int = 2026) -> dict[str, Any]:
    """Compare clean features with deterministic order and flicker controls."""
    video_features = np.asarray(video_features, dtype=np.float32)
    selected = [i for i, record in enumerate(records) if record["split"] == split]
    if len(selected) < 2:
        raise ValueError("Perturbation diagnostics require at least two videos in the split")
    clean = evaluate_quality(video_features[selected], np.asarray(text_features)[selected],
                             [records[i] for i in selected], split=split,
                             flicker_threshold=flicker_threshold, seed=seed)
    shuffled = video_features[selected].copy()
    rng = np.random.default_rng(seed)
    for row in shuffled:
        rng.shuffle(row, axis=0)
    shuffled_result = evaluate_quality(shuffled, np.asarray(text_features)[selected],
                                       [records[i] for i in selected], split=split,
                                       flicker_threshold=flicker_threshold, seed=seed)
    flicker = video_features[selected].copy()
    donor = np.roll(video_features[selected], -1, axis=0)
    flicker[:, 1::2] = donor[:, 1::2]
    flicker_result = evaluate_quality(flicker, np.asarray(text_features)[selected],
                                      [records[i] for i in selected], split=split,
                                      flicker_threshold=flicker_threshold, seed=seed)
    return {
        "split": split,
        "seed": seed,
        "clean": clean["summary"],
        "shuffled_frames": shuffled_result["summary"],
        "cross_video_flicker": flicker_result["summary"],
        "interpretation": "Controls are diagnostics, not proof that a metric matches human preference.",
    }


def write_quality_report(result: dict[str, Any], output: Path | str) -> dict[str, str]:
    """Write JSON, CSV, Markdown, and a small offline HTML report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "quality.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = result.get("videos", [])
    if rows:
        with (output / "quality.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = result["summary"]
    diagnostics = result.get("diagnostics", {})
    lines = [
        "# Video quality diagnostics", "",
        "This report uses frozen CLIP frame/text features. The combined score is a transparent heuristic, not a human-calibrated perceptual metric.", "",
        f"- Videos: {summary['videos']}",
        f"- Mean quality score: {summary['quality_score_mean']:.2f} ± {summary['quality_score_std']:.2f}",
        f"- 95% bootstrap interval: {summary['quality_score_ci95'][0]:.2f}–{summary['quality_score_ci95'][1]:.2f}",
        f"- Mean frame-text alignment: {summary['alignment_mean']:.4f}",
        f"- Mean adjacent-frame cosine: {summary['temporal_adjacent_cosine']:.4f}",
        f"- Mean flicker rate (threshold {result['protocol']['flicker_threshold']:.2f}): {summary['flicker_rate']:.3f}",
        "", "## Perturbation diagnostics", "",
    ]
    for name, value in diagnostics.items():
        if isinstance(value, dict) and "quality_score_mean" in value:
            lines.append(f"- {name}: score {value['quality_score_mean']:.2f}, alignment {value['alignment_mean']:.4f}, adjacent cosine {value['temporal_adjacent_cosine']:.4f}, flicker rate {value['flicker_rate']:.3f}")
    lines += ["", "## By semantic label", ""]
    for semantic_id, value in result.get("by_semantic_id", {}).items():
        lines.append(f"- {semantic_id}: {value['quality_score_mean']:.2f} (95% CI {value['quality_score_ci95'][0]:.2f}–{value['quality_score_ci95'][1]:.2f}), alignment {value['alignment_mean']:.4f}, flicker {value['flicker_rate']:.3f}")
    lines += ["", "## Limitations", "", "The score can reward visually smooth but semantically wrong videos and can penalize intentional fast motion. Generated-video evaluation should be calibrated against human pairwise judgments before being used as a model-selection target."]
    (output / "quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = "".join(f"<tr><td>{r['id']}</td><td>{r['quality_score']:.2f}</td><td>{r['alignment_mean']:.4f}</td><td>{r['temporal_adjacent_cosine']:.4f}</td><td>{r['flicker_rate']:.3f}</td></tr>" for r in rows)
    html = """<!doctype html><meta charset='utf-8'><title>Video quality diagnostics</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#17324d}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.note{background:#fff3d6;padding:14px;border-radius:8px}</style><h1>Video quality diagnostics</h1><div class='note'>Frozen CLIP alignment plus temporal stability. The combined score is a heuristic, not a human-calibrated perceptual metric.</div><p>Mean score: <b>""" + f"{summary['quality_score_mean']:.2f} ± {summary['quality_score_std']:.2f}""" + """</p><p>Alignment: """ + f"{summary['alignment_mean']:.4f}""" + " · Adjacent cosine: """ + f"{summary['temporal_adjacent_cosine']:.4f}""" + " · Flicker rate: """ + f"{summary['flicker_rate']:.3f}""" + """</p><table><thead><tr><th>Video</th><th>Score</th><th>Alignment</th><th>Adjacent cosine</th><th>Flicker</th></tr></thead><tbody>""" + table + "</tbody></table>"""
    (output / "quality.html").write_text(html, encoding="utf-8")
    return {key: str(output / filename) for key, filename in (("json", "quality.json"), ("csv", "quality.csv"), ("markdown", "quality.md"), ("html", "quality.html"))}
