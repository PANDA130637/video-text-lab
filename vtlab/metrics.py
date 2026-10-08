"""Retrieval and order diagnostics on normalized embeddings.

Retrieval accepts every matching semantic_id as correct; text candidates are
deduplicated by (semantic_id, text). Recall is the expectation under uniform
random ordering of scores tied at the cutoff, so candidate storage order cannot
manufacture a gain. Chance R@1 depends on class multiplicities, not simply 1/N.
Order matching has chance 0.5 and is reported separately from ordinary retrieval.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any

import numpy as np


TIE_ATOL = 1e-6


def _recall(scores: np.ndarray, query_labels: list[str], candidate_labels: list[str],
            ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    totals = {k: 0.0 for k in ks}
    labels = np.asarray(candidate_labels)
    for row, label in zip(scores, query_labels):
        positive = labels == label
        if not positive.any():
            continue
        best_positive = float(row[positive].max())
        tied = np.isclose(row, best_positive, rtol=0, atol=TIE_ATOL)
        above = int(((row > best_positive) & ~tied).sum())
        good = int((tied & positive).sum())
        bad = int((tied & ~positive).sum())
        for k in ks:
            slots = min(k, len(row)) - above
            if slots <= 0:
                credit = 0.0
            elif slots > bad:
                credit = 1.0
            else:
                # Probability of drawing only negatives without replacement.
                miss = 1.0
                for i in range(slots):
                    miss *= (bad - i) / (bad + good - i)
                credit = 1.0 - miss
            totals[k] += credit
    return {f"recall_at_{k}": value / len(query_labels) for k, value in totals.items()}


def _credit(margin: float) -> float:
    return 0.5 if math.isclose(margin, 0.0, abs_tol=TIE_ATOL) else float(margin > 0)


def retrieval_metrics(video_embeddings: np.ndarray, text_embeddings: np.ndarray,
                      records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate aligned video/text embeddings, returning JSON-safe output.

    A pair is exactly two records sharing an explicitly annotated pair_id,
    with distinct semantic labels. Pair accuracy compares the sum of scores for
    the correct assignment against the swapped assignment. Query accuracy tests
    each video's own caption against the opposite caption; ties receive 0.5.
    """
    videos = np.asarray(video_embeddings, dtype=np.float32)
    texts = np.asarray(text_embeddings, dtype=np.float32)
    if (videos.ndim != 2 or texts.ndim != 2 or len(videos) != len(records)
            or len(texts) != len(records) or len(records) == 0
            or videos.shape[1] != texts.shape[1]):
        raise ValueError("Embeddings must be nonempty, aligned [N,D] arrays")
    if not np.isfinite(videos).all() or not np.isfinite(texts).all():
        raise ValueError("Embeddings contain nonfinite values")
    labels = [str(record["semantic_id"]) for record in records]
    candidate_indices: list[int] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        key = (labels[index], str(record["text"]))
        if key not in seen:
            seen.add(key)
            candidate_indices.append(index)
    candidate_labels = [labels[index] for index in candidate_indices]
    candidates = texts[candidate_indices]
    # Chunking bounds the temporary matrix for larger CPU-only collections.
    v2t = {f"recall_at_{k}": 0.0 for k in (1, 5, 10)}
    predictions = []
    for start in range(0, len(videos), 256):
        scores = videos[start:start + 256] @ candidates.T
        current = _recall(scores, labels[start:start + 256], candidate_labels)
        for key, value in current.items():
            v2t[key] += value * len(scores) / len(videos)
        for offset, row in enumerate(scores):
            index = start + offset
            predicted = candidate_indices[int(np.argmax(row))]
            predictions.append(dict(id=str(records[index]["id"]),
                                    semantic_id=labels[index],
                                    predicted_semantic_id=labels[predicted],
                                    predicted_text=str(records[predicted]["text"]),
                                    score=float(row.max()),
                                    correct=labels[index] == labels[predicted]))
    t2v = {f"recall_at_{k}": 0.0 for k in (1, 5, 10)}
    for start in range(0, len(candidates), 256):
        scores = candidates[start:start + 256] @ videos.T
        current = _recall(scores, candidate_labels[start:start + 256], labels)
        for key, value in current.items():
            t2v[key] += value * len(scores) / len(candidates)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        # Source grouping prevents leakage; it does not itself annotate an
        # opposite-order pair. Never invent order supervision from group_id.
        group = record.get("pair_id")
        if group:
            groups[str(group)].append(index)
    pairs = []
    for group, indices in sorted(groups.items()):
        if len(indices) != 2:
            continue
        a, b = indices
        if labels[a] == labels[b]:
            continue
        scores = videos[[a, b]] @ texts[[a, b]].T
        margin_a = float(scores[0, 0] - scores[0, 1])
        margin_b = float(scores[1, 1] - scores[1, 0])
        pairs.append(dict(pair_id=group, ids=[str(records[i]["id"]) for i in indices],
                          pair_accuracy=_credit(margin_a + margin_b),
                          query_accuracy=(_credit(margin_a) + _credit(margin_b)) / 2,
                          query_margins=[margin_a, margin_b]))
    candidate_counts, video_counts = Counter(candidate_labels), Counter(labels)
    metrics = dict(video_to_text=v2t, text_to_video=t2v,
                   mean_recall_at_1=(v2t["recall_at_1"] + t2v["recall_at_1"]) / 2,
                   order_pair_accuracy=(float(np.mean([p["pair_accuracy"] for p in pairs]))
                                        if pairs else None),
                   order_query_accuracy=(float(np.mean([p["query_accuracy"] for p in pairs]))
                                         if pairs else None),
                   video_to_text_chance_at_1=float(np.mean([
                       candidate_counts[label] / len(candidate_labels) for label in labels])),
                   text_to_video_chance_at_1=float(np.mean([
                       video_counts[label] / len(labels) for label in candidate_labels])))
    return dict(metrics=metrics, predictions=predictions, order_pairs=pairs,
                counts=dict(videos=len(records), text_candidates=len(candidates),
                            semantic_classes=len(set(labels)), order_pairs=len(pairs)),
                tie_policy="expected recall under uniform tie breaking; atol=1e-6")


compute_metrics = retrieval_metrics
