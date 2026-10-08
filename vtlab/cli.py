"""Command-line entry points; heavy imports are kept out of --help/doctor."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def doctor() -> dict:
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("torch", "numpy", "PIL", "cv2", "transformers")}
    result = {"python": sys.executable, "python_version": platform.python_version(),
              "project": str(PROJECT), "packages": packages, "cpu_is_default": True}
    if packages["torch"]:
        import torch
        result.update(torch_version=torch.__version__,
                      xpu_available=bool(hasattr(torch, "xpu") and torch.xpu.is_available()))
    return result


def seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Provide one or more distinct seeds.")
    return seeds


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-first video/text retrieval research lab.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect existing Python and dependencies; changes nothing.")
    commands.add_parser("test", help="Run the scientific correctness checks.")
    kth = commands.add_parser("prepare-kth", help="Download the official non-commercial KTH d1 three-action subset.")
    kth.add_argument("--output", type=Path, default=Path("data/kth"))

    synth = commands.add_parser("synthetic", help="Generate clearly labelled synthetic controls.")
    synth.add_argument("--output", type=Path, default=Path("data/synthetic"))
    synth.add_argument("--groups", type=int, default=80)
    synth.add_argument("--frames", type=int, default=8)
    synth.add_argument("--seed", type=int, default=2026)

    validate = commands.add_parser("validate", help="Check manifest paths and group split isolation.")
    validate.add_argument("manifest", type=Path)

    extract = commands.add_parser("extract", help="Freeze encoder and cache video/text features.")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--cache", type=Path, required=True)
    extract.add_argument("--encoder", choices=["color", "clip"], default="clip")
    extract.add_argument("--frames", type=int, default=8)
    extract.add_argument("--batch-size", type=int, default=16)
    extract.add_argument("--threads", type=int, default=4)
    extract.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    extract.add_argument("--device", choices=["cpu", "xpu"], default="cpu")
    extract.add_argument("--offline", action="store_true", help="Use downloaded model files without network access.")

    train_parser = commands.add_parser("train", help="Train using train/val only.")
    train_parser.add_argument("--cache", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--model", choices=["mean", "mlp", "gru"], default="gru")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--epochs", type=int, default=40)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=0.003)
    train_parser.add_argument("--threads", type=int, default=4)
    train_parser.add_argument("--device", choices=["cpu", "xpu"], default="cpu")

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate saved validation-selected weights.")
    evaluate_parser.add_argument("--cache", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    evaluate_parser.add_argument("--shuffle-frames", action="store_true")
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument("--device", choices=["cpu", "xpu"], default="cpu")
    evaluate_parser.add_argument("--output", type=Path)

    baseline = commands.add_parser("baseline", help="Evaluate frozen CLIP mean pooling, with no training.")
    baseline.add_argument("--cache", type=Path, required=True)
    baseline.add_argument("--split", choices=["train", "val", "test"], default="test")
    baseline.add_argument("--output", type=Path)

    demo = commands.add_parser("demo", help="Offline synthetic train/evaluate/report in one command.")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    demo.add_argument("--groups", type=int, default=80)
    demo.add_argument("--frames", type=int, default=8)
    demo.add_argument("--epochs", type=int, default=40)
    demo.add_argument("--seeds", type=seed_list, default=[17, 42, 123])
    demo.add_argument("--threads", type=int, default=4)

    benchmark = commands.add_parser("benchmark", help="Fixed-protocol CLIP and trained model comparison.")
    benchmark.add_argument("--cache", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    benchmark.add_argument("--epochs", type=int, default=40)
    benchmark.add_argument("--seeds", type=seed_list, default=[17, 42, 123])
    benchmark.add_argument("--threads", type=int, default=4)

    quality = commands.add_parser("quality", help="Evaluate text alignment and temporal stability from cached features.")
    quality.add_argument("--cache", type=Path, required=True)
    quality.add_argument("--output", type=Path, default=Path("runs/quality"))
    quality.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    quality.add_argument("--flicker-threshold", type=float, default=0.85)
    quality.add_argument("--seed", type=int, default=2026)
    quality.add_argument("--bootstrap-samples", type=int, default=2000)

    search_parser = commands.add_parser("search", help="Retrieve cached videos with a text query.")
    search_parser.add_argument("--cache", type=Path, required=True)
    search_parser.add_argument("--checkpoint", type=Path, required=True)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    search_parser.add_argument("--device", choices=["cpu", "xpu"], default="cpu")
    search_parser.add_argument("--offline", action="store_true")
    return parser


def run_demo(args) -> dict:
    from .data import make_synthetic
    from .encoders import extract_features
    from .training import train, evaluate
    from .report import write_report
    output = args.output.resolve()
    manifest = make_synthetic(output / "data", groups=args.groups, frames=args.frames, seed=2026)
    cache = output / "features.npz"
    extraction = extract_features(manifest, cache, encoder="color", frames=args.frames)
    runs = []
    for model in ("mean", "gru"):
        for seed in args.seeds:
            print(f"Training {model}, seed={seed} ...", flush=True)
            directory = output / f"{model}_{seed}"
            train_result = train(cache, directory, model_name=model, seed=seed,
                                 epochs=args.epochs, threads=args.threads)
            checkpoint = directory / "best.pt"
            normal = evaluate(cache, checkpoint, split="test", seed=seed)
            shuffled = evaluate(cache, checkpoint, split="test", shuffle_frames=True, seed=seed)
            dump_json(directory / "test.json", normal)
            dump_json(directory / "test_shuffled.json", shuffled)
            runs.append({"model": model, "seed": seed, "training": train_result,
                         "normal": normal, "shuffled": shuffled})
    result = {"experiment": "synthetic_control", "data_seed": 2026,
              "notice": "Synthetic shapes + handcrafted features + one-hot text. Not real-video or CLIP performance.",
              "manifest": str(manifest), "cache": str(cache), "extraction": extraction, "runs": runs}
    dump_json(output / "results.json", result)
    paths = write_report(result, output)
    return {"result": str(output / "results.json"), **paths}


def run_search(args) -> dict:
    import numpy as np
    import torch
    from .encoders import encode_text_queries
    from .training import load_model, load_cache, validate_cache_compatibility
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    cache = load_cache(args.cache)
    features, texts = cache["video_features"], cache["text_features"]
    records, provenance = cache["records"], cache["provenance"]
    model, checkpoint = load_model(args.checkpoint, device=args.device)
    validate_cache_compatibility(cache, checkpoint)
    indices = [i for i, r in enumerate(records) if args.split == "all" or r["split"] == args.split]
    if not indices:
        raise ValueError(f"No videos in split {args.split}.")
    exact = next((i for i, r in enumerate(records) if r["text"] == args.query), None)
    if exact is not None:
        query = texts[exact:exact+1]
    else:
        query = encode_text_queries([args.query], encoder=provenance.get("encoder", "clip"),
                                    device=args.device, model_name=provenance.get("model_name", "openai/clip-vit-base-patch32"),
                                    cache_dir=PROJECT / ".cache" / "huggingface")
    with torch.no_grad():
        video_emb = model.encode_video(torch.from_numpy(features[indices]).to(args.device))
        text_emb = model.encode_text(torch.from_numpy(query).to(args.device))
        scores = (video_emb @ text_emb.T).squeeze(1).cpu().numpy()
    order = np.argsort(-scores, kind="stable")[:args.top_k]
    return {"query": args.query, "split": args.split, "provenance": provenance,
            "results": [{"rank": rank+1, "score": float(scores[position]), **records[indices[position]]}
                        for rank, position in enumerate(order)]}


def dispatch(args):
    if args.command == "doctor":
        return doctor()
    if args.command == "prepare-kth":
        from .kth import prepare_kth
        return prepare_kth(args.output)
    if args.command == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return {"tests": result.testsRun, "successful": result.wasSuccessful()}
    if args.command == "synthetic":
        from .data import make_synthetic
        return {"manifest": str(make_synthetic(args.output, groups=args.groups, frames=args.frames, seed=args.seed))}
    if args.command == "validate":
        from .data import validate_manifest
        rows = validate_manifest(args.manifest)
        return {"videos": len(rows), "groups": len({r["group_id"] for r in rows}),
                "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}}
    if args.command == "extract":
        from .encoders import extract_features
        if args.threads < 1:
            raise ValueError("--threads must be positive.")
        if args.encoder == "clip":
            import torch
            torch.set_num_threads(args.threads)
        return extract_features(args.manifest, args.cache, encoder=args.encoder, frames=args.frames,
                                device=args.device, model_name=args.model_name, batch_size=args.batch_size)
    if args.command == "baseline":
        import numpy as np
        from .training import load_cache
        from .metrics import retrieval_metrics
        cache = load_cache(args.cache)
        if cache["provenance"].get("encoder") != "clip":
            raise ValueError("The untrained cosine baseline requires aligned CLIP image/text features.")
        selected = [i for i, record in enumerate(cache["records"]) if record["split"] == args.split]
        if not selected:
            raise ValueError(f"No records in split {args.split}.")
        videos = cache["video_features"][selected].mean(axis=1)
        texts = cache["text_features"][selected]
        videos /= np.maximum(np.linalg.norm(videos, axis=1, keepdims=True), 1e-12)
        texts = texts / np.maximum(np.linalg.norm(texts, axis=1, keepdims=True), 1e-12)
        result = retrieval_metrics(videos, texts, [cache["records"][i] for i in selected])
        result.update(model="frozen_clip_mean", split=args.split, provenance=cache["provenance"], trained=False)
        if args.output:
            dump_json(args.output, result)
        return result
    if args.command == "train":
        from .training import train
        return train(args.cache, args.output, model_name=args.model, seed=args.seed,
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     threads=args.threads, device=args.device)
    if args.command == "evaluate":
        from .training import evaluate
        result = evaluate(args.cache, args.checkpoint, split=args.split,
                          shuffle_frames=args.shuffle_frames, seed=args.seed, device=args.device)
        if args.output:
            dump_json(args.output, result)
        return result
    if args.command == "demo":
        return run_demo(args)
    if args.command == "benchmark":
        from .benchmark import benchmark
        return benchmark(args.cache, args.output, epochs=args.epochs, seeds=args.seeds, threads=args.threads)
    if args.command == "quality":
        from .quality import evaluate_quality, perturbation_diagnostics, write_quality_report
        from .training import load_cache
        cache = load_cache(args.cache)
        split = None if args.split == "all" else args.split
        result = evaluate_quality(cache["video_features"], cache["text_features"], cache["records"],
                                  split=split, flicker_threshold=args.flicker_threshold,
                                  bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        result["cache_provenance"] = cache["provenance"]
        if split is not None:
            result["diagnostics"] = perturbation_diagnostics(
                cache["video_features"], cache["text_features"], cache["records"],
                split=split, flicker_threshold=args.flicker_threshold, seed=args.seed)
        result["files"] = write_quality_report(result, args.output)
        return result
    if args.command == "search":
        return run_search(args)
    raise ValueError(f"Unknown command: {args.command}")


def main(argv=None) -> int:
    # Avoid placing downloaded model weights in the user's global model cache.
    os.environ.setdefault("HF_HOME", str(PROJECT / ".cache" / "huggingface"))
    parser = make_parser()
    args = parser.parse_args(argv)
    if getattr(args, "offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        result = dispatch(args)
    except (OSError, ValueError, RuntimeError, ImportError, ArithmeticError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Run 'python run.py doctor'; use the existing interpreter and install missing project dependencies only.", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if isinstance(result, dict) and result.get("successful") is False else 0
