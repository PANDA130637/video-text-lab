"""One fixed experimental protocol, full results including negative findings."""
from __future__ import annotations

from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import statistics

import numpy as np

from .metrics import retrieval_metrics
from .training import load_cache, train, evaluate


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def frozen(cache, frame=None):
    indices = [i for i, row in enumerate(cache["records"]) if row["split"] == "test"]
    video = cache["video_features"][indices]
    video = video.mean(axis=1) if frame is None else video[:, frame]
    text = cache["text_features"][indices]
    video = video / np.maximum(np.linalg.norm(video, axis=1, keepdims=True), 1e-12)
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    return retrieval_metrics(video, text, [cache["records"][i] for i in indices])


def benchmark(cache_path: Path, output: Path, epochs=40, seeds=(17, 42, 123), threads=4):
    cache_path, output = Path(cache_path).resolve(), Path(output).resolve()
    cache = load_cache(cache_path)
    if cache["provenance"].get("encoder") != "clip":
        raise ValueError("This benchmark requires shared-space CLIP features.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing experiments; choose a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(cache=str(cache_path), provenance=cache["provenance"], epochs=epochs,
                    seeds=list(seeds), models=["mean", "mlp", "gru"], batch_size=32, lr=0.003,
                    threads=threads, device="cpu", shuffle_seed=2026,
                    checkpoint_selection="val mean bidirectional Recall@1, then val loss",
                    test_policy="Configuration written before training; all test evaluations follow all training.",
                    dataset_sources=sorted({r.get("source", "unspecified") for r in cache["records"]}),
                    records_by_split=dict(Counter(r["split"] for r in cache["records"])),
                    groups_by_split={s: len({r["group_id"] for r in cache["records"] if r["split"] == s})
                                     for s in ("train", "val", "test")})
    write_json(output / "protocol.json", protocol)
    runs = []
    for name in protocol["models"]:
        for seed in seeds:
            directory = output / f"{name}_{seed}"
            print(f"Training {name}, seed={seed} (test remains untouched)", flush=True)
            training = train(cache_path, directory, model_name=name, seed=seed, epochs=epochs, threads=threads)
            runs.append(dict(model=name, seed=seed, training=training, directory=str(directory)))
    for run in runs:
        directory = Path(run["directory"])
        run["test"] = evaluate(cache_path, directory / "best.pt")
        run["shuffled"] = evaluate(cache_path, directory / "best.pt", shuffle_frames=True, seed=2026)
        write_json(directory / "test.json", run["test"])
        write_json(directory / "test_shuffled.json", run["shuffled"])
    baselines = {"frozen_clip_mean": frozen(cache),
                 "frozen_clip_middle_frame": frozen(cache, cache["video_features"].shape[1] // 2)}
    result = dict(protocol=protocol, baselines=baselines, runs=runs)
    write_json(output / "results.json", result)
    write_report(result, output)
    return {"report": str(output / "report.html"), "results": str(output / "results.json"),
            "runs": len(runs), "synthetic": cache["provenance"].get("synthetic")}


def write_report(result, output):
    rows = []
    for name, evaluated in result["baselines"].items():
        m = evaluated["metrics"]
        rows.append(dict(model=name, seed="—", parameters=0, v2t=m["video_to_text"]["recall_at_1"],
                         t2v=m["text_to_video"]["recall_at_1"], shuffled_v2t=None))
    for run in result["runs"]:
        m = run["test"]["metrics"]
        rows.append(dict(model=run["model"], seed=run["seed"], parameters=run["training"]["trainable_parameters"],
                         v2t=m["video_to_text"]["recall_at_1"], t2v=m["text_to_video"]["recall_at_1"],
                         shuffled_v2t=run["shuffled"]["metrics"]["video_to_text"]["recall_at_1"]))
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rate = lambda value: "—" if value is None else f"{value*100:.1f}%"
    protocol = result["protocol"]
    is_kth = protocol["dataset_sources"] == ["KTH_official_d1_three_actions"]
    source_note = ("KTH 三类动作、d1 场景、整段 AVI；使用官方人物划分。文本由动作标签转成固定英文描述。"
                   if is_kth else "来源：" + ", ".join(protocol["dataset_sources"]))
    lines = ["# 视频文本检索：固定协议对照", "", source_note,
             "根据当前标注定义正例；类别模板实验不等同于自由视频描述或动作顺序基准。", "",
             f"记录数量：{protocol['records_by_split']}；人物数量：{protocol['groups_by_split']}。", "",
             "| 模型 | 种子 | 可训练参数 | 视频→文本 R@1 | 文本→视频 R@1 | 打乱帧后 视频→文本 R@1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['seed']} | {r['parameters']} | {rate(r['v2t'])} | {rate(r['t2v'])} | {rate(r['shuffled_v2t'])} |")
    lines.extend(["", "## 跨种子汇总（视频→文本 R@1）", ""])
    for name in protocol["models"]:
        values = [r["v2t"] for r in rows if r["model"] == name]
        lines.append(f"- {name}：均值 {rate(statistics.mean(values))}，种子间标准差 {rate(statistics.pstdev(values))}。")
    lines.extend(["", "## 固定展示的失败案例", "",
                  "使用预先列出的第一个 GRU 种子，不按测试分数挑选运行。以下为其误检：", ""])
    example = next(r for r in result["runs"] if r["model"] == "gru")
    failures = [p for p in example["test"]["predictions"] if not p["correct"]]
    for p in failures[:12]:
        lines.append(f"- {p['id']}：真实 {p['semantic_id']} → 预测 {p['predicted_semantic_id']}。")
    if not failures:
        lines.append("- 该种子在此小测试集没有误检；不能据此声称广泛泛化。")
    lines.extend(["", "## 解释边界", "",
                  "- MLP 对帧顺序不敏感，参数量与 GRU 接近，用于控制模型容量因素。",
                  "- 测试集不用于选择训练轮数或超参数；每个运行都由验证集选择检查点。",
                  "- 同一人物的所有动作仅处于一个 split；未将同源视频随机拆到多个集合。",
                  "- 三类文本候选下 R@5/R@10 没有区分价值，因此主表仅展示 R@1。",
                  "- 文本→视频有多个正确视频；其 R@1 不等于视频分类准确率。",
                  "- 没有显式动作顺序对照标注，因此不计算顺序准确率。打乱帧仅为诊断。",
                  "- 同一划分的三个种子不代表跨数据集置信区间；小测试集结果应谨慎外推。",
                  "- 结果提升还需更大数据、更多动作和背景变化验证；如 GRU 不优于 MLP，应如实保留该结论。", "",
                  ("数据：[KTH 官方来源](https://www.csc.kth.se/cvap/actions/)，仅非商业使用；引用 Schuldt、Laptev、Caputo，ICPR 2004。" if is_kth else "数据来源见 protocol.json。"), ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    table = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in
                    (r["model"], r["seed"], r["parameters"], rate(r["v2t"]), rate(r["t2v"]), rate(r["shuffled_v2t"]))) + "</tr>" for r in rows)
    # Escape all dataset-derived strings; no remote scripts or media are used.
    detail = "\n".join(lines[lines.index("## 跨种子汇总（视频→文本 R@1）"):])
    html = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>视频文本实验</title><style>body{font:16px/1.8 system-ui;background:#f3f6f8;color:#182d3d;margin:32px auto;max-width:1100px;padding:20px}table{border-collapse:collapse;background:white;width:100%}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}pre{white-space:pre-wrap;font:inherit}.notice{background:#fff0d5;padding:16px;border-radius:10px}.table{overflow:auto}a{color:#176d59}</style><h1>视频文本检索 · 固定协议实验</h1><p class="notice">'+escape(source_note)+'</p><p>'+escape(str(protocol['records_by_split']))+'</p><div class="table"><table><tr><th>模型</th><th>种子</th><th>可训练参数</th><th>视频→文本 R@1</th><th>文本→视频 R@1</th><th>打乱帧后</th></tr>'+table+'</table></div><pre>'+escape(detail)+'</pre><p><a href="results.json">完整结果</a> · <a href="protocol.json">预先固定的协议</a> · <a href="summary.csv">结果表</a></p>'
    preview = failure_preview(result, failures, output)
    if preview:
        html += '<h2>固定 GRU 种子的误检样例</h2><p>每行从原视频均匀抽取四帧；标签来自数据集，预测来自该次测试。</p><img alt="Misclassified videos with true and predicted labels" style="max-width:100%" src="failures.jpg">'
    (output / "report.html").write_text(html, encoding="utf-8")


def failure_preview(result, failures, output):
    """Show the first six errors, without selecting visually favorable examples."""
    if not failures:
        return False
    from PIL import Image, ImageDraw, ImageFont
    from .data import load_video_frames, resolve_video_path
    cache = load_cache(result["protocol"]["cache"])
    records = {record["id"]: record for record in cache["records"]}
    manifest = Path(cache["provenance"]["manifest_path"])
    selected = failures[:6]
    canvas = Image.new("RGB", (680, len(selected) * 172), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    for row, failure in enumerate(selected):
        record = records[failure["id"]]
        path = resolve_video_path(manifest, record["video_path"])
        frames = load_video_frames(path, frames=4)
        draw.text((12, row * 172 + 8), f"{failure['id']}: {failure['semantic_id']} -> {failure['predicted_semantic_id']}",
                  fill="#183345", font=font)
        for column, frame in enumerate(frames):
            frame.thumbnail((160, 128))
            canvas.paste(frame, (12 + column * 166, row * 172 + 36))
    canvas.save(output / "failures.jpg", quality=90)
    return True
