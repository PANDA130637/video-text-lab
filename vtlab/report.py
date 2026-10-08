"""Local, self-contained reports with honest provenance and cached-query exploration."""
from __future__ import annotations

import base64
from collections import defaultdict
import csv
from html import escape
import io
import json
from pathlib import Path
import statistics


def percent(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def gallery(result: dict) -> dict:
    """Use the first configured GRU seed, never the best test result, for browsing."""
    import numpy as np
    import torch
    from PIL import Image
    from .training import load_model

    run = next(r for r in result["runs"] if r["model"] == "gru")
    checkpoint = Path(result["cache"]).parent / f"gru_{run['seed']}" / "best.pt"
    model, _ = load_model(checkpoint)
    with np.load(result["cache"], allow_pickle=False) as cache:
        records = json.loads(str(cache["metadata"].item()))
        video_features = cache["video_features"]
        text_features = cache["text_features"]
        selected = [i for i, record in enumerate(records) if record["split"] == "test"]
        unique = {}
        for index in selected:
            unique.setdefault((records[index]["semantic_id"], records[index]["text"]), index)
        queries = list(unique.values())
        with torch.no_grad():
            scores = (model.encode_text(torch.from_numpy(text_features[queries])) @
                      model.encode_video(torch.from_numpy(video_features[selected])).T).numpy()
    videos = []
    for index in selected:
        record = records[index]
        path = Path(record["video_path"])
        if not path.is_absolute():
            path = Path(result["manifest"]).parent / path
        thumbnail = None
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            images = []
            for frame in files[:16]:
                with Image.open(frame) as original:
                    image = original.convert("RGB")
                    image.thumbnail((160, 160))
                    images.append(image)
            if images:
                buffer = io.BytesIO()
                images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:],
                               duration=130, loop=0)
                thumbnail = "data:image/gif;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        videos.append({"id": record["id"], "text": record["text"],
                       "semantic_id": record["semantic_id"], "thumbnail": thumbnail})
    return {"seed": run["seed"], "videos": videos,
            "queries": [{"text": records[index]["text"], "semantic_id": records[index]["semantic_id"],
                         "scores": scores[i].tolist()} for i, index in enumerate(queries)]}


def write_report(result: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows, groups = [], defaultdict(list)
    for run in result["runs"]:
        normal, shuffled = run["normal"]["metrics"], run["shuffled"]["metrics"]
        row = {"model": run["model"], "seed": run["seed"],
               "v2t_r1": normal["video_to_text"]["recall_at_1"],
               "t2v_r1": normal["text_to_video"]["recall_at_1"],
               "order_accuracy": normal["order_query_accuracy"],
               "shuffled_order_accuracy": shuffled["order_query_accuracy"]}
        rows.append(row)
        groups[row["model"]].append(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = ["# 视频文本检索：合成控制实验", "",
                "**数据：程序生成的移动色块；视觉特征：颜色质心；文本特征：独热编码。**",
                "这是训练与评估流程的正确性验证，不是 CLIP 或真实视频理解效果。", "",
                "所有模型只用 train 训练、val 选择检查点；正序和逆序共享场景分组，测试场景独立。", "",
                "| 模型 | 随机种子 | 视频→文本 R@1 | 文本→视频 R@1 | 成对方向判断 | 打乱帧后方向判断 |",
                "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {row['model']} | {row['seed']} | {percent(row['v2t_r1'])} | {percent(row['t2v_r1'])} | {percent(row['order_accuracy'])} | {percent(row['shuffled_order_accuracy'])} |")
    markdown.extend(["", "## 跨随机种子的汇总", ""])
    summaries = []
    for name, values in groups.items():
        valid = [v["order_accuracy"] for v in values if v["order_accuracy"] is not None]
        mean, std = (statistics.mean(valid), statistics.pstdev(valid)) if valid else (None, None)
        summaries.append((name, mean, std))
        markdown.append(f"- {name} 成对方向判断：{percent(mean)}，标准差 {percent(std)}（{len(valid)} 个训练种子，同一数据划分）。")
    markdown.extend(["", "## 如何理解", "",
                     "- 成对方向判断比较同一场景两个相反描述的分数；近似平分记作 0.5。",
                     "- 平均池化无法区分完全反序的同一组帧，因此成对方向判断应为 50%。",
                     "- 文本候选去重，同一语义的多个视频均是正确检索结果；不同检索方向的候选规模不同。",
                     "- 打乱帧是诊断；掉分不等于理解真实动作语义。",
                     "- 多个训练种子的标准差不是跨数据集的置信区间。",
                     "- 本实验用于确认模型能学到受控的方向信息，不能写成真实视频模型性能。", "",
                     "## 下一步", "",
                     "录制或整理真实视频 → 按拍摄场次分组 → 冻结 CLIP 提取特征 → 重复相同对照。",
                     "参考项目 README 和 docs/real-data.md。完整逐样本结果保存在每次运行的 test.json。", ""])
    (output / "report.md").write_text("\n".join(markdown), encoding="utf-8")

    table_rows = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in
                      [r["model"].upper(), r["seed"], percent(r["v2t_r1"]), percent(r["t2v_r1"]),
                       percent(r["order_accuracy"]), percent(r["shuffled_order_accuracy"])]) + "</tr>" for r in rows)
    cards = "".join(f'<div class="card"><span>{escape(name.upper())} · 成对方向判断</span><strong>{percent(mean)}</strong><small>跨种子标准差 {percent(std)}</small></div>'
                    for name, mean, std in summaries)
    data = json.dumps(gallery(result), ensure_ascii=False).replace("<", "\\u003c")
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视频文本检索 · 实验记录</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#172231;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1080px;margin:auto;padding:48px 24px}h1{font-size:34px;margin:8px 0}h2{font-size:22px;margin-top:32px}p{color:#526070}small{color:#657386}.tag{color:#176d59;font-weight:700;letter-spacing:.1em}.notice{background:#fff5da;border-left:4px solid #e0af34;padding:16px 20px;border-radius:8px}.cards{display:flex;gap:18px;margin:24px 0}.card{background:white;border:1px solid #dce2e8;border-radius:14px;padding:20px;flex:1}.card span,.card small{display:block}.card strong{font-size:36px;color:#176d59}table{width:100%;border-collapse:collapse;background:white}td,th{text-align:left;padding:13px;border-bottom:1px solid #e3e8ed}th{font-size:13px;background:#eaf0f4}.table{overflow:auto;border-radius:12px}select{width:100%;padding:13px;border:1px solid #cbd4de;border-radius:9px;font:inherit;background:white}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-top:18px}.video{background:white;padding:14px;border:1px solid #dce2e8;border-radius:12px}.video img{display:block;width:100%;image-rendering:auto;border-radius:8px;margin:8px 0}.video p{font-size:13px;margin:5px 0}.correct{color:#176d59}.wrong{color:#a44b35}footer{margin-top:36px;border-top:1px solid #dce2e8;padding-top:16px;color:#657386;font-size:13px}a{color:#176d59}@media(max-width:600px){main{padding:24px 14px}h1{font-size:26px}.cards{flex-direction:column}}
</style><main><div class="tag">VIDEO × TEXT / LOCAL LAB</div><h1>视频文本检索 · 实验记录</h1>
<p>平均池化与时序模型，在相同数据和特征上的对照。</p><div class="notice"><b>合成控制实验</b>：移动色块 + 颜色质心 + 独热文本。此报告验证训练与评估流程，不代表 CLIP 或真实视频理解效果。</div>
<div class="cards">__CARDS__</div><h2>固定测试集 · 全部训练种子</h2><div class="table"><table><thead><tr><th>模型</th><th>种子</th><th>视频→文本 R@1</th><th>文本→视频 R@1</th><th>成对方向判断</th><th>打乱帧后</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p>检查点由验证集选择。成对判断中平分记 0.5；打乱帧仅为辅助诊断。多个种子共享同一数据划分。</p>
<h2>查看一次实际检索</h2><p id="gallery-note"></p><select id="query" aria-label="选择文字描述"></select><div class="gallery" id="gallery"></div>
<h2>接下来验证什么</h2><p>换成真实录制的视频及冻结 CLIP 特征，检查背景、物体、动作顺序与描述质量的影响。当前合成结果不能外推到真实视频。</p>
<footer>所有内容保存在本地 · 完整数值：<a href="results.json">results.json</a> · <a href="summary.csv">summary.csv</a> · <a href="report.md">实验说明</a></footer></main>
<script>const data=__DATA__;const select=document.getElementById('query');document.getElementById('gallery-note').textContent='固定使用第一个 GRU 训练种子 '+data.seed+'，显示测试集前 5 项；描述来自缓存，页面不调用在线模型。';data.queries.forEach((q,i)=>{const o=document.createElement('option');o.value=i;o.textContent=q.text;select.appendChild(o)});function render(){const q=data.queries[Number(select.value)];const target=document.getElementById('gallery');target.replaceChildren();q.scores.map((s,i)=>({s,i})).sort((a,b)=>b.s-a.s).slice(0,5).forEach((entry,rank)=>{const v=data.videos[entry.i];const card=document.createElement('div');card.className='video';const title=document.createElement('b');title.textContent='#'+(rank+1)+' · '+entry.s.toFixed(3);card.appendChild(title);if(v.thumbnail){const image=document.createElement('img');image.src=v.thumbnail;image.alt=v.text;card.appendChild(image)}const text=document.createElement('p');text.textContent=v.text;card.appendChild(text);const label=document.createElement('p');label.className=v.semantic_id===q.semantic_id?'correct':'wrong';label.textContent=v.semantic_id===q.semantic_id?'匹配语义':'不匹配';card.appendChild(label);target.appendChild(card)})}select.addEventListener('change',render);render();</script></html>"""
    html = html.replace("__CARDS__", cards).replace("__ROWS__", table_rows).replace("__DATA__", data)
    (output / "report.html").write_text(html, encoding="utf-8")
    return {"markdown_report": str(output / "report.md"), "interactive_report": str(output / "report.html")}
