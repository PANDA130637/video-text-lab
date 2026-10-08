# Video Text Lab

一个面向低算力环境的视频文本检索实验项目。项目研究一个具体问题：在冻结视觉与文本编码器、只使用 CPU 训练轻量模块的条件下，视频帧的时间组织是否能改善视频与文本的匹配？

项目定位是一个可复现的课程/研究型原型，不是视频生成模型，也不声称完成了大规模预训练或分布式训练。

目前包含两个明确分开的入口：

- **合成控制实验**：程序生成移动色块，使用颜色质心和独热文本特征；不下载模型即可验证训练、评估和反序对照。
- **真实视频入口**：读取 MP4 等视频或图片序列，冻结 CLIP ViT-B/32 提取特征；可使用已完成的 KTH 子集实验，也可接入自己的视频及标注。

合成结果只用于检查训练、缓存、评估和帧顺序诊断，不能作为真实视频理解或视频生成模型的效果。

已完成的本机验证、实际数值和限制见 [验证记录](docs/validation.md)。

**真实视频实验也已完成**：[KTH 实验说明](docs/kth-experiment.md)，结果页面为 [公开报告](reports/kth-8f/report.html)。75 段真实视频、官方人物划分、三个训练种子，并增加了与 GRU 参数量接近的无时序 MLP 对照。

实验协议、数据限制、结果表和复现命令见 [KTH 实验说明](docs/kth-experiment.md)。可直接查看已经整理的 [实验报告](reports/kth-8f/report.html)。

真实视频实验使用 KTH Actions 的 boxing、handclapping、handwaving 三类动作，仅取 d1 场景的 75 段整段 AVI。官方人物划分用于 train/val/test，测试集包含 27 段、9 名独立人物；文本是三个类别的固定英文描述，不是开放域字幕。

| 方法 | 测试视频→文本 Recall@1 均值 |
|---|---:|
| 冻结 CLIP 平均池化 | 33.3% |
| 冻结 CLIP 中间单帧 | 37.0% |
| mean + projection | 69.1% |
| MLP + projection | 72.8% |
| GRU + projection | 75.3% |

GRU 相对 MLP 只高约 2.5 个百分点，而且测试规模较小。因此这些数值只描述本实验协议下的结果，不能外推为稳定的时序优势、开放域视频理解或视频生成效果。

## 视频质量评估原型

项目还提供 `quality` 命令，用冻结 CLIP 特征估计文本一致性、相邻帧稳定性和闪烁比例，并输出带 bootstrap 95% 区间、动作分组和扰动对照的启发式综合分数。对 KTH 测试集，干净视频的平均分数为 74.23（95% 区间 74.03–74.43）；跨视频交替替换帧的扰动使相邻帧余弦从 0.9397 降到 0.8743，闪烁比例从 0.021 升到 0.339。这个对照说明指标能发现明显的局部不稳定，但它还没有经过人类偏好校准。

```powershell
python run.py quality --cache data/kth/features_8f.npz --output runs/kth-quality --split test
```

指标定义和限制见 [视频质量评估说明](docs/quality-evaluation.md)，结果见 [质量评估报告](reports/kth-quality/quality.html)。

## 直接运行

项目可以使用已有 Python 环境运行。缺少的依赖位于项目 `.deps`，`run.py` 自动加载；也可以在自己的虚拟环境中安装依赖。

```powershell
python run.py doctor
python run.py test
python run.py demo
```

也可以用 `powershell -ExecutionPolicy Bypass -File .\run.ps1 demo` 运行。这里的执行策略参数只作用于该次进程，不修改系统策略。设置 `$env:VTLAB_PYTHON` 可以指定另一个已有 Python。

`demo` 默认生成 80 个场景、160 段视频、8 个动作描述类别，每段 8 帧；对 mean 和 GRU 各训练 3 个随机种子。编码器冻结，训练只有小型时序模块和投影层，默认使用 CPU 的 4 个线程。

产物位于 `runs/demo/`：

| 文件 | 内容 |
|---|---|
| `report.html` | 离线交互报告，可选描述查看实际检索结果 |
| `report.md` | 结果、解释与限制 |
| `summary.csv` / `results.json` | 全部种子的数值和来源信息 |
| `features.npz` | 帧特征、文本特征、元数据、输入内容指纹 |
| `mean_17/best.pt` 等 | 由验证集选择的检查点 |
| 各运行目录的 `history.json` | 每轮训练与验证记录 |
| 各运行目录的 `test.json` | 测试指标及逐样本预测 |

更小的验证：

```powershell
python run.py demo --output runs/quick --groups 24 --epochs 10 --seeds 42
```

同一输出目录可以按相同数据设置重跑。改变数据生成参数时使用新的输出目录，防止混用旧数据；训练输出会更新，所以保留重要实验时也应另取目录。

## 检索一个描述

```powershell
python run.py search --cache runs/demo/features.npz --checkpoint runs/demo/gru_17/best.pt --query 'A red dot moves from left to right.' --top-k 5
```

合成编码器只接受预设的 8 条英文描述；真实 CLIP 缓存支持新英文查询。报告页面为了能离线打开，只展示缓存中的描述，未运行在线模型。

## 接入真实视频

先阅读 [真实数据采集与实验协议](docs/real-data.md)，复制 `examples/manifest.example.csv` 并填写真实文件路径。示例 CSV 的文件不存在，不能直接作为数据运行。

```powershell
python run.py validate data/real/manifest.csv
python run.py extract --manifest data/real/manifest.csv --cache data/real/features.npz --encoder clip --frames 8 --batch-size 8
python run.py baseline --cache data/real/features.npz --output runs/real-frozen-clip.json
python run.py train --cache data/real/features.npz --output runs/real-mean-42 --model mean --seed 42
python run.py train --cache data/real/features.npz --output runs/real-gru-42 --model gru --seed 42
python run.py evaluate --cache data/real/features.npz --checkpoint runs/real-gru-42/best.pt --output runs/real-gru-42/test.json
```

第一次 CLIP 提取会下载 `openai/clip-vit-base-patch32` 权重，缓存保存在项目 `.cache`；免费，不需要云 GPU。提取耗时请先用少量样本实测，再决定规模。CPU 可运行，核显 `--device xpu` 需要已经支持 XPU 的 PyTorch 与驱动；本项目 CPU 包并不自动具备核显加速能力。

下载完成后，`extract` 与 `search` 可加 `--offline`，只使用本地模型缓存，避免联网检查。当前本机的官方 CLIP 权重已经下载。

如果换了一台机器，优先选已有 Python 3.11–3.13 和已有依赖。仅在缺少依赖时，可装入项目目录而不改全局环境：

```powershell
python -m pip install --target .deps --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install --target .deps -r requirements.txt -r requirements-real.txt
```

若已有 PyTorch，可直接运行 `python run.py doctor` 后补缺项；不要重复安装不同后端的 torch 到同一目录。

## 实验设计

1. 同一场景的正向与反向视频共享 `group_id` 和 `pair_id`，所有同源内容只能在一个 split 中。
2. mean 和 GRU 使用完全相同的帧/文本特征。mean 也有可训练投影层，以免把“训练过与未训练过”的差异误当成时序收益。
   真实 CLIP 实验还应保留 `baseline` 的完全不训练结果，检查微调投影是否破坏了原有跨模态对齐。
3. 对比学习将相同 `semantic_id` 的样本视为多个正例，避免重复描述互相成为负例。
4. 仅用 train 更新参数，val 选择检查点，最后单独评估 test。
5. 视频→文本候选按 `(semantic_id, text)` 去重；文本→视频中所有语义匹配视频都是正例。
6. 成对方向判断比较正确与相反描述，近似平分记 0.5。平均池化对完全反序的帧应为 50%。
   只有显式填写 `pair_id` 的对照才计算该指标；共享 `group_id` 不自动代表动作顺序对照。
7. 打乱帧只是辅助诊断；分数降低不能独立证明真实动作理解。

合成实验中的 mean 与 GRU 参数量不同。真实 KTH 实验已增加无时序 MLP 基线，参数量与 GRU 相差约 0.16%，用于检查容量因素；完整协议见 `docs/kth-experiment.md`。

## 代码阅读顺序

1. `vtlab/data.py`：数据分组、抽帧、缓存指纹。
2. `vtlab/encoders.py`：合成控制编码器与冻结 CLIP。
3. `vtlab/models.py`：mean / MLP / GRU、归一化投影。
4. `vtlab/training.py`：多正例损失、训练与验证、检查点。
5. `vtlab/metrics.py`：检索与成对判断的定义。
6. `vtlab/report.py`：完整结果与本地交互展示。
7. `vtlab/kth.py` 与 `vtlab/benchmark.py`：真实数据准备、固定实验协议、多种子对照与错误案例。
8. `vtlab/quality.py`：文本一致性、时序稳定性、闪烁诊断和扰动对照。

阅读实验时可以重点关注：为什么需要多正例；为什么要按来源分组；冻结特征省了什么、限制了什么；为什么平均池化不能区分反序；如何区分参数容量、数据偏差与时序信息的贡献。

参考：[CLIP 官方实现](https://github.com/openai/CLIP)、[Hugging Face CLIP 文档](https://huggingface.co/docs/transformers/model_doc/clip)、[PyTorch Intel GPU 文档](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)。
