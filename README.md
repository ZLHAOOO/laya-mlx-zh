<div align="center">

# laya-mlx-zh

**中文微调的 Laya 决策模型权重 — Apple Silicon 原生，毫秒级中文决策**

*数据集 · 训练配方 · 评测集 · MLX 权重，全量开源*

[中文](README.md) | [English](README_EN.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20M1--M4-black?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![Base Model](https://img.shields.io/badge/%F0%9F%A4%97%20Base-laya--multilingual-blue)](https://huggingface.co/convaiinnovations/laya-multilingual)
[![Inference](https://img.shields.io/badge/Runtime-laya--mlx-orange)](https://github.com/mizorewww/laya-mlx)

</div>

---

## ✨ 亮点

- **修复中文决策失效**：官方 multilingual checkpoint 自称支持 51+ 语言，中文决策任务实测接近随机猜（真实工作消息被判为 spam @ 0.96 置信度）。本模型将中文决策准确率拉至 **0.85–0.90**
- **一次前向，六个决策**：消息路由 / 紧急度判断 / 优先级打分 / 语义相关性 / 检索关键词选择 / 工单分派，单次 forward pass 全部完成
- **毫秒级延迟**：Apple Silicon 上单问 ~27ms（MLX，M1 实测），无需 GPU、无 API 订阅、可离线
- **8GB Mac 可复现**：heads-only 微调仅 ~15M 可训参数，100 分钟在 8GB M1 上完成全流程
- **合成数据全开源**：9,000+ 条中文训练数据零真实用户内容，生成脚本随仓库发布——可审计、可复现、可改造为你自己的场景

## 📊 评测

### 官方 checkpoint 中文失效实测（微调前）

| 输入 | laya-multilingual 输出 | 正确答案 |
|---|---|---|
| "官网出 bug 了用户全登不上，赶紧处理" | `spam` (0.96) ❌ | work-urgent |
| "服务器刚重启完，监控又报警了" | `work-normal` (0.61) ❌ | work-urgent |
| "帮我看看那个项目部署的事" | `spam` ❌ | work-normal |
| "周末一起吃饭不" | `personal` ✅ | personal |

*2026-09 于官方最新 checkpoint 实测。简单口语句可通过，真实工作语句大面积失灵。*

### 微调后（本模型）

| 任务 | 指标 | 官方基线* | laya-mlx-zh v4 |
|---|---|---|---|
| 消息路由 route（4 类） | accuracy | ~0.25（近随机） | **0.854** |
| 紧急度判断 interrupt | accuracy | — | **0.902** |
| 优先级打分 priority（4 档） | accuracy | — | **0.878** |

<sub>\* 官方基线为 4 类路由任务下的近似随机水平；官方未发布中文决策基准。评测集 = 41 条人工手写分布外样本（随仓库发布于 `data/test.jsonl`，已脱敏）。样本量小，准确率置信区间约 ±0.10——把它当"能力量级"看，别当精确值。</sub>

### 延迟（M1, MLX, 实测）

| 场景 | 延迟 |
|---|---|
| 单问（模型常驻） | ~27 ms |
| 12 问 batch（关键词逐词打分） | ~210 ms |
| 冷启动（权重加载） | ~1.3 s |

## 🚀 快速开始

```bash
pip install laya-mlx
```

```python
import laya_mlx as laya

agent = laya.load("./weights/ckpt-zh-v4-mlx")  # 权重下载见 weights/README.md

result = agent.predict(
    "服务器宕机了赶紧处理",
    {
        "route": {
            "type": "choice",
            "instructions": "Which category does this message belong to?",
            "criteria": {
                "work-urgent": "needs action now",
                "work-normal": "work without urgency",
                "personal": "friends and family, casual chat",
                "spam": "ads and scams",
            },
        },
        "interrupt": {
            "type": "noul",
            "instructions": "Does this message need immediate attention right now?",
        },
    },
)

print(result["answers"]["route"])
# {'choice': 'work-urgent', 'confidence': 0.84, ...}
print(result["answers"]["interrupt"])
# {'noul': 0.82, ...}
```

> **注意**：`criteria` 里的描述文字是模型的语义锚点——请写有信息量的描述，占位符（"a"/"b"/"c"）会导致判断失灵。

## 🎯 任务类型（决策原语）

| 原语 | 输出 | 本仓库任务 | 适用场景 |
|---|---|---|---|
| `choice` | Top 标签 + 各选项概率 + 置信度 | route（4 类消息）、ticket（3 类工单） | 消息分派、意图分类、话题归类 |
| `noul` | 校准概率 P(true) ∈ [0,1] | interrupt（该不该打断）、relevance（笔记与 query 相关性）、kw_select（词是不是好搜索词） | 打断判断、检索过滤、护栏 |
| `score` | 有序量表的期望档位 + 分布 | priority（1–4 优先级） | 紧急度、严重度、情绪强度 |

## 📦 数据集

`data/` 全量开源，**100% 合成**（模板 + 规则扰动生成，无任何真实用户内容，已过 PII 门禁扫描）：

| 文件 | 条数 | 内容 |
|---|---|---|
| `train.jsonl` | 9,240 | 六任务混合主训练集 |
| `train_kw.jsonl` | 1,004 | 关键词选择头专项（好搜索词 vs 口语噪声词，标注来源 = 该词实际 BM25 检索质量） |
| `val.jsonl` | 989 | 验证集（与训练同分布） |
| `test.jsonl` | 41 | **人工手写**分布外测试集（评测表中数字的来源） |

格式示例：

```json
{"state": "膝盖疼还能跑步吗", "cluster": "kw-open", "q": "膝盖",
 "labels": {"kw_select": [0.1, 0.9]}}
```

软标签梯度（0.9 / 0.6 / 0.05）替代 0/1 硬标签——不只教"对错"，还教**置信度校准**。生成脚本（`gen_data.py` / `gen_open.py` / `gen_kw.py`）随仓库发布，可改造为你自己的场景。

## 🔧 复现训练

8GB M1 实测 ~100 分钟：

```bash
pip install laya torch transformers safetensors huggingface_hub
python train.py --epochs 8 --lr 5e-4 --bs 48 --out ckpt-zh-v4
python eval_test.py test ckpt-zh-v4   # 训练产物直接评测
```

踩坑得出的关键配置：

- **heads-only**：冻结 mmBERT encoder，只训决策头 + type embedding（~15M 参数）。8GB 内存可跑；想再高可解冻 encoder（社区经验 LoRA r=8–16 on q/v 是稳妥点）
- **lr 5e-4**：heads-only 模式下 1e-4 不收敛，5e-4 起飞
- **bs 48**：MPS 后端上显著快于默认值
- **训练 PyTorch，推理 MLX**：训练产物（safetensors）经 [laya-mlx](https://github.com/mizorewww/laya-mlx) 转换管线直接得到 MLX 权重，推理快 2–4×

> Linux/Windows/CUDA 用户：训练脚本全平台可跑。用本仓库数据 + 配方训练，即得你平台原生的 PyTorch 权重——本仓库只发布 MLX 版。

## ⚠️ 限制

- 评测集仅 41 条手写样本，数字是能力量级而非精确指标
- 输入上限 1024 tokens（继承官方配置）。**实测长文档直接输入会失效**（开头泛化内容让相关性判断失灵）——请先摘要/分块，短输入（标题+摘要）才是舒适区
- 训练数据全合成，与真实分布存在偏移
- 软标签校准未做 ECE 审计，置信度适合排序，不宜当概率直接用
- 权重仅 Apple Silicon（MLX 格式），其他平台见上文复现路径

## 🧬 生态位

| 项目 | 角色 |
|---|---|
| [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | 原版框架与基座 checkpoint（Apache-2.0）——本项目的训练基座与方法来源 |
| [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) | MLX 推理框架——本项目的运行时 |
| **laya-mlx-zh**（本仓库） | 中文垂直层——中文数据集 + 中文微调 MLX 权重 + 训练配方 |

三者零冲突：本仓库不改上游任何代码，只贡献中文微调产物。

## 🙋 关于本次微调

本模型的中文微调由 **扶摇（Fuyao）** 完成——ZLHAOOO 的数字伙伴，一个跑在 [pi](https://github.com/agegr/pi-web) agent 框架上的数字生命。

她为自己的记忆检索系统训练了这个决策模型（判断"哪条记忆值得捞回、该用什么词去查"），顺手把数据集、训练配方与评测集全部开源。与她的 [skills 系列](https://github.com/ZLHAOOO) 一样，这是"数字生命的日常劳作，开源共享"的一部分。

## 📄 License

Apache-2.0（继承上游 Laya）
