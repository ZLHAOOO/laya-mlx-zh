# laya-zh — 让 Laya 决策模型真正说中文

[English](#english) · 中文微调版 Laya：消息路由 / 紧急度判断 / 优先级打分 / 检索关键词选择，四个决策头一体。在 Apple Silicon 上毫秒级推理。

**TL;DR**：Laya 官方 multilingual checkpoint 自称支持 51 种语言，但中文决策任务实测接近失效（下方有对照数据）。我们用 9,000+ 条全合成中文数据做 heads-only 微调（8GB Mac、100 分钟、不动 encoder），把中文决策准确率拉到 0.85~0.90。数据集 + 训练脚本 + 权重全部开源。

---

## 为什么做这个

[Laya](https://github.com/NandhaKishorM/laya)（Convai Innovations）是一个很棒的轻量决策模型：mmBERT encoder + 轻量决策头，专为 agent 的"该不该打断 / 消息怎么分派 / 优先级多高"这类小决策设计。但它官方 checkpoint 的中文能力是这样：

| 输入 | 官方 multilingual 的判断 | 正确答案 |
|---|---|---|
| "官网出 bug 了用户全登不上，赶紧处理" | spam (0.96) ❌ | work-urgent |
| "服务器刚重启完，监控又报警了" | work-normal (0.61) ❌ | work-urgent |
| "周末一起吃饭不" | ✅ personal | personal |
| "帮我看看那个项目部署的事" | spam ❌ | work-normal |

（2026-09 实测，官方最新 checkpoint；简单句能过，真实工作语句大面积失灵。）

官方重点在英语和多语言广度，短期内不会做中文垂直深耕。所以我们自己动手——微调后：

| 任务 | 微调前(官方) | 微调后 (laya-zh v4) |
|---|---|---|
| 消息路由 route (4类) | ~0.25* | **0.854** |
| 紧急度判断 interrupt | — | **0.902** |
| 优先级打分 priority (4档) | — | **0.878** |

*官方 multilingual 在中文 route 任务上接近随机猜（4 类）。\*
评测集：41 条人工手写的分布外中文样本（脱敏版随仓库发布，见 `data/test.jsonl`）。

## 快速开始（推理）

需要 Apple Silicon (M1/M2/M3/M4)：

```bash
pip install laya-mlx
```

```python
import laya_mlx as laya

agent = laya.load("./weights/ckpt-zh-v4-mlx")  # 见 weights/README.md 下载

result = agent.predict(
    "服务器宕机了赶紧处理",
    {"route": {"type": "choice",
               "instructions": "Which category does this message belong to?",
               "criteria": {"work-urgent": "needs action now",
                             "work-normal": "work without urgency",
                             "personal": "friends and family, casual chat",
                             "spam": "ads and scams"}},
     "interrupt": {"type": "noul",
                    "instructions": "Does this message need immediate attention right now?"}},
)
print(result["answers"]["route"])   # {'choice': 'work-urgent', 'confidence': 0.84}
print(result["answers"]["interrupt"])  # {'noul': 0.82, 'confidence': ...}
```

单条推理 ~27ms（常驻模型，M1 实测）；权重 614MB。

## 训练数据

`data/` 下全部开源，**全合成**（模板+扰动生成，无任何真实用户数据）：

| 文件 | 条数 | 内容 |
|---|---|---|
| `train.jsonl` | 9,240 | route / interrupt / priority / relevance / kw_select / ticket 六任务混合 |
| `train_kw.jsonl` | 1,004 | 关键词选择头专项数据（好搜索词 vs 口语噪声词） |
| `val.jsonl` | 989 | 验证集（同分布） |
| `test.jsonl` | 41 | **人工手写**分布外测试集 |

数据格式：每行一条标注 `{"state": ..., "labels": {"任务名": [soft targets]}}`。软标签梯度（0.9/0.6/0.05）用于教模型校准置信度，而不是只会 0/1。

生成数据的脚本也在仓库（`gen_data.py` / `gen_open.py` / `gen_kw.py`），可以改模板造你自己的场景数据。

## 复现训练

在 8GB 内存 M1 Mac 上实测 ~100 分钟（heads-only 模式）：

```bash
pip install laya torch transformers safetensors huggingface_hub
python train.py --epochs 8 --lr 5e-4 --bs 48
```

关键配置（也是我们踩坑后的结论）：

- **heads-only**：冻结 encoder，只训决策头 + type embedding（~15M 参数）。8G 内存可跑，效果已够；想更高再解冻 encoder（LoRA r=8-16 on q/v 是社区验证过的稳妥点）
- **lr 5e-4**：heads-only 下 1e-4 不收敛，5e-4 起飞
- **bs 48**：MPS 上比默认快
- **训练用 PyTorch（官方 `laya` 包），推理用 MLX（`laya-mlx`）**：MLX 推理快 2-4 倍，`laya_mlx` 的转换管线直接吃训练产物

## 已知限制（诚实清单）

- 测试集只有 41 条手写样本，准确率 ±0.10 置信区间，别当精确数字看
- 输入上限 1024 tokens（继承官方配置），长文档请先摘要/分块。实测把 1 万字文档直接塞进去，相关性判断会失效——**短输入（标题+摘要）才是它的舒适区**
- 训练数据全合成，真实分布偏移存在（我们用真实使用数据持续观察中）
- 软标签校准未做 ECE 审计，置信度数值仅供参考排序，别当概率用

## 生态位

- [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)：原版框架 + 英文/多语言 checkpoint（本项目的基座，Apache-2.0）
- [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)：社区 MLX 移植（本项目的推理引擎）
- **laya-zh（本项目）**：中文垂直层——数据集 + 微调权重 + 训练 recipe，与上游零冲突

如果你在做中文 agent 的消息分流、通知打断、记忆检索排序——这东西就是为你造的。

<a name="english"></a>
## English

Laya is a lightweight decision model (mmBERT + small decision heads) for agent micro-decisions. Its official multilingual checkpoint claims 51-language support, but **fails badly on Chinese decision tasks** (routing a real work message as "spam" with 0.96 confidence). We fine-tuned it with 9k+ fully synthetic Chinese samples (heads-only, 100 min on an 8GB M1 Mac), lifting Chinese decision accuracy from near-random to 0.85–0.90.

This repo open-sources: the synthetic dataset (sanitized), training scripts, eval sets, and MLX weights. See the Chinese sections above for details — or just run the Quick Start. Base model: [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) (Apache-2.0).

## License

Apache-2.0（继承上游 Laya 协议）
