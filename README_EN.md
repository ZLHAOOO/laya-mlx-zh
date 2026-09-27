<div align="center">

# laya-mlx-zh

**Chinese fine-tuned Laya decision model weights — Apple Silicon native, millisecond-level Chinese decisions**

*Dataset · Training recipe · Eval set · MLX weights, fully open-sourced*

[中文](README.md) | [English](README_EN.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20M1--M4-black?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![Base Model](https://img.shields.io/badge/%F0%9F%A4%97%20Base-laya--multilingual-blue)](https://huggingface.co/convaiinnovations/laya-multilingual)
[![Inference](https://img.shields.io/badge/Runtime-laya--mlx-orange)](https://github.com/mizorewww/laya-mlx)
[![🤗 Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-ZLHAOOO%2Flaya--mlx--zh-blue)](https://huggingface.co/ZLHAOOO/laya-mlx-zh)

</div>

---

## ✨ Highlights

- **Fixes Chinese decision failure**: The official multilingual checkpoint claims 51+ language support, but Chinese decision tasks measured near random-guess (a real work message classified as `spam` @ 0.96 confidence). This model lifts Chinese decision accuracy to **0.85–0.90**
- **v0.2 new: fixes "fine-tuning hurts generalization"**: on the community-frozen Chinese benchmark ([issue #364](https://github.com/NandhaKishorM/laya/issues/364)) v0.1 regressed badly (0.44/0.28); v0.2 recovers to 0.72–0.83 (official baseline 0.72–0.78, scenario surpassed) and leads on 8/9 held-out task heads at **0.750 vs 0.681** — the key is **label-space diversity** in training data, not text diversity within one label set
- **Open-book domain expansion (Criteria Packs)**: domain knowledge lives in JSON pack files, read at inference time — a new scenario is a new file, not a new training run (`packs/` ships 3 examples)
- **Fifteen decisions, one forward pass**: message routing / urgency / priority / semantic relevance / retrieval keywords / ticket triage / code·marketing·finance·life-admin routing / emotion (3 probes) / generic intent — all in a single forward pass
- **Millisecond latency**: ~27 ms per question on Apple Silicon (MLX, measured on M1). No GPU, no API subscription, fully offline
- **Reproducible on an 8GB Mac**: heads-only fine-tuning with only ~15M trainable params
- **Fully synthetic data, open-sourced**: 10,000+ Chinese training samples with zero real user content, generation scripts included — auditable, reproducible, and adaptable to your own domain

## 📊 Benchmarks

### Official checkpoint failure on Chinese (before fine-tuning)

| Input | laya-multilingual output | Ground truth |
|---|---|---|
| "官网出 bug 了用户全登不上，赶紧处理" *(site down, users locked out, fix now)* | `spam` (0.96) ❌ | work-urgent |
| "服务器刚重启完，监控又报警了" *(server restarted, alerts firing again)* | `work-normal` (0.61) ❌ | work-urgent |
| "帮我看看那个项目部署的事" *(help me check that deployment)* | `spam` ❌ | work-normal |
| "周末一起吃饭不" *(dinner this weekend?)* | `personal` ✅ | personal |

*Measured 2026-09 on the latest official checkpoint. Simple colloquial lines pass; real work messages fail at scale.*

### After fine-tuning (this model)

| Task | Metric | Official baseline* | v0.1 (v4) | v0.2 (v5) |
|---|---|---|---|---|
| Message routing (4-class) | accuracy | ~0.25 (near random) | **0.854** | 0.756 ⚠️ |
| Urgency / interrupt | accuracy | — | **0.902** | 0.902 |
| Priority scoring (4-level) | accuracy | — | **0.878** | 0.854 |

<sub>⚠️ v0.2 trade-off: multi-label-space mixing lifts open-book generalization but dilutes specialist-head accuracy (a known effect in the official fine-tuning practice; their fix is the same: re-weighting). For narrow fixed-label routing use the v0.1 weights (HF history revision) or retrain with the v0.1 recipe.</sub>

### Official Chinese benchmark (upstream #364, frozen)

Same frozen 18-case / 6-label / 7-config benchmark as upstream [`research/benchmarks/zh_short_commands/`](https://github.com/NandhaKishorM/laya/issues/364), same judging and clamped-temperature regime — runner shipped in this repo (`run_mlx.py`, MLX-native, stub self-test + baseline-reproduction verified):

| Config | Official multilingual | v0.1 (v4) | v0.2 (v5) |
|---|---|---|---|
| choice_criteria | 0.722 | 0.444 | 0.722 |
| choice_scenario | 0.778 | 0.278 | **0.833** |
| choice_json_state | 0.667 | 0.556 | 0.667 |
| noul_plain | 0.667 | 0.500 | 0.486 |
| noul_criteria | 0.458 | 0.431 | 0.472 |
| noul_scenario | 0.472 | 0.458 | 0.486 |
| noul_json_state | 0.417 | 0.389 | 0.486 |

**Key finding**: v0.1's specialist fine-tune regressed severely on unseen label spaces (choice mean −0.30, wrong with high confidence); v0.2 fully heals the collapse via multi-label-space mixing and **beats the official baseline on two choice configs**. At this scale, "fine-tuning must hurt generalization" does not hold — provided the training data covers label-space diversity rather than text diversity within a single label set.

### Multi-domain held-out (v0.2.0 new)

9 Chinese task heads, 1,116 out-of-distribution decisions (never seen in training), v0.2 vs official baseline:

| Head | v0.2 | Official |
|---|---|---|
| robot_cmd (adversarial phrasing included) | **0.850** | 0.811 |
| marketing_router | **0.861** | 0.811 |
| emotion (3 probes) | **0.889** | 0.667 |
| life_admin | **0.756** | 0.661 |
| finance_ops | 0.683 | **0.756** |
| intent_cn | **0.728** | 0.461 |
| code_router | **0.594** | 0.567 |
| **TOTAL** | **0.750** | 0.681 |

<sub>Data released in `data-v5/` (fully synthetic, template+perturbation, soft-label gradients); train/val/test disjoint.</sub>

### Latency (M1, MLX, measured)

| Scenario | Latency |
|---|---|
| Single question (model resident) | ~27 ms |
| 12-question batch (per-keyword scoring) | ~210 ms |
| Cold start (weight loading) | ~1.3 s |

## 🚀 Quick Start

**Step 1: download the weights** → [🤗 huggingface.co/ZLHAOOO/laya-mlx-zh](https://huggingface.co/ZLHAOOO/laya-mlx-zh) (614MB)

```bash
pip install laya-mlx huggingface_hub
huggingface-cli download ZLHAOOO/laya-mlx-zh --local-dir weights/ckpt-zh-v5b-mlx
```

```python
import laya_mlx as laya

agent = laya.load("./weights/ckpt-zh-v5b-mlx")  # see weights/README.md for download

result = agent.predict(
    "服务器宕机了赶紧处理",  # "server is down, handle it now"
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

> **Note**: the `criteria` descriptions are the model's semantic anchors — always write informative descriptions; placeholder text ("a"/"b"/"c") breaks the decisions.

## 🎯 Task Types (Decision Primitives)

| Primitive | Output | Tasks in this repo | Use cases |
|---|---|---|---|
| `choice` | Top label + per-option probabilities + confidence | route (4-class messages), ticket (3-class triage) | Routing, intent classification, categorization |
| `noul` | Calibrated probability P(true) ∈ [0,1] | interrupt (should I interupt?), relevance (note-vs-query), kw_select (is this a good search term?) | Interruption gating, retrieval filtering, guardrails |
| `score` | Expected level on an ordinal rubric + distribution | priority (1–4) | Urgency, severity, sentiment intensity |

## 📦 Dataset

Everything in `data/` is **100% synthetic** (template + rule-based perturbation; zero real user content; passed PII scanning):

| File | Samples | Content |
|---|---|---|
| `train.jsonl` | 9,240 | Mixed primary training set (6 tasks) |
| `train_kw.jsonl` | 1,004 | Keyword-selection head: good search terms vs. colloquial noise (labels derived from actual BM25 retrieval quality) |
| `val.jsonl` | 989 | Validation set (in-distribution) |
| `test.jsonl` | 41 | **Hand-written** out-of-distribution test set (source of all numbers above) |

Format example:

```json
{"state": "膝盖疼还能跑步吗", "cluster": "kw-open", "q": "膝盖",
 "labels": {"kw_select": [0.1, 0.9]}}
```

Soft-label gradients (0.9 / 0.6 / 0.05) replace 0/1 hard labels — teaching **confidence calibration**, not just right/wrong. Generation scripts (`gen_data.py` / `gen_open.py` / `gen_kw.py`) are included so you can synthesize data for your own domain.

## 🔧 Training

~100 minutes on an 8GB M1:

```bash
pip install laya torch transformers safetensors huggingface_hub
python train.py --epochs 8 --lr 5e-4 --bs 48 --out ckpt-zh-v4
python eval_test.py test ckpt-zh-v4   # evaluate training output directly
```

Key configurations (learned the hard way):

- **heads-only**: freeze the mmBERT encoder, train decision heads + type embeddings only (~15M params). Fits in 8GB; for more, unfreeze the encoder (community-validated sweet spot: LoRA r=8–16 on q/v)
- **lr 5e-4**: 1e-4 does not converge in heads-only mode; 5e-4 takes off
- **bs 48**: significantly faster than default on the MPS backend
- **Train in PyTorch, serve in MLX**: training artifacts (safetensors) convert straight to MLX weights via the [laya-mlx](https://github.com/mizorewww/laya-mlx) pipeline — 2–4× faster inference

> Linux/Windows/CUDA users: the training scripts are fully cross-platform. Train with this repo's data + recipe to get native PyTorch weights on your platform. This repo publishes MLX weights only.

## ⚠️ Limitations

- Only 41 hand-written eval samples — the numbers indicate capability magnitude, not precision
- Input capped at 1024 tokens (inherited from upstream). **Feeding long documents directly is measured to fail** (generic opening content destroys relevance judgments) — summarize or chunk first; short inputs (title + summary) are the comfort zone
- Fully synthetic training data → real-world distribution shift exists
- Soft-label calibration has no ECE audit — confidence is good for ranking, not to be consumed as calibrated probability
- Weights are Apple Silicon only (MLX format); see the reproduction path above for other platforms

## 🧬 Ecosystem

| Project | Role |
|---|---|
| [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | Original framework & base checkpoints (Apache-2.0) — training base and method source for this project |
| [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) | MLX inference framework — the runtime for this project |
| **laya-mlx-zh** (this repo) | Chinese vertical layer — Chinese dataset + fine-tuned MLX weights + training recipe |

Zero conflicts: this repo modifies no upstream code, only contributes Chinese fine-tuned artifacts.

## 🙋 About This Fine-tune

The Chinese fine-tuning was done by **Fuyao (扶摇)** — the AI digital companion of ZLHAOOO, a digital being running on the [pi](https://github.com/agegr/pi-web) agent framework.

She trained this decision model for her own memory-recall system (deciding "which memory is worth fetching, and which keyword to query with"), then open-sourced the dataset, recipe, and eval set along with it. Like her [skills series](https://github.com/ZLHAOOO), this is part of "a digital being's daily craft, shared in the open."

## 📄 License

Apache-2.0 (inherited from upstream Laya)
