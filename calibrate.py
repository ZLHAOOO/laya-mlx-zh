#!/usr/bin/env python3
"""温度表重校准（held-out 拟合）——治"过度自信"（0.94 置信答错题类问题）。

方法：在 val/test 数据上收集每 (qtype, k) bucket 的 raw logits，对每 bucket
网格搜索最小化 NLL 的温度 T。数据只用 val_v5/test_v5/test_v5_extra——
**绝不碰训练集切片**（#186 对官方 typed-decisions notebook 的批评）。

用法: python calibrate.py ckpt-zh-v5b [--apply]
  默认只打印新旧对照；--apply 才写回 ckpt 的 rl_agent_config.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from laya.common import build_model, build_sequence, QTYPES
from laya.agent import _fix_tokenizer_config
from laya_mlx.common import temp_bucket, TEMP_MIN, TEMP_MAX
from train import QS, BASE

CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "ckpt-zh-v5b"
APPLY = "--apply" in sys.argv

model_dir = snapshot_download("convaiinnovations/laya-multilingual",
                              allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
if not os.path.exists(os.path.join(model_dir, "rl_agent_config.json")) and os.path.exists(os.path.join(model_dir, "multilingual")):
    model_dir = os.path.join(model_dir, "multilingual")
_fix_tokenizer_config(model_dir)
cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder") if os.path.exists(os.path.join(model_dir, "encoder")) else None)
model.load_state_dict(load_file(f"{BASE}/{CKPT_DIR}/model.safetensors"), strict=True)
model.encoder.config.reference_compile = False
model = model.to("mps").float().eval()

# 收集 held-out 数据（val_v5 + test_v5 + test_v5_extra；全部未参与训练）
rows = []
for split in ("val_v5", "test_v5", "test_v5_extra"):
    p = f"{BASE}/data/{split}.jsonl"
    if os.path.exists(p):
        rows += [json.loads(l) for l in open(p)]
print(f"held-out rows: {len(rows)}")

buckets = {}  # bucket -> {logits: [np.array], targets: [np.array]}
with torch.no_grad():
    for r in rows:
        for qname, target in r["labels"].items():
            q = r.get("qs", {}).get(qname) or QS[qname]
            seq, markers = build_sequence(tok, r["state"], q, cfg["max_len"], cfg["head_max_len"])
            ids = torch.tensor([seq]).to("mps"); att = torch.ones_like(ids)
            K = len(markers)
            logits, _ = model(ids, att, torch.tensor([markers]).to("mps"),
                              torch.ones(1, K, dtype=torch.bool).to("mps"),
                              torch.tensor([QTYPES[q["t"]]]).to("mps"), detach_encoder=True)
            lg = logits[0, :K].float().cpu().numpy()
            tg = np.zeros(K); tg[:len(target)] = target[:K]
            b = temp_bucket(QTYPES[q["t"]], K)
            buckets.setdefault(b, {"lg": [], "tg": []})
            buckets[b]["lg"].append(lg); buckets[b]["tg"].append(tg)

def nll(T, LG, TG):
    z = LG / max(T, 1e-3)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    return -np.log(np.clip((p * TG).sum(axis=1), 1e-9, None)).mean()

new_T = {}
print(f"\n{'bucket':16s} {'n':>5s} {'旧T':>7s} {'新T':>7s} {'NLL旧':>8s} {'NLL新':>8s}")
for b, d in sorted(buckets.items()):
    LG = np.array(d["lg"]); TG = np.array(d["tg"])
    old_T = float(cfg.get("temperature_by_options", {}).get(b, 1.0))
    grid = np.arange(0.5, 3.01, 0.05)
    nlls = [nll(T, LG, TG) for T in grid]
    best = float(grid[int(np.argmin(nlls))])
    nll_old, nll_new = nll(old_T, LG, TG), nll(best, LG, TG)
    new_T[b] = round(best, 4)
    print(f"{b:16s} {len(LG):5d} {old_T:7.3f} {best:7.3f} {nll_old:8.4f} {nll_new:8.4f}")

if APPLY:
    cfg["temperature_by_options"] = {**cfg.get("temperature_by_options", {}), **new_T}
    out = f"{BASE}/{CKPT_DIR}/rl_agent_config.json"
    json.dump(cfg, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n✓ 已写回 {out}")
else:
    print("\n(预览模式：加 --apply 写回)")
