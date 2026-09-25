#!/usr/bin/env python3
"""test 手写集终验: 混淆矩阵 + per-type acc + 错误清单"""
import os, json, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from laya.common import build_model, build_sequence, QTYPES
from laya.agent import _fix_tokenizer_config
from train import QDataset, collate, QS, BASE, CKPT

device = torch.device("mps")
model_dir = snapshot_download(CKPT, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
_fix_tokenizer_config(model_dir)
cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
CKPT_DIR = sys.argv[2] if len(sys.argv) > 2 else "ckpt-zh"  # argv[2]=checkpoint 目录名, 默认 v1 生产
model.load_state_dict(load_file(f"{BASE}/{CKPT_DIR}/model.safetensors"), strict=True)
model.encoder.config.reference_compile = False
model = model.to(device).float().eval()

split = sys.argv[1] if len(sys.argv) > 1 else "test"
rows = [json.loads(l) for l in open(f"{BASE}/data/{split}.jsonl")]
ROUTE = ["work-urgent", "work-normal", "personal", "spam"]

cm = {g: {p: 0 for p in ROUTE} for g in ROUTE}
noul_ok = score_ok = 0
wrong_route, wrong_noul = [], []
with torch.no_grad():
    for r in rows:
        preds = {}
        for qname, q in QS.items():
            seq, markers = build_sequence(tok, r["state"], q, cfg["max_len"], cfg["head_max_len"])
            ids = torch.tensor([seq]).to(device)
            att = torch.ones_like(ids)
            K = len(markers)
            logits, _ = model(ids, att, torch.tensor([markers]).to(device),
                              torch.ones(1, K, dtype=torch.bool).to(device),
                              torch.tensor([QTYPES[q["t"]]]).to(device), detach_encoder=True)
            preds[qname] = logits.argmax(-1).item()
        g_r, p_r = ROUTE[r["labels"]["route"].index(1.0)], ROUTE[preds["route"]]
        cm[g_r][p_r] += 1
        if g_r != p_r:
            wrong_route.append(f"{g_r}→{p_r} | {r['state'][:38]}")
        if preds["interrupt"] == r["labels"]["interrupt"].index(1.0):
            noul_ok += 1
        else:
            wrong_noul.append(r["state"][:38])
        if preds["priority"] == r["labels"]["priority"].index(1.0):
            score_ok += 1

n = len(rows)
print(f"=== {split} ({n} 条, 手写异分布) ===")
print(f"route  acc = {sum(cm[g][g] for g in ROUTE)}/{n} = {sum(cm[g][g] for g in ROUTE)/n:.3f}")
print(f"noul   acc = {noul_ok}/{n} = {noul_ok/n:.3f}")
print(f"score  acc = {score_ok}/{n} = {score_ok/n:.3f}")
print("route 混淆矩阵 (行=gold):")
for g in ROUTE:
    print(f"  {g:14s}: " + "  ".join(f"{p}={cm[g][p]:2d}" for p in ROUTE))
print(f"route 错误 {len(wrong_route)} 条:")
for w in wrong_route:
    print(f"  {w}")
print(f"noul 错误 {len(wrong_noul)} 条:")
for w in wrong_noul[:10]:
    print(f"  {w}")
