#!/usr/bin/env python3
"""v5 新头 held-out 评测：test_v5.jsonl 逐头 acc（支持 --baseline 对照原版 multilingual）

用法: python eval_v5.py [ckpt目录名] [split] [--baseline]
默认: ckpt-zh-v5 test_v5
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from laya.common import build_model, build_sequence, QTYPES
from laya.agent import _fix_tokenizer_config
from train import QS, BASE

CKPT_DIR = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "ckpt-zh-v5"
SPLIT = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "test_v5"
BASELINE = "--baseline" in sys.argv

model_dir = snapshot_download("convaiinnovations/laya-multilingual",
                              allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
if not os.path.exists(os.path.join(model_dir, "rl_agent_config.json")) and os.path.exists(os.path.join(model_dir, "multilingual")):
    model_dir = os.path.join(model_dir, "multilingual")
_fix_tokenizer_config(model_dir)
cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder") if os.path.exists(os.path.join(model_dir, "encoder")) else None)
if BASELINE:
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)  # 原版权重
else:
    model.load_state_dict(load_file(f"{BASE}/{CKPT_DIR}/model.safetensors"), strict=True)
model.encoder.config.reference_compile = False
model = model.to("mps").float().eval()

rows = [json.loads(l) for l in open(f"{BASE}/data/{SPLIT}.jsonl")]
per, wrong = {}, []
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
            pred = logits.argmax(-1).item()
            gold = target.index(max(target))
            ok = int(pred == gold)
            per.setdefault(qname, [0, 0]); per[qname][1] += 1; per[qname][0] += ok
            if not ok:
                wrong.append(f"{qname}: {r['state'][:30]}")
print(f"=== {SPLIT} @ {'原版multilingual(baseline)' if BASELINE else CKPT_DIR} ===")
for qn, (ok, n) in sorted(per.items()):
    print(f"  {qn:18s} acc={ok}/{n} = {ok/n:.3f}")
tot_ok = sum(v[0] for v in per.values()); tot_n = sum(v[1] for v in per.values())
print(f"  TOTAL acc={tot_ok}/{tot_n} = {tot_ok/tot_n:.3f}")
