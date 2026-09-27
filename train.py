#!/usr/bin/env python3
"""laya-zh Phase 2: heads-only 微调训练脚本
- 冻结 encoder(mmBERT) + act_head, 只训 head/type_emb/scorer (8G M1 可跑)
- 损失 = soft-CE (strictly proper scoring rule; 有标注场景等价替代官方 RLCD policy gradient)
  # ponytail: heads-only 若 val 不达标(route acc<90%), 升级路径=LoRA on encoder (pip install peft)
- 产物: 完整 state_dict 存 model.safetensors(自包含), laya-mlx 转换管线可直接吃
"""
import os, json, math, time, random, argparse
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.common import build_model, build_sequence, QTYPES
from laya.agent import _fix_tokenizer_config

BASE = os.path.expanduser("~/.pi/agent/tools/laya-zh")
CKPT = "convaiinnovations/laya-multilingual"
QS = {
    "route": {"t": "choice", "ins": "Which category does this message belong to?",
              "crit": {"work-urgent": "needs action now: incidents, deadlines, complaints, outages",
                       "work-normal": "work matters without urgency: reports, notices, schedules",
                       "personal": "friends and family, casual chat, social plans",
                       "spam": "ads, scams, phishing, cold promotions"}},
    "interrupt": {"t": "noul", "ins": "Does this message need immediate attention right now?", "crit": {}},
    "priority": {"t": "score", "ins": "How urgent is this message?",
                 "crit": ["not urgent", "soon", "urgent", "critical"]},
    # v2: recall 语义重排头。ins 是前缀，query 在每行 r["q"]，QDataset 动态拼接
    "relevance": {"t": "noul", "ins": "Is this note relevant to the query? Q:", "crit": {}},
    # v3: 关键词选择头。ins 前缀 + 行内词 w
    "kw_select": {"t": "noul", "ins": "Is this word a good search term to recall relevant memory? W:", "crit": {}},
    # v4: 通用工单路由(开源场景, 官方 laya 的 billing/technical/sales 中文版)
    "ticket": {"t": "choice", "ins": "Which department should handle this ticket?",
               "crit": {"billing": "invoices, payments, refunds",
                            "technical": "bugs, outages, errors",
                            "sales": "pricing, plans, new purchases"}},
    # v5: 多标签空间混训（治 #364 实锤的 criteria 塌缩：让头学会读陌生选项而不是背词串）
    #     主描述与 gen_v5.py 的 QS_V5 一致；训练数据 27% 行携带措辞变体（行内 qs）
    "code_router": {"t": "choice", "ins": "这条消息该路由给哪个代码类技能？",
               "crit": {"bugfix": "排查修复：报错、崩溃、异常", "feature": "新功能开发：加功能、改交互",
                            "refactor": "重构优化：不改行为只改质量", "docs": "文档查询：API 用法、配置说明",
                            "deploy": "部署运维：构建、发布、环境", "chat": "与代码无关的消息"}},
    "marketing_router": {"t": "choice", "ins": "这条消息该路由给哪个营销类技能？",
               "crit": {"content": "内容创作：文案、图文、脚本", "seo": "流量获取：关键词、SEO、投放",
                            "data": "数据分析：转化、漏斗、复盘", "crm": "客户关系：社群、私聊、跟进",
                            "brand": "品牌定位：slogan、视觉、调性", "none": "与营销无关的消息"}},
    "finance_ops": {"t": "choice", "ins": "这条消息该路由给哪个财务类技能？",
               "crit": {"expense": "报销付款：报销、垫付、转账", "invoice": "发票：开票、收票、整理",
                            "budget": "预算管理：预算、成本、超支", "recon": "对账核对：账单、流水、差异",
                            "tax": "税务合规：报税、税率、合规", "none": "与财务无关的消息"}},
    "life_admin": {"t": "choice", "ins": "这条生活消息该交给哪个管家技能？",
               "crit": {"schedule": "日程安排：约会、会议、提醒", "shopping": "购物消费：想买、下单、退换",
                            "health": "健康：就医、用药、锻炼", "travel": "出行：通勤、订票、路线",
                            "home": "家务：保洁、维修、缴费", "chat": "纯闲聊，不用执行"}},
    "robot_cmd": {"t": "choice", "ins": "这条指令要求机器人做什么？",
               "crit": {"faster": "让机器人提速", "slower": "让机器人减速", "stop": "让机器人停下",
                            "left": "向左侧转向", "right": "向右侧转向", "none": "不是对机器人的运动指令"}},
    "intent_cn": {"t": "choice", "ins": "这条中文消息的主要意图是什么？",
               "crit": {"ask": "提问咨询：想知道某事", "do": "请求执行：让助手做某事",
                            "feedback": "反馈：确认、感谢、抱怨", "chat": "闲聊寒暄",
                            "spam": "广告推销骚扰", "other": "其他意图"}},
    "emotion_emo": {"t": "noul", "ins": "这条消息是否带有明显情绪（喜、怒、哀、焦虑）？", "crit": {}},
    "emotion_care": {"t": "noul", "ins": "这条消息是否需要安抚或共情回应？", "crit": {}},
    "emotion_urgent": {"t": "noul", "ins": "这条消息是否涉及需要立刻处理的情绪危机？", "crit": {}},
}

def load_rows(split, train_file="train.jsonl"):
    import glob
    # v5: 多标签空间混训合并数据；v4: --data 显式指定合并文件列表(避免 glob 双计), 兼容旧 glob 模式
    if split == "train" and os.path.exists(f"{BASE}/data/train_v5.jsonl") and os.environ.get("LAYA_V5") == "1":
        return [json.loads(l) for l in open(f"{BASE}/data/train_v5.jsonl")]
    if split == "val" and os.path.exists(f"{BASE}/data/val_v5.jsonl") and os.environ.get("LAYA_V5") == "1":
        return [json.loads(l) for l in open(f"{BASE}/data/val_v5.jsonl")]
    if split == "train" and os.path.exists(f"{BASE}/data/train_v4.jsonl") and os.environ.get("LAYA_V4") == "1":
        return [json.loads(l) for l in open(f"{BASE}/data/train_v4.jsonl")]
    if split == "val" and os.path.exists(f"{BASE}/data/val_v4.jsonl") and os.environ.get("LAYA_V4") == "1":
        return [json.loads(l) for l in open(f"{BASE}/data/val_v4.jsonl")]
    if split == "train":
        rows = []
        for f in sorted(glob.glob(f"{BASE}/data/train*.jsonl")):
            rows += [json.loads(l) for l in open(f)]
        return rows
    rows = []
    for f in sorted(glob.glob(f"{BASE}/data/{split}*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows

class QDataset(Dataset):
    """每条 state × 3 题摊平成 items: (ids, markers, qtype, target)"""
    def __init__(self, rows, tok, cfg):
        self.items = []
        for r in rows:
            for qname, target in r["labels"].items():
                # v5: 行内 qs 覆盖（criteria 措辞变体，逼模型读语义不背词串）
                q = r.get("qs", {}).get(qname) or QS[qname]
                if "q" in r:  # relevance 头：把行内 query 拼进指令
                    q = dict(q)
                    q["ins"] = q["ins"] + " " + r["q"]
                seq, markers = build_sequence(tok, r["state"], q, cfg["max_len"], cfg["head_max_len"])
                self.items.append({
                    "ids": seq, "markers": markers,
                    "qtype": QTYPES[q["t"]],
                    "target": target,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def collate(batch):
    B = len(batch)
    L = max(len(x["ids"]) for x in batch)
    K = max(len(x["markers"]) for x in batch)
    ids = torch.zeros(B, L, dtype=torch.long)
    att = torch.zeros(B, L, dtype=torch.long)
    mpos = torch.zeros(B, K, dtype=torch.long)
    mmask = torch.zeros(B, K, dtype=torch.bool)
    qtype = torch.zeros(B, dtype=torch.long)
    target = torch.zeros(B, K)
    tmask = torch.zeros(B, K, dtype=torch.bool)
    for i, x in enumerate(batch):
        n, k = len(x["ids"]), len(x["markers"])
        ids[i, :n] = torch.tensor(x["ids"]); att[i, :n] = 1
        mpos[i, :k] = torch.tensor(x["markers"]); mmask[i, :k] = True
        qtype[i] = x["qtype"]
        t = torch.tensor(x["target"])
        target[i, :len(t)] = t; tmask[i, :len(t)] = True
    return ids, att, mpos, mmask, qtype, target, tmask

def loss_fn(logits, target, tmask):
    logp = F.log_softmax(logits.masked_fill(~tmask, -1e4), -1)
    return -(target * logp.clamp_min(-30) * tmask).sum(-1).div(tmask.sum(-1).clamp(min=1)).mean()

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct = tot = 0
    per = {}
    for ids, att, mpos, mmask, qtype, target, tmask in dl:
        logits, _ = model(ids.to(device), att.to(device), mpos.to(device), mmask.to(device), qtype.to(device), detach_encoder=True)
        pred = logits.argmax(-1)
        gold = target.to(device).argmax(-1)  # 软标签兼容: 取最大概率位置
        ok = (pred == gold) & mmask.to(device).gather(1, pred[:, None]).squeeze(1)
        for qt, o in zip(qtype.tolist(), ok.tolist()):
            per.setdefault(qt, [0, 0]); per[qt][1] += 1; per[qt][0] += int(o)
            tot += 1; correct += int(o)
    model.train()
    return correct / max(tot, 1), {k: f"{a}/{b}" for k, (a, b) in sorted(per.items())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--smoke", action="store_true", help="小子集快速验证")
    ap.add_argument("--rev", default=None, help="checkpoint revision (默认 main; 传 sha 可用旧版, 如 MPS 不稳时退回 052592a)")
    ap.add_argument("--train", default="train.jsonl", help="训练数据文件名(data/ 下, 如 train_v2.jsonl)")
    ap.add_argument("--out", default="ckpt-zh", help="输出目录名(勿覆盖生产 ckpt-zh, v2 用 ckpt-zh-v2)")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")
    model_dir = snapshot_download(CKPT, revision=args.rev, allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    if "subfolder" in model_dir or not os.path.exists(os.path.join(model_dir, "rl_agent_config.json")):
        model_dir = os.path.join(model_dir, "multilingual")
    _fix_tokenizer_config(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder") if os.path.exists(os.path.join(model_dir, "encoder")) else None)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.config.reference_compile = False
    model = model.to(device).float()

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.act_head.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训参数 {n_train/1e6:.1f}M / 全量 {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    rows_tr, rows_va = load_rows("train", args.train), load_rows("val")
    if args.smoke:
        rows_tr, rows_va = rows_tr[:60], rows_va[:20]
    ds_tr = QDataset(rows_tr, tok, cfg)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(QDataset(rows_va, tok, cfg), batch_size=args.bs * 2, collate_fn=collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl_tr))

    acc0, per0 = evaluate(model, dl_va, device)
    print(f"baseline(未微调): val_acc={acc0:.3f} per_qtype={per0}")

    model.train()
    for ep in range(args.epochs):
        t0, run = time.time(), []
        for step, batch in enumerate(dl_tr):
            ids, att, mpos, mmask, qtype, target, tmask = [b.to(device) for b in batch]
            logits, _ = model(ids, att, mpos, mmask, qtype, detach_encoder=True)
            loss = loss_fn(logits, target, tmask)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run.append(loss.item())
            if step % 20 == 0:
                print(f"  ep{ep} step{step}/{len(dl_tr)} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
        acc, per = evaluate(model, dl_va, device)
        print(f"ep{ep} done: loss={sum(run)/len(run):.4f} val_acc={acc:.3f} per_qtype={per} ({time.time()-t0:.0f}s)")

    out = f"{BASE}/{args.out}"
    os.makedirs(out, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, f"{out}/model.safetensors")
    json.dump(cfg, open(f"{out}/rl_agent_config.json", "w"))
    for sub in ("tokenizer", "encoder"):
        src = os.path.join(model_dir, sub)
        if os.path.exists(src):
            os.system(f"cp -RL '{src}' '{out}/'")  # -L: snapshot 内是 symlink, 保留链接会断链
    print(f"✓ 已保存 {out}")

if __name__ == "__main__":
    main()
