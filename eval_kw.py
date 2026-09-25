#!/usr/bin/env python3
"""kw_select 头定向验收: 已知失败案例救援测试 (需配合 BM25 环境, 参考实现)
场景: 多词口语化 query 整句查 BM25 常因"多词打分稀释"漏掉正解;
验收: laya 选词 → 词分别查 BM25 → 正解块能否捞回 (多路召回合并)

⚠️ 本脚本依赖本地 BM25 服务(recall 引擎)与私有知识库, 开源仓库里作为方法参考。
CASES 与 bm25() 需按你自己的文档集适配; 权重加载部分(ckpt-zh-v4-mlx)可直接复用。
"""
import os, sys, json, subprocess
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
import laya_mlx as laya
agent = laya.load(os.path.join(HERE, "weights", "ckpt-zh-v4-mlx"))

import jieba

CASES = [
    # (query, 正解路径关键词, 说明) — 示例沿用作者知识库的脱敏别名, 换成你自己的 case
    ("老板作息 上下班时间", "profile.md", "已知失败案例: 多词稀释"),
    ("帮我看看那个部署的事，官网那边", "website.md", "混合场景"),
    ("知识库服务 定价的事", "kb.md", "BM25 强项对照"),
    ("之前说的公司上班的事", "work|note", "别名词"),
]

def bm25(q, k=3):
    r = subprocess.run([os.path.expanduser("~/.pi/agent/bin/recall"), "q", q, "--top", str(k), "--json"],
                       capture_output=True, text=True)
    try:
        return [(d["path"], d["title"][:30]) for d in json.loads(r.stdout[r.stdout.index("{"):])["recall"]]
    except Exception:
        return []

for query, expect_key, note in CASES:
    print(f"\n=== {query}  ({note}) ===")
    whole = bm25(query)
    whole_hit = any(expect_key.split("|")[0] in p for p, _ in whole)
    print(f"  整句直查: {'✓命中' if whole_hit else '✗miss'} {[p.split('/')[-1] for p, _ in whole[:2]]}")
    # jieba 候选词(模拟真实管线)
    words = [w.strip() for w in jieba.cut(query) if len(w.strip()) >= 2 and w.strip() not in
             {"帮我", "看看", "那个", "之前", "说的", "一下", "的事"}]
    scored = []
    for w in words:
        a = agent.predict(query, {"kw": {"type": "noul",
              "instructions": f"Is this word a good search term to recall relevant memory? W: {w}"}})["answers"]["kw"]
        scored.append((a["noul"], w))
    scored.sort(reverse=True)
    picked = [w for s, w in scored[:2] if s > 0.5]
    print(f"  jieba 候选: {words}")
    print(f"  laya 选词: {[(w, round(s,2)) for s, w in scored]} → 选 {picked}")
    merged, seen = [], set()
    for w in picked or words[:1]:
        for p, t in bm25(w, 3):
            if p not in seen:
                seen.add(p); merged.append((p, t))
    merged_hit = any(expect_key.split("|")[0] in p for p, _ in merged[:4])
    print(f"  选词分查合并: {'✓命中' if merged_hit else '✗miss'} {[p.split('/')[-1] for p, _ in merged[:3]]}")
    verdict = "救援成功" if (not whole_hit and merged_hit) else ("本来就中" if whole_hit else "仍未解决")
    print(f"  → {verdict}")
