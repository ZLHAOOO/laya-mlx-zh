#!/usr/bin/env python3
"""kw_select 头数据生成器 v2: 教 laya 从口语化 query 里挑"有搜索价值的词"
标注定义(v2 修正): 词的价值 = 它查 BM25 的返回质量(top1 分数分档), 不依赖特定目标块
  top1 ≥ 8 → P=0.9 | 3~8 → P=0.6 | 无结果或 <3 → P=0.05
模板双槽: 好词槽(真有料) + 噪声词槽(真出现在 query 里, 标负) → 教 laya 排除口语噪声

⚠️ 参考实现: 依赖一个已建好的 BM25 索引(本仓库作者用自有知识库的 recall 引擎)。
适配你自己的环境：把 load_docs()/bm25_score() 换成你的文档集+BM25 实现即可。
不想自建数据？直接用仓库自带的 data/train_kw.jsonl (1000 条已生成样本)。
"""
import os, sys, json, random, sqlite3, pickle, subprocess, time
from collections import OrderedDict

random.seed(20260924)
AGENT = os.path.expanduser("~/.pi/agent")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import perturb

STOP = set("""帮我 看看 那个 这个 什么 怎么 一下 问题 事情 东西 感觉 现在 可以 需要 应该 还是 就是
我们 你们 他们 自己 没有 是不是 怎么样 为什么 记得 说过 上次 之前 找一下 相关 哪里 谁的 来着 样子 的 了 吗 呢""".split())

def load_docs():
    db = sqlite3.connect(f"{AGENT}/memory/.recall-index.db")
    meta = dict(db.execute("SELECT k, v FROM meta"))
    db.close()
    return pickle.loads(meta["docs"])

_cache = OrderedDict()
def bm25_score(word, topk=5):
    """词查 BM25 返回 (top1分数, 命中数)。LRU 缓存"""
    if word in _cache:
        return _cache[word]
    if len(_cache) > 4000:
        _cache.popitem(last=False)
    r = subprocess.run([f"{AGENT}/bin/recall", "q", word, "--top", str(topk), "--json"],
                       capture_output=True, text=True)
    try:
        hits = json.loads(r.stdout[r.stdout.index("{"):])["recall"]
        val = (hits[0]["score"] if hits else 0.0, len(hits))
    except Exception:
        val = (0.0, 0)
    _cache[word] = val
    return val

def label_for(word):
    s, n = bm25_score(word)
    if s >= 8:
        p = 0.9
    elif s >= 3:
        p = 0.6
    else:
        p = 0.05
    return [round(1 - p, 2), p]

TEMPLATES2 = [
    "帮我看看{w}的{n}", "{w}的{n}在哪", "之前那个{w}的{n}怎么样了", "找下{w}相关的{n}",
    "{w}这个{n}你来跟进", "有{w}的{n}吗", "上次说的{w}的{n}", "整理下{w}的{n}",
    "{w}是啥来着", "我记得有{w}的{n}", "老板问{w}的{n}", "{w}现在什么{n}",
]
NOISE = ["问题", "情况", "资料", "记录", "要点", "内容", "方案", "数据", "信息", "结果"]

def title_words(text):
    import jieba
    ws = [w.strip() for w in jieba.cut(text, cut_all=False)
          if len(w.strip()) >= 2 and w.strip() not in STOP]
    seen, out = set(), []
    for w in ws:
        if w not in seen:
            seen.add(w); out.append(w)
    return out

def main():
    docs = load_docs()
    random.shuffle(docs)
    blocks = docs[:400]
    rows, seen_keys = [], set()
    stat = {"hi": 0, "mid": 0, "neg": 0}
    t0 = time.time()
    for bi, d in enumerate(blocks):
        cands = title_words(f"{d['title']}：{d['gist']}")[:3]
        if not cands:
            continue
        good = random.sample(cands, min(len(cands), random.choice([1, 2])))
        noise = random.choice(NOISE)
        for qw in good:
            q_raw = random.choice(TEMPLATES2).format(w=qw, n=noise)
            if random.random() < 0.3:
                q_raw = perturb(q_raw)
            for word in dict.fromkeys([qw, noise]):  # 好词+噪声词都要标(噪声真在句中)
                lbl = label_for(word)
                p = lbl[1]
                stat["hi" if p >= 0.9 else ("mid" if p > 0.05 else "neg")] += 1
                key = f"{q_raw}|{word}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                rows.append({"state": q_raw, "cluster": "kw", "q": word, "labels": {"kw_select": lbl}})
        if bi % 50 == 0:
            print(f"  {bi}/{len(blocks)}, {len(rows)} 行, {time.time()-t0:.0f}s", flush=True)
    random.shuffle(rows)
    n = len(rows); nv = int(n * 0.12)
    with open(f"{OUT}/train_kw.jsonl", "w") as f:
        for x in rows[:-nv]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(f"{OUT}/val_kw.jsonl", "w") as f:
        for x in rows[-nv:]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"✓ train {n-nv} / val {nv}; 分布 {stat}; 独立词查询 {len(_cache)}; 耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
