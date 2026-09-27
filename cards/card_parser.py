#!/usr/bin/env python3
"""Decision Card 解析器 + lint（决策卡规范 v0.1，见 cards/SPEC.md）

.card.md → questions dict（与 Agent.predict 的 questions 参数同构）
--lint：校验 frontmatter / 类型 / criteria / token 预算（本地 build_sequence 试算）
"""
import os
import re
import sys


def parse_card(path):
    """解析 .card.md → (meta, questions, errors, warnings)"""
    errors, warnings = [], []
    text = open(path, encoding="utf-8").read()
    # --- frontmatter ---
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    meta = {}
    if not m:
        errors.append("缺少 frontmatter（文件需以 --- 开头的元数据块）")
        text_body = text
    else:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text_body = text[m.end():]
        for k in ("card", "version", "description"):
            if k not in meta:
                errors.append(f"frontmatter 缺必填字段: {k}")
    # --- Questions 区 ---
    qm = re.search(r"^## Questions\s*$", text_body, re.M)
    if not qm:
        errors.append("缺少 '## Questions' 区")
        return meta, {}, errors, warnings
    body = text_body[qm.end():]
    # 截掉下一个 '## ' 一级附注区（如有）
    nxt = re.search(r"^## (?!#)", body[3:], re.M)
    if nxt:
        body = body[: nxt.start() + 3]
    questions = {}
    sections = re.split(r"^### (.+)$", body, flags=re.M)[1:]  # [标题, 内容, 标题, 内容...]
    for i in range(0, len(sections) - 1, 2):
        head, content = sections[i].strip(), sections[i + 1]
        hm = re.match(r"^(\S+)\s*[（(]\s*(choice|noul|score)\s*[）)]\s*$", head)
        if not hm:
            errors.append(f"问题标题格式错：'{head}'（应为 '### id （type）'）")
            continue
        qid, qtype = hm.group(1), hm.group(2)
        lines = [l for l in content.splitlines() if l.strip()]
        instructions = lines[0].strip() if lines else ""
        if not instructions or instructions.startswith("-"):
            errors.append(f"[{qid}] 缺问题句（instructions）——### 标题后第一行应是自然语言问题")
            continue
        items = []
        for l in lines[1:]:
            lm = re.match(r"^-\s+(.+)$", l.strip())
            if lm:
                items.append(lm.group(1).strip())
        if qtype == "choice":
            crit = {}
            for it in items:
                if ":" not in it and "：" not in it:
                    errors.append(f"[{qid}] choice 列表项需 '标签: 描述'：{it[:30]}")
                    continue
                lab, desc = re.split(r"[:：]", it, maxsplit=1)
                lab, desc = lab.strip(), desc.strip()
                if lab in crit:
                    errors.append(f"[{qid}] 标签重复: {lab}")
                if len(desc) < 4:
                    warnings.append(f"[{qid}] 标签 '{lab}' 描述过短（'{desc}'）——写判断依据，别写占位符")
                crit[lab] = desc
            if len(crit) < 2:
                errors.append(f"[{qid}] choice 至少需要 2 个选项")
                continue
            questions[qid] = {"type": "choice", "instructions": instructions, "criteria": crit}
        elif qtype == "noul":
            q = {"type": "noul", "instructions": instructions}
            if items:
                d = {}
                for it in items:
                    if ":" in it or "：" in it:
                        k, v = re.split(r"[:：]", it, maxsplit=1)
                        d[k.strip()] = v.strip()
                if set(d) >= {"true", "false"}:
                    q["criteria"] = {"true": d["true"], "false": d["false"]}
            questions[qid] = q
        else:  # score
            if len(items) < 2:
                errors.append(f"[{qid}] score 需要至少 2 个有序档位")
                continue
            questions[qid] = {"type": "score", "instructions": instructions, "criteria": items}
    return meta, questions, errors, warnings


def lint_budget(path, questions, agent=None):
    """token 预算试算。有 laya agent 用真 tokenizer；无则返回提示。"""
    out = []
    if agent is None:
        return [(qid, "跳过 token 试算（需 laya 环境）") for qid in questions]
    from laya_mlx.common import build_sequence, render_options
    cfg = agent.cfg
    for qid, q in questions.items():
        try:
            internal = agent._to_internal(q)
            ids, markers = build_sequence(agent.tok, "试", internal,
                                          cfg.get("max_len", 512), cfg.get("head_max_len", 192))
            if len(markers) != len(render_options(internal)):
                out.append((qid, f"✗ 超出 head_max_len={cfg.get('head_max_len', 192)}：候选被截断，删减描述或减少选项"))
            else:
                room = cfg.get("max_len", 512) - len(ids)
                out.append((qid, f"✓ 前缀 {len(ids)}/{cfg.get('head_max_len', 192)} token，state 余量 ≈{int(room/0.68)}字"))
        except Exception as e:
            out.append((qid, f"✗ 试算失败: {e}"))
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path or not os.path.exists(path):
        print("用法: python card_lint.py <卡文件.card.md>"); sys.exit(1)
    meta, questions, errors, warnings = parse_card(path)
    print(f"卡: {meta.get('card', '?')} v{meta.get('version', '?')} — {meta.get('description', '')}")
    print(f"问题数: {len(questions)}  →  {', '.join(questions)}")
    for e in errors:
        print(f"  ✗ {e}")
    for w in warnings:
        print(f"  ⚠ {w}")
    if not errors:
        print("✓ 格式校验通过" + ("（token 预算需在 laya 环境跑 --lint）" if len(sys.argv) < 3 else ""))
    sys.exit(1 if errors else 0)
