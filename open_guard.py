#!/usr/bin/env python3
"""开源门禁两件套:
1. sanitize_for_open(): 私人实体 → 通用替换(语义等价), 处理 jsonl 训练数据
2. pii_scan(): PII 扫描, 命中私有实体名单 = fail (exit 1)
用法:
  python3 open_guard.py sanitize data/train.jsonl data/train_open.jsonl
  python3 open_guard.py scan data/train_open.jsonl
"""
import sys, json, re

# 私有实体 → 通用替换表(顺序敏感: 长词优先)
# 示例替换表(项目名/代号 → 通用等价词)。原则: 语义等价替换保训练信号; 长词优先防误替换。
# 用你自己的实体名单: 产品名→产品类别, 人称代称→通用称呼, 账号名→user
REPLACE = [
    ("my-kb-service", "知识库服务"), ("私有云Pro", "私有云"),
    ("内部代号A", "公司"), ("某账号名", "user"),
    ("我的记忆库", "知识库"), ("记忆库", "知识库"),
]

# 无论替换与否, 出现即 fail 的词(家人/真实人名等不可替换语义)
# 无论替换与否, 出现即 fail 的词(示例: 真实人名/不可替换语义的专名)
# 使用时换成你自己的不可替换实体名单
HARD_FAIL = ["某人真名", "某公司实名"]

def sanitize_text(s):
    for a, b in REPLACE:
        s = s.replace(a, b)
    return s

def sanitize_jsonl(src, dst):
    """替换 jsonl 所有字符串字段中的私人实体; 含 'q' 键的行(relevance/kw)同样处理"""
    n, changed = 0, 0
    with open(dst, "w") as f:
        for line in open(src):
            n += 1
            orig = line
            line = sanitize_text(line)
            if line != orig:
                changed += 1
            f.write(line)
    print(f"sanitized {src} → {dst}: {n} 行, 替换命中 {changed}")

def pii_scan(path):
    """扫描 jsonl: 私有实体残留 = fail"""
    bad = []
    for i, line in enumerate(open(path), 1):
        for w in [a for a, _ in REPLACE] + HARD_FAIL:
            if w in line:
                bad.append((i, w, line[:60].strip()))
                break
    if bad:
        print(f"✗ PII FAIL: {len(bad)} 行残留私有实体:")
        for i, w, frag in bad[:10]:
            print(f"  行{i} [{w}] {frag}")
        sys.exit(1)
    print(f"✓ PII PASS: {path} 无私有实体残留")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sanitize":
        sanitize_jsonl(sys.argv[2], sys.argv[3])
    elif cmd == "scan":
        pii_scan(sys.argv[2])
