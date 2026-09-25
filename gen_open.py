#!/usr/bin/env python3
"""通用场景数据生成器 v4: 开源普适性补强
脱离特定助手语境的通用中文决策场景 × 4 簇:
  A. 通用消息分流 (route) - IT运维/电商/客服/教育/办公
  B. 通用工单路由 (route 变体) - 官方 laya 的 billing/technical/sales 场景中文版
  C. 通用关键词选择 (kw_select) - 通用文档库
  D. 通用相关性 (relevance) - 通用 RAG 域
所有内容零 PII、零私人实体(通用公司名/产品名虚构)。
"""
import json, random, os
random.seed(20260924)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def onehot(i, n):
    v = [0.0] * n
    v[i] = 1.0
    return v

# ============ A. 通用消息分流 ============
U_URGENT = lambda: random.choice([
    f"{random.choice(['线上','生产','正式'])}环境{random.choice(['宕机','崩溃','不可用'])}，{random.choice(['立即排查','马上处理'])}",
    f"{random.choice(['订单','支付','物流'])}系统{random.choice(['批量失败','全部超时','卡死'])}，影响{random.choice(['线上交易','大量用户'])}",
    f"客户{random.choice(['在群里发火','要求立即退款','投诉到总部'])}，说{random.choice(['服务不可用','数据丢了','一直没人处理'])}",
    f"{random.choice(['数据库','存储','接口'])}{random.choice(['锁死','报错','容量告警'])}，{random.choice(['业务全停','大量工单进来'])}",
    f"{random.choice(['证书','域名','备案'])}明天到期，{random.choice(['不续就断服','今天必须搞定'])}",
    f"{random.choice(['直播','发布会','促销活动'])}马上开始，{random.choice(['关键环节还没准备好'])}",
    f"安全{random.choice(['告警','扫描发现漏洞'])}，{random.choice(['疑似被入侵','有异常登录'])}",
    f"{random.choice(['备份','定时任务'])}连续{random.choice(['三天','一周'])}失败，今天必须解决",
])
U_NORMAL = lambda: random.choice([
    f"{random.choice(['周报','月报','季度总结'])}写好了，{random.choice(['在共享盘','发你邮箱'])}，有空看看",
    f"{random.choice(['下周二','周三','周五'])}{random.choice(['下午','上午'])}开会，{random.choice(['记得参加','别迟到'])}",
    f"这个月{random.choice(['用户增长','销售额','转化率'])}{random.choice(['涨了15%','持平','略有下滑'])}",
    f"{random.choice(['新同事','实习生'])}下周入职，{random.choice(['账号我开好了'])}",
    f"{random.choice(['采购','报销','合同'])}流程{random.choice(['走完了','在审批中'])}",
    f"文档{random.choice(['更新到最新版','整理完了'])}，不急，慢慢看",
    f"{random.choice(['例行维护','版本升级'])}{random.choice(['完成','顺利'])}，一切正常",
    f"{random.choice(['供应商','物流'])}那边说{random.choice(['后天到货','明天发走'])}",
])
U_PERSONAL = lambda: random.choice([
    f"{random.choice(['晚上','周末','明天'])}{random.choice(['一起吃饭吗','打球去不','看电影吧'])}",
    f"{random.choice(['哈哈哈','笑死','绝了'])}，{random.choice(['你看这个','发你好玩'])}",
    f"最近{random.choice(['睡得怎么样','别太累','注意身体'])}",
    f"{random.choice(['我妈','我朋友'])}{random.choice(['问你好','说改天聚'])}",
    f"{random.choice(['那家餐厅','新开的咖啡店'])}真不错，{random.choice(['下次再去'])}",
    f"{random.choice(['游戏','球赛'])}看没看，{random.choice(['太精彩了','一起玩啊'])}",
])
U_SPAM = lambda: random.choice([
    f"{random.choice(['低息贷款','无抵押借款','秒批额度'])}了解一下，{random.choice(['利息超低'])}",
    f"恭喜您{random.choice(['中奖','被抽中'])}，{random.choice(['点击领取','填卡领奖'])}",
    f"{random.choice(['兼职刷单','在家日结'])}，{random.choice(['日入500','加群了解'])}",
    f"您的{random.choice(['包裹','会员','账号'])}异常，{random.choice(['点击链接','回复Y'])}处理",
    f"{random.choice(['专业建站','软件开发'])}{random.choice(['请联系','加V聊'])}",
    f"我是你{random.choice(['领导','老板'])}的小号，{random.choice(['加一下有安排'])}",
])

# ============ B. 通用工单路由(官方 laya 场景中文版) ============
TICKETS = [
    ("我上个月被扣了两次费，要求退回重复的部分", "billing"),
    ("发票抬头开错了，需要重新开具", "billing"),
    ("你们的定价方案里企业版包含哪些服务", "sales"),
    ("想了解一下年付有没有折扣", "sales"),
    ("APP 打开就闪退，安卓和苹果都这样", "technical"),
    ("数据导出功能一直转圈加载不出来", "technical"),
    ("想咨询团队版可以添加多少个成员", "sales"),
    ("付款成功了但服务还没开通", "billing"),
    ("网页在 Safari 上排版全乱了", "technical"),
    ("你们和竞品比核心优势是什么", "sales"),
]
Q_TICKET = {"type": "choice", "instructions": "Which department should handle this ticket?",
            "criteria": {"billing": "invoices, payments, refunds",
                         "technical": "bugs, outages, errors",
                         "sales": "pricing, plans, new purchases"}}

def gen_tickets(n=300):
    rows = []
    for _ in range(n):
        text, dept = random.choice(TICKETS)
        rows.append({"state": text, "cluster": "ticket",
                     "labels": {"ticket": onehot(list(Q_TICKET["criteria"]).index(dept), 3)}})
    return rows

# ============ C/D: 通用文档域(kw + rel 共用) ============
OPEN_DOMAINS = {
    "编程": (["代码报错怎么排查", "学什么语言入门好", "git 怎么回滚版本"],
             ["Python 异常处理最佳实践：try/except 与日志", "Git 分支管理策略与回滚操作指南", "程序员成长路线图：从初级到高级"]),
    "营销": (["小红书怎么涨粉", "投放预算怎么分配", "品牌定位怎么做"],
             ["小红书冷启动指南：从0到1000粉", "广告投放 ROI 优化实操手册", "品牌定位方法论：定位理论与案例"]),
    "健康": (["减脂期怎么吃", "睡眠质量怎么提升", "膝盖疼还能跑步吗"],
             ["科学减脂饮食指南：热量缺口与营养配比", "睡眠卫生手册：改善睡眠的十个习惯", "跑步运动损伤预防与恢复"]),
    "旅游": (["第一次去日本怎么规划", "行李清单有哪些必备", "签证怎么办"],
             ["日本自由行攻略：关西关东路线规划", "长途旅行行李清单模板", "签证办理流程与材料清单"]),
    "理财": (["基金定投怎么开始", " emergency fund 留多少", "股票和基金区别"],
             ["指数基金定投入门：策略与心态", "个人财务管理：应急资金与资产配置", "常见投资品对比：股债基黄金"]),
}
NEAR = {"编程": ["营销"], "营销": ["理财"], "健康": ["旅游"], "旅游": ["健康"], "理财": ["编程"]}

def gen_kw_rel(n_kw=800, n_rel=800):
    import jieba
    rows = []
    all_dom = list(OPEN_DOMAINS)
    STOP = set("帮我 看看 那个 这个 什么 怎么 一下 怎么样 的 了 吗 呢 哪里".split())
    for _ in range(n_kw):
        dom = random.choice(all_dom)
        queries, docs = OPEN_DOMAINS[dom]
        q = random.choice(queries)
        # 候选词: query 分词 + 噪声词
        words = [w.strip() for w in jieba.cut(q) if len(w.strip()) >= 2 and w.strip() not in STOP]
        if not words:
            continue
        good = random.choice(words)
        noise = random.choice(["问题", "情况", "资料", "内容", "东西"])
        # 好词是 query 主题词(标签 0.9); 噪声词(0.05); query 中另一个普通词(0.5~0.6)
        rows.append({"state": q, "cluster": "kw-open", "q": good,
                     "labels": {"kw_select": [0.1, 0.9]}})
        rows.append({"state": q, "cluster": "kw-open", "q": noise,
                     "labels": {"kw_select": [0.95, 0.05]}})
    for _ in range(n_rel):
        dom = random.choice(all_dom)
        queries, docs = OPEN_DOMAINS[dom]
        q, note = random.choice(queries), random.choice(docs)
        rows.append({"state": note, "cluster": "rel-open", "q": q,
                     "labels": {"relevance": [0.1, 0.9]}})
        # 弱相关/无关
        other = random.choice([x for x in all_dom if x != dom])
        if random.random() < 0.6:  # 无关
            q2 = random.choice(OPEN_DOMAINS[other][0])
            rows.append({"state": note, "cluster": "rel-open", "q": q2,
                         "labels": {"relevance": [0.9, 0.1]}})
        else:  # 弱相关(相邻域的 query, 本域文档)
            q3 = random.choice(OPEN_DOMAINS[NEAR[dom][0]][0])
            rows.append({"state": note, "cluster": "rel-open", "q": q3,
                         "labels": {"relevance": [0.35, 0.65]}})
    return rows

def main():
    rows = []
    n = {"open-work-urgent": 550, "open-work-normal": 550, "open-personal": 350, "open-spam": 450}
    for name, fn, ri in [("open-work-urgent", U_URGENT, 0), ("open-work-normal", U_NORMAL, 1),
                          ("open-personal", U_PERSONAL, 2), ("open-spam", U_SPAM, 3)]:
        for _ in range(n[name]):
            it = 1 if name == "work-urgent" else 0
            pr = random.choice([2, 3]) if name == "work-urgent" else (1 if name == "work-normal" else 0)
            rows.append({"state": fn(), "cluster": f"open-{name}",
                         "labels": {"route": onehot(ri, 4), "interrupt": onehot(it, 2),
                                    "priority": onehot(pr, 4)}})
    rows += gen_tickets()
    rows += gen_kw_rel()
    random.shuffle(rows)
    nv = int(len(rows) * 0.1)
    with open(f"{OUT}/train_open.jsonl", "w") as f:
        for x in rows[:-nv]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(f"{OUT}/val_open.jsonl", "w") as f:
        for x in rows[-nv:]:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"✓ open 数据: train {len(rows)-nv} / val {nv}; 簇分布 {dict(Counter(x['cluster'] for x in rows))}")

if __name__ == "__main__":
    main()
