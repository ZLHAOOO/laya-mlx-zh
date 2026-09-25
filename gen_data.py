#!/usr/bin/env python3
"""laya-zh Phase 1: 中文决策合成数据生成器
输出 jsonl: {state, labels: {route: [...4类概率], interrupt: [P(false),P(true)], priority: [...4级概率]}}
设计:
 - 模板簇带 ground truth 标签; 程序做词汇/语气/标点/中英混杂扰动
 - train/val 从模板簇生成; test 独立手写(零模板共享), 防背题
"""
import json, random, os
random.seed(20260923)

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# ============ 称呼 / 语气 / 扰动素材 ============
BOSS = ["老板", "领导", "老板儿", "哥", ""]
SOFT = ["", "麻烦看一下", "有空看看", "辛苦", "多谢", "拜托"]
URGENT_SUFFIX = ["！", "！！", "!!!", "，急！", "，尽快！", "，在线等", "，速回"]
POLITE_END = ["。", "~", "～", "哈", "啦", "呢"]
TYPO_MAP = {"环境": "还境", "页面": "夜面", "文档": "纹档", "数据": "数剧", "会议": "会义", "赶紧": "感紧", "检查": "捡查"}

def typos(s, p=0.15):
    if random.random() < p:
        for k, v in TYPO_MAP.items():
            if k in s and random.random() < 0.5:
                s = s.replace(k, v); break
    return s

def mix_en(s, p=0.25):
    """中英混杂扰动(贴近真实工作消息)"""
    m = {"环境": "env", "页面": "page", "数据": "data", "文档": "doc", "bug": "bug", "部署": "deploy"}
    if random.random() < p:
        for k, v in m.items():
            if k in s:
                s = s.replace(k, v, 1); break
    return s

def perturb(s, urgent=False):
    s = mix_en(typos(s))
    if urgent and random.random() < 0.7:
        s += random.choice(URGENT_SUFFIX)
    elif random.random() < 0.4:
        s += random.choice(POLITE_END)
    if random.random() < 0.15:
        s = random.choice(["[zzz]", "😂", "🙏", "⚠️", "‼️"]) + s
    return s

# ============ 模板簇: (意图, 句子工厂) ============
# route: work-urgent / work-normal / personal / spam
# interrupt: 0/1  priority: 0低 1普通 2急 3特急

C_URGENT = lambda: random.choice([
    f"{random.choice(['生产','线上','正式'])}环境{random.choice(['挂了','崩了','宕机','挂掉'])}，{random.choice(['赶紧','快点','立刻'])}看看",
    f"{random.choice(['官网','商城','小程序','后台'])}{random.choice(['白屏','打不开','报错500','接口全挂'])}",
    f"客户{random.choice(['投诉','炸了','在群里发火'])}，{random.choice(['说系统用不了','订单全失败','要退单'])}",
    f"数据库{random.choice(['磁盘满了','写入失败','锁死了'])}，{random.choice(['服务全停了','业务不可用'])}",
    f"{random.choice(['支付','登录','推送','短信'])}{random.choice(['回调失败','服务异常','全量超时'])}，影响{random.choice(['全部用户','线上订单','核心流程'])}",
    f"刚发布的版本有{random.choice(['严重bug','致命问题'])}，{random.choice(['赶紧回滚','先回滚再说'])}",
    f"{random.choice(['推广链接','活动页','落地页','报名入口'])}挂了{random.choice(['两小时','半天','一上午'])}了{random.choice(['还没人发现吗','影响报名了'])}",
    f"用户{random.choice(['反馈','投诉'])}{random.choice(['注册收不到验证码','登录不上','付不了款'])}，{random.choice(['先排查下','看看什么情况'])}",
    f"{random.choice(['域名','SSL证书','备案'])}{random.choice(['过期了','快到期了','被注销了'])}，{random.choice(['赶紧续','需要马上处理'])}",
    f"{random.choice(['服务器','硬盘','内存'])}{random.choice(['报警了','负载爆了','CPU 100%下不来'])}",
    f"{random.choice(['甲方','合作方','那家MCN'])}说{random.choice(['今天下班前要答复','合同有问题','要毁约'])}，对方在等",
    f"{random.choice(['备份','定时任务','爬虫'])}连着{random.choice(['三天','一周'])}没{random.choice(['跑成功','跑完'])}了，今天必须解决",
    f"{random.choice(['直播','发布会','活动'])}{random.choice(['马上开始','半小时后开始'])}，{random.choice(['推流还没调好','链接还没配','流程还没对'])}",
    f"账号{random.choice(['被封了','异常登录','被盗了'])}，{random.choice(['赶紧申诉','先冻结'])}",
    f"{random.choice(['数据','素材','代码'])}{random.choice(['丢了','被误删了','回滚失败了'])}，{random.choice(['看看还能不能找回来','急死'])}",
])

C_NORMAL = lambda: random.choice([
    f"{random.choice(['这个月','上周','今天'])}的{random.choice(['投放数据','运营数据','涨粉数据'])}出来了，{random.choice(['涨了30%','整体平稳','有点下滑'])}",
    f"{random.choice(['周报','方案','文档','会议纪要'])}我{random.choice(['整理好了','写完了','放云文档了'])}，有空看一下就行",
    f"{random.choice(['下周三下午三点','明晚八点','周五上午'])}开{random.choice(['例会','复盘会','选题会'])}，记得来",
    f"{random.choice(['官网','商城','自动化脚本'])}的{random.choice(['日常巡检','例行维护','定时任务'])}{random.choice(['都正常','跑完了','没问题'])}",
    f"素材库{random.choice(['更新了一批','补了几个新模板','整理完旧素材'])}，{random.choice(['之后可以用','先存着'])}",
    f"合同{random.choice(['对方已经寄出','走完流程了','章盖好了'])}，{random.choice(['等快递到','留档一份'])}",
    f"{random.choice(['飞书文档','网盘','权限'])}我给开了，需要的话自己进去{random.choice(['拿素材','看资料','取文件'])}",
    f"{random.choice(['新来的实习生','新同事'])}{random.choice(['下周入职','后天到岗'])}，{random.choice(['工位和账号我提前弄好','帮忙带一下'])}",
    f"{random.choice(['上个月','这个月'])}的{random.choice(['报销单','发票','对账单'])}{random.choice(['财务说少了签字','我提交了','核对完了'])}",
    f"{random.choice(['公众号','小红书','抖音'])}{random.choice(['的排版调好了','排期排好了','封面图做好了'])}，{random.choice(['过目一下','明早发'])}",
    f"{random.choice(['供应商','快递','打印店'])}那边{random.choice(['报价发我了','说后天到','弄好了'])}",
    f"{random.choice(['选题池','内容日历','排期表'])}{random.choice(['初稿写好了','更新了','发你邮箱了'])}，不急，慢慢看",
    f"{random.choice(['上期视频','昨天那条笔记','上周的推文'])}{random.choice(['数据出来了','评论区有人问','被转发了不少'])}",
    f"{random.choice(['例行备份','每周同步','版本归档'])}{random.choice(['做完了','没问题','已存好'])}",
    f"{random.choice(['工具账号','会员','订阅'])}{random.choice(['该续费了','我续好了','下月到期'])}，{random.choice(['记得走报销','不用急'])}",
])

C_PERSONAL = lambda: random.choice([
    f"{random.choice(['晚上','周末','明天'])}{random.choice(['一起打游戏不','吃个饭','看电影去','爬山去'])}",
    f"{random.choice(['哈哈','笑死','绝了'])}，{random.choice(['你看这个','给你看个好玩的','这个梗太对了'])}",
    f"{random.choice(['周末','假期','国庆'])}{random.choice(['有安排吗','打算干嘛','回家不'])}",
    f"最近{random.choice(['睡得咋样','腰好点没','别太累'])}",
    f"{random.choice(['我妈','我朋友','我表弟'])}{random.choice(['问我那个事','说谢谢咱','想认识你'])}",
    f"{random.choice(['刷到','看到'])}一个{random.choice(['视频','帖子','段子'])}{random.choice(['笑死我了','绝了','发你看下'])}",
    f"{random.choice(['咖啡豆','茶叶','零食'])}到了，给你{random.choice(['留了一包','带了一份'])}",
    f"你上次推荐的那家{random.choice(['面馆','咖啡店','烧烤'])}真不错，{random.choice(['改天再去','下次带朋友去'])}",
    f"{random.choice(['今天好累','哈欠连天','困死了'])}，{random.choice(['先睡了','今天先到这吧','明天见'])}",
    f"{random.choice(['我爸','我妈','我奶奶'])}让我{random.choice(['问你','问一下'])}{random.choice(['中秋回不回老家','周末来不来吃饭'])}",
    f"{random.choice(['晚上','明天早上'])}{random.choice(['一起跑步不','去游泳吗','打羽毛球吗'])}",
    f"{random.choice(['新剧','那个电影','这个综艺'])}看了没，{random.choice(['太上头了','一起追啊','剧透一波'])}",
    f"{random.choice(['生日','节日'])}快乐呀，{random.choice(['祝你','愿你'])}{random.choice(['天天开心','万事顺意'])}",
    f"{random.choice(['家里','楼下'])}{random.choice(['那只猫','小狗'])}{random.choice(['又干了件好事','超可爱','想它了'])}",
    f"{random.choice(['好久没聚了','下次见面'])}{random.choice(['找个时间吃个饭','聊聊近况'])}",
])

C_SPAM = lambda: random.choice([
    f"{random.choice(['恭喜发财','恭喜您','尊敬的用户'])}，{random.choice(['分期贷款','低息借款','无抵押贷款'])}了解一下",
    f"{random.choice(['限时','最后一天','仅限今日'])}{random.choice(['大奖待领','抽奖机会','免费领取'])}{random.choice(['iPhone','代金券','会员'])}",
    f"{random.choice(['兼职刷单','在家日结','躺赚项目'])}{random.choice(['加微信','私聊','点链接'])}",
    f"您{random.choice(['有一个包裹无法投递','的会员即将过期','订阅的服务异常'])}，{random.choice(['点击链接','回复Y','联系客服'])}处理",
    f"{random.choice(['专业代开发','网站建设','软件开发'])}{random.choice(['需求','服务'])}{random.choice(['请联系','加V详聊'])}",
    f"【官方】您的{random.choice(['账户','账号'])}存在异常登录，立即验证身份{random.choice(['bit.ly/xxxx','短链'])}",
    f"{random.choice(['急招','招聘'])}{random.choice(['兼职点赞员','线上客服','打字员'])}，日结{random.choice(['300','500'])}，{random.choice(['无需经验','加Q群'])}",
    f"我是你{random.choice(['领导','老板'])}，加一下我这个{random.choice(['小号','工作号'])}，{random.choice(['有工作安排','帮我个忙'])}",
    f"{random.choice(['银行','金融'])}{random.choice(['信用贷','备用金'])}利率{random.choice(['3.2%','2.9%'])}{random.choice(['急批','秒批'])}，回1立即办理",
    f"亲，您{random.choice(['店铺的信誉分不足','的店铺被降权'])}，点击升级避免{random.choice(['处罚','封店'])}",
    f"{random.choice(['高质量美女','同城交友','寂寞少妇'])}{random.choice(['视频聊天','约会'])}{random.choice(['下载APP','注册就玩'])}",
    f"{random.choice(['虚拟币','区块链','元宇宙'])}{random.choice(['投资','项目'])}{random.choice(['稳赚不赔','内幕消息','跟着老师买'])}",
    f"您的{random.choice(['快递','邮件'])}被{random.choice(['扣留在海关','检测出问题'])}，{random.choice(['支付关税','点此申述'])}",
    f"{random.choice(['教育资源','网课','考证'])}{random.choice(['最后名额','限时免费'])}{random.choice(['扫码进群','先到先得'])}",
    f"{random.choice(['低价代开发','破解软件','免费资源'])}{random.choice(['大礼包','合集'])}{random.choice(['关注公众号领取','回复关键词'])}",
])

def make_q():
    return {
        "route": {"type": "choice", "instructions": "Which category does this message belong to?",
                  "criteria": {
                      "work-urgent": "needs action now: incidents, deadlines, complaints, outages",
                      "work-normal": "work matters without urgency: reports, notices, schedules",
                      "personal": "friends and family, casual chat, social plans",
                      "spam": "ads, scams, phishing, cold promotions",
                  }},
        "interrupt": {"type": "noul", "instructions": "Does this message need immediate attention right now?"},
        "priority": {"type": "score", "instructions": "How urgent is this message?",
                     "criteria": ["not urgent", "soon", "urgent", "critical"]},
    }

CLUSTERS = {  # 簇 -> (route_idx, interrupt, priority, 句子工厂, urgent_flag)
    "work-urgent": (0, 1, 3, C_URGENT, True),
    "work-normal": (1, 0, 1, C_NORMAL, False),
    "personal":    (2, 0, 0, C_PERSONAL, False),
    "spam":        (3, 0, 0, C_SPAM, False),
}

def onehot(i, n):
    v = [0.0] * n
    v[i] = 1.0
    return v

def gen_item(cluster):
    ri, it, pr, factory, urgent = CLUSTERS[cluster]
    body = perturb(factory(), urgent=urgent)
    boss = random.choice(BOSS)
    state = (boss + "，" + body) if boss else body
    if cluster == "work-normal" and random.random() < 0.3:
        state += random.choice(SOFT)
    return {
        "state": state,
        "cluster": cluster,
        "labels": {
            "route": onehot(ri, 4),
            "interrupt": onehot(it, 2),   # noul: [P(false), P(true)]; it=1(要打扰)→[0,1]
            "priority": onehot(pr, 4),
        },
    }

# ============ test 集: 独立手写,零模板共享 ============
TEST_HANDWRITTEN = [
    # (state, cluster)
    ("服务器刚重启完，监控又报警了， CPU 100% 下不来", "work-urgent"),
    ("甲方的对接人说 logo 再改第 18 版，不然不验收", "work-urgent"),
    ("推广链接挂了两个小时了还没人发现吗，赶紧处理下", "work-urgent"),
    ("备份任务连着三天没跑成功，今天必须解决", "work-urgent"),
    ("官网的用户反馈注册收不到验证码，先排查下", "work-urgent"),
    ("那家 MCN 说合同今天下班前要给答复，对方在等", "work-urgent"),
    ("上个月的报销单财务说少了个签字，补一下", "work-normal"),
    ("我把下季度的选题池初稿发你邮箱了，不急，慢慢看", "work-normal"),
    ("飞书文档权限我给开了，需要的话自己进去拿素材", "work-normal"),
    ("周三的直播复盘会议纪要整理好了，重点在第三页", "work-normal"),
    ("账号后台的评论我清了一轮，都是常规维护", "work-normal"),
    ("新来的实习生下周入职，工位和账号我提前弄好", "work-normal"),
    ("哈欠连天，今天先到这吧，明天见", "personal"),
    ("刷到一个视频笑死我了，发你看下", "personal"),
    ("我爸让我问你中秋回不回老家，好安排车", "personal"),
    ("这周囤的咖啡豆到了，给你留了一包", "personal"),
    ("晚上要不要一起拼个桌打球", "personal"),
    ("你上次推荐的那家面馆真不错，改天再去", "personal"),
    ("【官方】您的账户存在异常登录，立即验证身份 bit.ly/xxxx", "spam"),
    ("恭喜您被抽中成为幸运用户，填写银行卡即可领取 8888 元", "spam"),
    ("急招兼职点赞员，日结 300，无需经验，加 Q 群", "spam"),
    ("我是你领导，加一下我这个小号，有工作安排", "spam"),
    ("银行信用贷利率 3.2% 急批，回 1 立即办理", "spam"),
    ("亲，您店铺的信誉分不足，点击升级避免降权", "spam"),
    # 第二批: 更多场景覆盖
    ("域名明天到期忘了续，网站已经打不开了", "work-urgent"),
    ("直播还有半小时就开始，推流还没调好", "work-urgent"),
    ("甲方说方案要大改，明天早上要新版本", "work-urgent"),
    ("公司账号被盗了，正在申诉", "work-urgent"),
    ("排期表我更新了，下月的内容都排好了", "work-normal"),
    ("公众号封面图做好了，明早八点准时发", "work-normal"),
    ("供应商报价发我了，你看下合不合理", "work-normal"),
    ("会员下月到期，我续好了，记得走报销", "work-normal"),
    ("昨天那条笔记评论区有人问合作方式", "work-normal"),
    ("下次见面找个时间吃个饭，聊聊近况", "personal"),
    ("新出的那个游戏你入了没，一起开黑", "personal"),
    ("我妈包了饺子，给你留了一盒放楼下了", "personal"),
    ("这只猫又偷吃我的外卖，服了", "personal"),
    ("央视网提醒您：点击链接领取国庆大礼包，先到先得", "spam"),
    ("您的邮箱存储已满，点击升级到会员避免被清空", "spam"),
    ("加我好友免费送皮肤，还可以赚钱哦", "spam"),
    ("房子急售，低于市场价三成，内部渠道", "spam"),
]

def label_for(cluster):
    ri, it, pr, _, _ = CLUSTERS[cluster]
    return {"route": onehot(ri, 4), "interrupt": onehot(it, 2), "priority": onehot(pr, 4)}

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def main():
    n_per = {"work-urgent": 600, "work-normal": 600, "personal": 450, "spam": 550}
    pool = []
    for c, n in n_per.items():
        pool += [gen_item(c) for _ in range(n)]
    seen, dedup = set(), []
    for x in pool:
        if x["state"] not in seen:
            seen.add(x["state"]); dedup.append(x)
    random.shuffle(dedup)
    n = len(dedup)
    nv = int(n * 0.15)
    write_jsonl(f"{OUT}/train.jsonl", dedup[:-nv])
    write_jsonl(f"{OUT}/val.jsonl", dedup[-nv:])
    test = [{"state": s, "cluster": c, "labels": label_for(c)} for s, c in TEST_HANDWRITTEN]
    write_jsonl(f"{OUT}/test.jsonl", test)

    from collections import Counter
    for split, items in [("train", dedup[:-nv]), ("val", dedup[-nv:]), ("test", test)]:
        cnt = Counter(x["cluster"] for x in items)
        print(f"{split}: {len(items)} 条  {dict(cnt)}")
    print("\n=== 样例抽查 ===")
    for x in random.sample(dedup, 6):
        print(f"[{x['cluster']:12s}] {x['state']}")

if __name__ == "__main__":
    main()
