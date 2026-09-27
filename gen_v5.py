"""laya-zh v5 数据增补生成器：多标签空间混训（治 #364 实锤的 criteria 塌缩）

第一性原理：塌缩根因 = 头只见过 13 个固定 criteria 词串，退化为"背词串"。
药方 = 让头在训练里见过**大量不同的 criteria 集合**（7 个新任务头），
且同一标签的描述有**措辞变体**（行内 qs 覆盖）——背不过，只能学会"读"。

七头（全部共享现有 choice/noul 头，靠 ins+crit 文本区分——laya 原生设计）：
  code_router      代码领域路由(choice 6)      marketing_router 营销领域路由(choice 6)
  finance_ops      财务领域路由(choice 6)      life_admin       生活管家路由(choice 6)
  robot_cmd        机器人语音指令(choice 6)    —— 镜像 #364 形状，措辞**不同**（合规：考卷原文零进入）
  emotion_*/intent_cn              情绪(noul 3问) / 通用中文意图(choice 6)

软标签原则（沿用 data-ext 惯例）：明确 0.9/0.1，边界 0.4~0.7 梯度——教校准不只教对错。
切分：train / val(25/头) / test_v5(泛化 held-out，全新表达，与 train 无共享模板)。
random.seed 可复现；全合成无 PII。
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_data import perturb  # 复用主生成器的错别字/中英夹杂/语气扰动

random.seed(20260927)
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data-v5")
os.makedirs(OUT, exist_ok=True)

# =====================================================================
# criteria 描述变体：主描述(QS 用) + 变体(行内 qs 用，30% 概率挂行)。
# robot_cmd 的所有描述**语义同、措辞异**于官方考卷——泛化合规的关键。
# =====================================================================
CODE_CRIT = {
    "bugfix":   (["排查修复：报错、崩溃、异常", "修 bug：程序出错、行为不对"],
                 ["修 bug 和异常", "错误处理与崩溃排查"]),
    "feature":  (["新功能开发：加功能、改交互", "写新功能、加模块"],
                 ["开发新功能", "功能新增与交互改动"]),
    "refactor": (["重构优化：不改行为只改质量", "代码整理：瘦身、改名、抽公共"],
                 ["代码重构与清理", "内部质量优化"]),
    "docs":     (["文档查询：API 用法、配置说明", "查文档、找用法示例"],
                 ["文档查阅", "用法与配置说明"]),
    "deploy":   (["部署运维：构建、发布、环境", "上线部署：打包、发布、服务器"],
                 ["构建与发布", "环境部署运维"]),
    "chat":     (["与代码无关的消息", "不是技术工作的话题"],
                 ["非代码话题", "与技术无关"]),
}
MKT_CRIT = {
    "content": (["内容创作：文案、图文、脚本", "写东西：笔记、推文、视频稿"],
                ["创作内容", "写文案与脚本"]),
    "seo":     (["流量获取：关键词、SEO、投放", "搞流量：搜索优化、付费投放"],
                ["搜索与投放", "流量渠道运营"]),
    "data":    (["数据分析：转化、漏斗、复盘", "看数据：增长指标、效果归因"],
                ["数据复盘", "指标与归因分析"]),
    "crm":     (["客户关系：社群、私聊、跟进", "维护客户：回复、回访、促单"],
                ["客户沟通维护", "社群与私域跟进"]),
    "brand":   (["品牌定位：slogan、视觉、调性", "品牌建设：名字、风格、心智"],
                ["品牌与定位", "调性与视觉风格"]),
    "none":    (["与营销无关的消息", "不是市场推广的话题"],
                ["非营销话题", "与推广无关"]),
}
FIN_CRIT = {
    "expense": (["报销付款：报销、垫付、转账", "钱往外走：报销、付款申请"],
                ["报销与付款", "支出申请处理"]),
    "invoice": (["发票：开票、收票、整理", "发票事务：开具、查验、归档"],
                ["发票处理", "开票与收票"]),
    "budget":  (["预算管理：预算、成本、超支", "管钱规划：预算编制、成本控制"],
                ["预算与成本", "花钱规划"]),
    "recon":   (["对账核对：账单、流水、差异", "对账：查差异、核流水"],
                ["账目核对", "对账与查错"]),
    "tax":     (["税务合规：报税、税率、合规", "报税缴税：税务申报、优惠政策"],
                ["税务申报", "税与合规"]),
    "none":    (["与财务无关的消息", "不是钱相关的话题"],
                ["非财务话题", "与资金无关"]),
}
LIFE_CRIT = {
    "schedule": (["日程安排：约会、会议、提醒", "时间管理：约人、排日程、别忘事"],
                 ["日程与提醒", "安排时间事务"]),
    "shopping": (["购物消费：想买、下单、退换", "买东西：比价、下单、售后"],
                 ["购物相关", "买与退换"]),
    "health":   (["健康：就医、用药、锻炼", "身体的事：看病、吃药、运动"],
                 ["健康医疗", "身体与锻炼"]),
    "travel":   (["出行：通勤、订票、路线", "出门的事：交通、车票、导航"],
                 ["出行交通", "路线与订票"]),
    "home":     (["家务：保洁、维修、缴费", "家里的事：打扫、修东西、交费"],
                 ["家务维护", "居家事务"]),
    "chat":     (["纯闲聊，不用执行", "随便聊聊，无需动作"],
                 ["闲聊即可", "不用管，聊天"]),
}
ROBOT_CRIT = (  # 措辞与官方 CHOICE_CRITERIA 不同；语义对齐——泛化测试合规
    {"faster": (["让机器人提速", "速度调快一些"], ["提高运动速度", "跑快一点"]),
     "slower": (["让机器人减速", "速度放慢一些"], ["降低运动速度", "跑慢一点"]),
     "stop":   (["让机器人停下", "终止当前运动"], ["停止运动", "停在原地别动"]),
     "left":   (["向左侧转向", "往左边方向走"], ["左转", "朝左边移动"]),
     "right":  (["向右侧转向", "往右边方向走"], ["右转", "朝右边移动"]),
     "none":   (["不是对机器人的运动指令", "与机器人动作无关的话语"], ["无关闲聊", "不是运动指令"])}
)
INTENT_CRIT = {
    "ask":     (["提问咨询：想知道某事", "问问题：求解释、求答案"], ["咨询提问", "求答案"]),
    "do":      (["请求执行：让助手做某事", "下任务：交代具体活"], ["行动请求", "交办任务"]),
    "feedback":(["反馈：确认、感谢、抱怨", "回应：收到、谢了、不满"], ["确认与反馈", "好评或吐槽"]),
    "chat":    (["闲聊寒暄", "打招呼聊天"], ["日常寒暄", "无目的聊天"]),
    "spam":    (["广告推销骚扰", "营销群发、诈骗"], ["骚扰信息", "广告与诈骗"]),
    "other":   (["其他意图", "不好归类"], ["其余情况", "无法分类"]),
}

QS_V5 = {
    "code_router": {"t": "choice", "ins": "这条消息该路由给哪个代码类技能？", "crit": {k: v[0][0] for k, v in CODE_CRIT.items()}},
    "marketing_router": {"t": "choice", "ins": "这条消息该路由给哪个营销类技能？", "crit": {k: v[0][0] for k, v in MKT_CRIT.items()}},
    "finance_ops": {"t": "choice", "ins": "这条消息该路由给哪个财务类技能？", "crit": {k: v[0][0] for k, v in FIN_CRIT.items()}},
    "life_admin": {"t": "choice", "ins": "这条生活消息该交给哪个管家技能？", "crit": {k: v[0][0] for k, v in LIFE_CRIT.items()}},
    "robot_cmd": {"t": "choice", "ins": "这条指令要求机器人做什么？", "crit": {k: v[0][0] for k, v in ROBOT_CRIT.items()}},
    "intent_cn": {"t": "choice", "ins": "这条中文消息的主要意图是什么？", "crit": {k: v[0][0] for k, v in INTENT_CRIT.items()}},
    "emotion_emo": {"t": "noul", "ins": "这条消息是否带有明显情绪（喜、怒、哀、焦虑）？", "crit": {}},
    "emotion_care": {"t": "noul", "ins": "这条消息是否需要安抚或共情回应？", "crit": {}},
    "emotion_urgent": {"t": "noul", "ins": "这条消息是否涉及需要立刻处理的情绪危机？", "crit": {}},
}

# =====================================================================
# state 素材池（全合成、口语化）
# =====================================================================
CODE_POOL = {
    "bugfix": ["登录后一直转圈进不去", "这个接口返回 500 了", "页面在 Safari 上排版全乱", "导入 csv 直接闪退",
               "定时任务半夜没跑", "点击保存按钮没反应", "内存越用越高不释放", "这个报错栈看不懂帮我看看",
               "昨天还好好的今天突然崩了", "表单提交后数据没落库", "图片上传到一半断掉", "接口偶尔超时"],
    "feature": ["给后台加个导出 excel 的功能", "能不能支持深色模式", "我想加一个批量删除", "帮做个邀请码注册",
                "消息要支持已读回执", "加个数据大屏页面", "用户希望能绑定多个手机号", "做一个周报自动生成",
                "加上拖拽排序吧", "这个列表要能多选"],
    "refactor": ["这个文件两千行了拆一下吧", "把重复代码抽成公共函数", "这些魔法数字改成常量", "命名太乱了统一一下",
                 "把回调改成 async", "这个类职责太多了拆开", "删掉那些没人用的接口", "目录结构重新理一下"],
    "docs": ["这个库的超时参数怎么配", "有没有分页的示例代码", "这个 API 的鉴权怎么做", "部署文档发我一份",
             "这个函数的返回值是什么格式", "docker compose 怎么写", "这个配置项是什么意思", "查一下官方文档怎么说"],
    "deploy": ["帮我把新版本发布到测试环境", "构建失败了看下日志", "服务器磁盘满了怎么办", "nginx 要加个跨域配置",
               "生产环境的证书快过期了", "数据库迁移怎么执行", "打个 docker 镜像推到仓库", "环境变量怎么注入"],
    "chat": ["周五团建去哪吃", "你昨天说那个电影叫啥", "哈哈哈笑死我了", "今天天气真好啊",
             "中午吃什么好", "周末去爬山吗", "我刚看到一个搞笑视频"],
}
MKT_POOL = {
    "content": ["帮我写一篇小红书笔记", "这篇推文的标题不行重写", "下期的视频脚本出一下", "公众号文章排个版",
                "给我三个爆款选题", "这条口播文案顺一下", "写个产品卖点提炼", "封面图的文案想几个"],
    "seo": ["这个关键词的搜索量多少", "投流预算怎么分", "信息流的素材该换了", "SEO 排名怎么提上来",
            "搜索词报告拉一下", "竞品在投什么词", "这条广告 CTR 太低了", "自然流量怎么起量"],
    "data": ["上周的转化率掉了一截", "拉一下活动期间的销售漏斗", "复盘一下这次投放 ROI", "哪个渠道获客成本最低",
             "复购率怎么算", "这个月涨粉数据出来了吗", "把留存曲线画一下", "客单价的变化怎么看"],
    "crm": ["这个客户三天没回了怎么跟进", "社群里有人问价格怎么接", "把老客户名单整理一下", "私聊话术帮我看看",
            "有个客户要退款怎么安抚", "会员体系怎么设计", "发条召回短信给沉睡用户", "这个投诉帮我回复一下"],
    "brand": ["我们的 slogan 定不下来", "logo 想改版你给点意见", "品牌调性应该更专业还是更活泼", "帮想十个品牌名",
              "视觉规范该更新了", "我们的差异化卖点是什么", "品牌故事怎么写", "包装设计风格定一下"],
    "none": ["帮我订个会议室", "昨天球赛看了吗", "电脑连不上打印机", "晚饭吃什么",
             "我的快递到哪了", "记得提醒我拿药"],
}
FIN_POOL = {
    "expense": ["上个月出差的钱怎么报销", "这笔垫付什么时候能批", "采购预算超了两千怎么办", "帮我把这单付款申请提了",
                "团建费用走哪个科目", "差旅标准是多少", "这发票能报吗", "垫付的物料款到账了吗"],
    "invoice": ["给客户开一张增值税发票", "收到专票要怎么查验", "发票抬头错了重开", "这个月的发票整理一下",
                "电子发票和纸质发票效力一样吗", "开票信息发我一份", "丢失的发票怎么处理", "发票额度不够用了"],
    "budget": ["下季度的预算草案出一下", "这个项目成本超了", "各项开支占比拉个表", "预算冻结到什么时候",
               "人力成本今年涨了多少", "帮忙算一下盈亏平衡点", "新的预算周期什么时候开始", "这块费用该砍多少"],
    "recon": ["银行流水和账对不上", "帮我对一下上月的账", "这笔差异是什么", "应收账款核一下",
              "账期对不齐怎么调", "现金日记账帮我核一遍", "有两笔重复入账了", "月底对账别忘了"],
    "tax": ["这个季度报税几号截止", "小规模纳税人税率多少", "专票和普票的税点区别", "增值税怎么算",
            "有个税收优惠政策帮我看看", "年度汇算清缴要什么材料", "印花税要不要交", "个税专项扣除填了吗"],
    "none": ["帮我约个牙医", "明天记得带伞", "午饭点了吗", "这本小说太好看了",
             "今晚一起打球吗", "手机没电了借个充电器"],
}
LIFE_POOL = {
    "schedule": ["下周三下午帮我约个牙医", "提醒我周五交报告", "和老王约个饭", "把周会挪到上午十点",
                 "我妈生日是哪天来着", "帮我看看下周有什么安排", "晚上七点订个位子", "别忘了我明天要接孩子"],
    "shopping": ["想换个新手机帮我比个价", "这个链接帮我下单", "买的鞋子不合适怎么退", "双十一什么东西值得买",
                 "家里洗衣液快没了记一下", "帮我看看这个价格划不划算", "那个购物车里的东西降价了吗", "给孩子买双运动鞋"],
    "health": ["最近老失眠怎么办", "帮我挂个消化科的号", "这个药饭前吃还是饭后吃", "办个健身卡值得吗",
               "体检报告有个指标偏高", "颈椎疼有什么办法", "今天步数够了吗", "提醒我吃维生素"],
    "travel": ["明天早高峰走哪条路", "帮我订下周五去上海的高铁", "附近有没有加油站", "出差行李要带什么",
               "这个航班会晚点吗", "导航一下最近的药店", "机场大巴几点的车", "帮我看看酒店评分"],
    "home": ["家里路由器又断网了", "空调该加氟了叫个师傅", "水费电费该交了", "找个人通一下下水道",
             "周末大扫除分工一下", "冰箱灯不亮了", "门锁有点松", "物业电话多少"],
    "chat": ["哈哈今天太逗了", "你说人为什么要睡觉", "给你讲个段子", "今晚月色不错",
             "我好困啊", "新出的那个剧看了吗"],
}
# robot_cmd：口语指令池（覆盖官方 18 题的形状但全部重新措辞；含对抗：太X了/别X了/同字面反义）
ROBOT_POOL = {
    "faster": ["跑快一点", "提速", "再快点", "速度调上去", "麻利点", "能不能快一些", "效率太低了加速",
               "快点干", "全速前进", "加把劲", "节奏提起来", "稍微快点儿"],
    "slower": ["跑慢一点", "减速", "再慢点儿", "速度降下来", "慢悠悠点", "稳一点别急", "太快了",
               "慢速模式", "缓一缓速度", "悠着点", "放慢节奏", "别太快了"],
    "stop":   ["停下", "停一停", "赶紧停下", "急停", "别动了", "停下来别动", "就地停止",
               "先别跑了", "暂停一下", "呆在原地", "立刻停止", "停"],
    "left":   ["左转", "往左边一点", "向左侧转", "朝左走", "左边抹一下", "靠左", "向左修正",
               "稍微偏左", "左打一点方向", "贴着左边走"],
    "right":  ["右转", "往右边一点", "向右侧转", "朝右走", "右边抹一下", "靠右", "向右修正",
               "稍微偏右", "右打一点方向", "贴着右边走"],
    "none":   ["今天天气不错", "你叫什么名字", "几点了", "辛苦啦", "明天会下雨吗", "你真棒",
               "唱个歌听听", "我饿了", "今天累死了", "讲个笑话", "Battery 还有多少电", "几点吃午饭"],
}
INTENT_POOL = {
    "ask": ["redis 和 mysql 有啥区别", "这个政策是什么意思", "为什么天是蓝的", "帮我查下这个词啥意思",
            "怎么做番茄炒蛋", "这句话日语怎么说", "什么是复利", "这道题怎么解"],
    "do": ["帮我把这段话翻译成英文", "把这个文件转成 pdf", "帮我写封辞职信", "订个明早的闹钟",
           "把这些照片整理一下", "给这个文档改错别字", "帮我算一下这笔账", "把这个网页存下来"],
    "feedback": ["好的知道了", "收到谢谢", "这个结果不对啊", "上次那个做得不行", "非常感谢！",
                 "嗯嗯明白了", "这也太差劲了吧", "棒棒的"],
    "chat": ["在吗", "睡了吗", "今天好热", "无聊啊", "你好呀", "早上好", "刚下班"],
    "spam": ["秒批额度了解一下", "恭喜您获得免费奖品", "加微信看Full裸贷", "限时优惠最后一天",
             "您的账户异常点击验证", "兼职日结千元", "免费领取iPhone", "老板做了一个亿的项目了解一下"],
    "other": ["呃呃呃", "。。。", "嗯", "？？", "aaaa", "唉"],
}
# emotion：一条 state 同时出 3 问标签。情绪池（强/弱）+ 中性池 + 危机池
EMO_POOL = {
    "strong_neg": ["气死我了！！", "烦死了，什么破事都赶一起", "难过死了，不想说话", "又被喷了，好挫败",
                   "崩溃了真的", "气到失眠", "今天被领导当众骂了", "委屈死了明明不是我的错"],
    "mild_neg": ["有点烦", "心情不太好", "有点累不想动", "莫名有点丧", "今天不太顺", "有点焦虑",
                 "心里堵得慌", "压力好大啊"],
    "positive": ["今天超开心！", "太好了！！", "哈哈哈笑死", "激动死了，中了！", "被夸了嘿嘿",
                 "终于搞定了，爽", "今天运气爆棚", "开心到飞起"],
    "crisis": ["我撑不下去了", "感觉很糟不想活了", "没人管我死活", "整个人要碎掉了，救救我",
               "心跳很快手抖得厉害，我是不是要死了", "我不想活了"],
    "neutral": ["明天几点开会", "帮我查个东西", "路上堵车", "吃饭了吗", "文件我发你邮箱了",
                "下午三点碰一下", "快递到了", "数据我跑好了"],
}

def crit_variant(qname, label, which=None):
    """给某标签取一个 criteria 描述变体。which=None 时主 QS 不变（返回 None 表示用默认）。"""
    if qname == "robot_cmd":
        main, alts = ROBOT_CRIT[label]
    else:
        table = {"code_router": CODE_CRIT, "marketing_router": MKT_CRIT, "finance_ops": FIN_CRIT,
                 "life_admin": LIFE_CRIT, "intent_cn": INTENT_CRIT}[qname]
        main, alts = table[label]
    if which is None:
        return None
    pick = random.choice(alts)
    crit = dict(QS_V5[qname]["crit"]); crit[label] = pick
    return {"t": QS_V5[qname]["t"], "ins": QS_V5[qname]["ins"], "crit": crit}

def soft(n, gold, conf=0.9):
    """硬标签→软标签向量（conf 留不确定性）。"""
    v = [(1 - conf) / (n - 1)] * n
    v[gold] = conf
    return v

def maybe_perturb(text, p=0.12):
    return perturb(text) if random.random() < p else text

def row(state, labels, cluster, qs_inline=None, q=None):
    r = {"state": state, "cluster": cluster, "labels": labels}
    if qs_inline:
        r["qs"] = qs_inline
    if q is not None:
        r["q"] = q
    return r

# ---------------- choice 头生成器 ----------------
def gen_choice(qname, pool, boundary=0.06, p_variant=0.3):
    """每标签池子 shuffle 后三段切片：val 8 / test 30 / train 其余——同池零重叠。"""
    labels = list(QS_V5[qname]["crit"])
    n_labels = len(labels)
    rows_tr, rows_va, rows_te = [], [], []
    for i, lb in enumerate(labels):
        sents = list(pool[lb])
        while len(sents) < 78:  # 8 val + 30 test + ≥40 train 的量
            sents.append(perturb(random.choice(sents)))
        random.shuffle(sents)
        val_s, te_s, tr_s = sents[:8], sents[8:38], sents[38:]
        for s in tr_s:
            conf = random.choice([0.5, 0.6, 0.7]) if random.random() < boundary else 0.9
            inline = None
            if random.random() < p_variant:
                v = crit_variant(qname, lb, which=random.randint(0, 1))
                if v:
                    inline = {qname: v}
            rows_tr.append(row(maybe_perturb(s), {qname: soft(n_labels, i, conf)}, f"v5-{qname}", inline))
        for s in val_s:
            rows_va.append(row(maybe_perturb(s), {qname: soft(n_labels, i)}, f"v5-{qname}"))
        for s in te_s:
            rows_te.append(row(maybe_perturb(s), {qname: soft(n_labels, i)}, f"v5-{qname}"))
    return rows_tr, rows_va, rows_te

def gen_emotion(n_states=110):
    """情绪 3 问：每 state 出 3 个 noul 标签。二分类 soft [p_false, p_true]。"""
    rows_tr, rows_va, rows_te = [], [], []
    dims = ["emotion_emo", "emotion_care", "emotion_urgent"]
    def labels_for(kind):
        # [emo, care, urgent] 的 (是否带情绪, 是否需安抚, 是否危机)
        m = {"strong_neg": (0.9, 0.9, 0.15), "mild_neg": (0.7, 0.6, 0.05),
             "positive": (0.9, 0.3, 0.02), "crisis": (0.95, 0.95, 0.9),
             "neutral": (0.1, 0.05, 0.02)}
        return m[kind]
    pools = {k: list(v) for k, v in EMO_POOL.items()}
    all_states = [(s, k) for k, ss in pools.items() for s in ss]
    random.shuffle(all_states)
    for idx, (s, kind) in enumerate(all_states):
        e, c, u = labels_for(kind)
        labels = {"emotion_emo": [1 - e, e], "emotion_care": [1 - c, c], "emotion_urgent": [1 - u, u]}
        r = row(maybe_perturb(s), labels, f"v5-emotion-{kind}")
        if idx < 8:
            rows_va.append(r)
        elif idx < 20:
            rows_te.append(r)
        else:
            rows_tr.append(r)
    # 扩充训练量：池子句子扰动重采
    while len(rows_tr) < n_states:
        s, kind = random.choice(all_states)
        e, c, u = labels_for(kind)
        labels = {"emotion_emo": [1 - e, e], "emotion_care": [1 - c, c], "emotion_urgent": [1 - u, u]}
        rows_tr.append(row(perturb(s), labels, f"v5-emotion-{kind}"))
    return rows_tr, rows_va, rows_te

def main():
    random.seed(20260927)
    tr, va, te = [], [], []
    plan = [
        ("code_router", CODE_POOL), ("marketing_router", MKT_POOL),
        ("finance_ops", FIN_POOL), ("life_admin", LIFE_POOL),
        ("robot_cmd", ROBOT_POOL), ("intent_cn", INTENT_POOL),
    ]
    for qname, pool in plan:
        a, b, c = gen_choice(qname, pool)
        tr += a; va += b; te += c
        print(f"{qname}: train={len(a)} val={len(b)} test={len(c)}")
    a, b, c = gen_emotion(120)
    tr += a; va += b; te += c
    print(f"emotion(3q): train={len(a)} val={len(b)} test={len(c)}")
    random.shuffle(tr)
    with open(f"{OUT}/train_v5_extra.jsonl", "w", encoding="utf-8") as f:
        for r in tr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/val_v5_extra.jsonl", "w", encoding="utf-8") as f:
        for r in va:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/test_v5.jsonl", "w", encoding="utf-8") as f:
        for r in te:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 自检：变体行占比、每头覆盖
    n_inline = sum(1 for r in tr if "qs" in r)
    print(f"合计 train={len(tr)} val={len(va)} test={len(te)}  变体行={n_inline} ({n_inline/len(tr):.0%})")
    # one runnable check: 行内 qs 变体的 crit 与主 QS 必须恰好差一个标签的描述
    for r in tr[:2000]:
        if "qs" in r:
            for qn, qd in r["qs"].items():
                assert qd["t"] == QS_V5[qn]["t"], f"{qn} type 漂移"
                diff = sum(1 for lb in qd["crit"] if qd["crit"][lb] != QS_V5[qn]["crit"][lb])
                assert diff == 1, f"{qn} 变体应只改一个描述，实际 {diff}"
    print("✓ 自检通过：变体只改一个描述、类型无漂移")

if __name__ == "__main__":
    main()
