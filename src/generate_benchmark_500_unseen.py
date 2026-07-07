"""生成 500 条全新测试用例（与现有 train/test 数据 query 不重叠）。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set


# 全新表述模板（与 benchmark_1k / test_cases / train_pairs 用词不同）
UNSEEN_TEMPLATES: Dict[str, List[tuple]] = {
    "doc_001": [
        ("线上服务挂了恢复时限有啥规定", "colloquial"),
        ("99.95%可用性是哪个文档说的", "formal"),
        ("接口响应P99不能超过多少", "abbreviation"),
        ("故障RTO十五分钟怎么理解", "abbreviation"),
        ("服务等级协议里写了几条硬性指标", "formal"),
        ("交易链路稳定性考核标准", "domain_synonym"),
        ("核心系统中断后多久必须恢复业务", "colloquial"),
        ("SLA文档对延迟的要求", "abbreviation"),
        ("可用性和恢复时间怎么平衡", "formal"),
        ("运维那边说的SLA红线是啥", "colloquial"),
    ],
    "doc_002": [
        ("顾客取消订单后钱怎么返", "colloquial"),
        ("售后单走完了多久能到账", "colloquial"),
        ("退货签收之后财务怎么处理", "formal"),
        ("原路返回大概几个工作日", "colloquial"),
        ("退款是不是一定要走售后流程", "formal"),
        ("用户拒收商品怎么退钱", "colloquial"),
        ("退款审核通过后谁打款", "formal"),
        ("电商退货资金回流规则", "domain_synonym"),
        ("取消交易后的退款路径", "formal"),
        ("仓库验货完成才退款吗", "colloquial"),
    ],
    "doc_003": [
        ("A7821这台设备负责什么业务", "product_code"),
        ("智能网关每天大概扛多少请求", "formal"),
        ("风控网关的规则引擎怎么用", "formal"),
        ("7821型号和风控决策啥关系", "product_code"),
        ("实时风控那台网关的说明", "colloquial"),
        ("SKU风控网关和ML模型怎么配合", "abbreviation"),
        ("拦截可疑交易靠哪套网关", "colloquial"),
        ("A7821部署在哪个环节", "product_code"),
        ("网关层的风控能力有哪些", "formal"),
        ("三千多万日请求是哪款产品", "colloquial"),
    ],
    "doc_004": [
        ("刷单团伙一般怎么被识别出来", "colloquial"),
        ("虚假账号批量注册怎么防", "colloquial"),
        ("设备指纹在风控里起什么作用", "formal"),
        ("行为序列异常能说明什么问题", "formal"),
        ("图谱关联怎么帮助抓黑产", "formal"),
        ("盗卡交易通常有哪些特征", "colloquial"),
        ("羊毛党薅羊毛怎么拦", "colloquial"),
        ("反欺诈平台覆盖哪些场景", "formal"),
        ("风控里说的关联网络是啥", "domain_jargon"),
        ("异常注册和盗刷是不是同一套系统管", "colloquial"),
    ],
    "doc_005": [
        ("配置变更能不能绕过审批直接上生产", "colloquial"),
        ("Apollo怎么做灰度放量", "entity"),
        ("微服务参数统一在哪维护", "formal"),
        ("配置回滚操作步骤", "formal"),
        ("测试环境和生产配置怎么隔离", "formal"),
        ("线上开关改完怎么验证", "colloquial"),
        ("配置中心是否支持多环境", "formal"),
        ("发布配置前要走什么流程", "colloquial"),
        ("Apollo和手工改配置文件区别", "entity"),
        ("禁止直改生产是出于什么考虑", "formal"),
    ],
    "doc_006": [
        ("注册当天没有任何点击怎么推", "colloquial"),
        ("冷启动阶段热门榜怎么选", "domain_jargon"),
        ("新上架商品没有销量怎么办", "colloquial"),
        ("协同过滤在没历史时还用吗", "formal"),
        ("新客首页推荐逻辑是什么", "colloquial"),
        ("内容特征召回用在什么场景", "formal"),
        ("零行为用户推荐兜底策略", "domain_jargon"),
        ("冷启动和内容推荐怎么结合", "formal"),
        ("没人买过的SKU如何获得曝光", "colloquial"),
        ("推荐系统对新用户的默认策略", "formal"),
    ],
    "doc_007": [
        ("feed流里预估用户会不会点广告", "colloquial"),
        ("Wide&Deep模型在我们这用在哪", "abbreviation"),
        ("点击率训练样本取哪些维度", "formal"),
        ("CTR模型里渠道特征怎么建", "abbreviation"),
        ("列表曝光后点击概率谁负责算", "colloquial"),
        ("预估CTR需要哪些用户侧特征", "formal"),
        ("点击模型和转化模型是一回事吗", "colloquial"),
        ("时段特征对点击率影响大吗", "formal"),
        ("类目特征在点击预估里怎么用", "formal"),
        ("点击概率排序靠哪个模型", "colloquial"),
    ],
    "doc_008": [
        ("对公付款需要哪些审批材料", "colloquial"),
        ("公司转账合规审查点有哪些", "formal"),
        ("发票合规和财务制度在哪看", "formal"),
        ("月度结账要注意什么合规项", "formal"),
        ("企业客户打款流程规范", "colloquial"),
        ("对公账户付款谁能批", "colloquial"),
        ("财务侧风控和发票管理", "domain_synonym"),
        ("审计留痕对财务有啥要求", "formal"),
        ("供应商对公转账是否同一流程", "colloquial"),
        ("公对公转账制度文档", "formal"),
    ],
    "doc_009": [
        ("会员分能抵扣现金吗规则在哪", "colloquial"),
        ("积分过期了还能用吗", "colloquial"),
        ("一百积分相当于多少钱", "colloquial"),
        ("积分和优惠券能一起用吗", "colloquial"),
        ("消费多少积一分", "formal"),
        ("积分体系有效期怎么算", "formal"),
        ("忠诚度积分兑换政策", "domain_synonym"),
        ("积分抵扣有没有上限", "colloquial"),
        ("促销期间积分规则变吗", "colloquial"),
        ("会员权益里积分怎么描述", "formal"),
    ],
    "doc_010": [
        ("买家问快递到哪了应该查哪个系统", "colloquial"),
        ("订单履约状态在哪个平台看", "colloquial"),
        ("拆成两个包裹发货在哪配置", "colloquial"),
        ("OMS负责哪些履约环节", "abbreviation"),
        ("异常单比如地址错了谁处理", "colloquial"),
        ("物流节点回传是OMS管吗", "formal"),
        ("订单路由到哪个仓谁决定", "colloquial"),
        ("创建订单到出库的全流程系统", "formal"),
        ("多渠道订单是不是都进OMS", "colloquial"),
        ("履约路由规则文档", "formal"),
    ],
    "doc_011": [
        ("搜索服务响应慢从哪排查", "colloquial"),
        ("ES索引冷热分层什么意思", "abbreviation"),
        ("分片数量规划和性能关系", "formal"),
        ("检索集群限流策略", "formal"),
        ("慢查询日志去哪看", "colloquial"),
        ("ElasticSearch运维手册在哪", "abbreviation"),
        ("全文检索延迟高常见原因", "colloquial"),
        ("索引重建会不会影响查询", "colloquial"),
        ("搜索集群熔断怎么配", "formal"),
        ("ES和MySQL慢查询是一类问题吗", "colloquial"),
    ],
    "doc_012": [
        ("日志输出手机号有什么限制", "colloquial"),
        ("敏感字段脱敏规则汇总", "formal"),
        ("身份证号在报表里怎么显示", "formal"),
        ("银行卡号能不能写进debug日志", "colloquial"),
        ("掩码处理手机号保留几位", "formal"),
        ("数据合规对日志字段的要求", "domain_synonym"),
        ("用户信息在日志里必须隐藏吗", "colloquial"),
        ("脱敏规范适用于哪些系统", "formal"),
        ("中间四位打星号是哪个规范", "colloquial"),
        ("隐私数据写入日志的禁令", "formal"),
    ],
    "doc_013": [
        ("Kafka的Topic名字怎么起", "abbreviation"),
        ("消息消费重复了怎么处理", "colloquial"),
        ("事件命名里版本号放哪", "formal"),
        ("消费组幂等是什么意思", "domain_jargon"),
        ("MQ消息格式有统一规范吗", "abbreviation"),
        ("业务域和事件类型怎么拼Topic", "formal"),
        ("消息队列消费失败重试策略", "colloquial"),
        ("Kafka规范文档在哪", "abbreviation"),
        ("跨服务事件总线命名约定", "formal"),
        ("幂等消费实现要注意啥", "colloquial"),
    ],
    "doc_014": [
        ("新人第一天要开哪些系统账号", "colloquial"),
        ("LDAP和VPN是不是一起申请", "abbreviation"),
        ("入职IT checklist有哪些", "domain_jargon"),
        ("终端安全软件必须装吗", "colloquial"),
        ("Wiki权限找谁开", "colloquial"),
        ("Jira访问权限申请流程", "formal"),
        ("新员工设备接入规范", "formal"),
        ("IT onboarding文档标题", "abbreviation"),
        ("入职后多久能登录内网系统", "colloquial"),
        ("办公电脑安全基线要求", "formal"),
    ],
    "doc_015": [
        ("大促之前容量怎么估", "colloquial"),
        ("压测报告要提前多久出", "formal"),
        ("QPS预估依据哪些历史指标", "formal"),
        ("库存深度和流量峰值关系", "formal"),
        ("弹性扩容方案谁评审", "colloquial"),
        ("促销前14天要做什么准备", "formal"),
        ("双11容量规划文档", "colloquial"),
        ("转化率假设在容量评估里怎么用", "formal"),
        ("峰值流量演练流程", "colloquial"),
        ("大促保障扩容checklist", "domain_jargon"),
    ],
}

UNSEEN_PREFIXES = ["麻烦查下", "想确认", "有没有说明", "帮我找文档", "请教一下", ""]
UNSEEN_SUFFIXES = ["", "？", "可以吗", "有相关文档吗", "谢谢"]


def load_existing_queries(root: Path) -> Set[str]:
    existing: Set[str] = set()
    files = [
        "data/test_cases.json",
        "data/benchmark_1k.json",
        "data/train_pairs.json",
        "data/train_triplets.json",
        "data/train_1k.json",
    ]
    for rel in files:
        path = root / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            q = item.get("query") or item.get("anchor")
            if q:
                existing.add(q)
    return existing


def expand(base: str, rng: random.Random) -> str:
    p = rng.choice(UNSEEN_PREFIXES)
    s = rng.choice(UNSEEN_SUFFIXES)
    text = base
    if p and not text.startswith(p):
        text = f"{p}{text}"
    if s and not text.endswith(s):
        text = f"{text}{s}"
    return text


def generate_unseen(target: int, seed: int, existing: Set[str]) -> List[dict]:
    rng = random.Random(seed)
    doc_ids = list(UNSEEN_TEMPLATES.keys())
    per_doc = target // len(doc_ids)
    remainder = target % len(doc_ids)
    cases: List[dict] = []
    global_seen: Set[str] = set(existing)

    for i, doc_id in enumerate(doc_ids):
        quota = per_doc + (1 if i < remainder else 0)
        templates = UNSEEN_TEMPLATES[doc_id]
        count = 0
        attempt = 0
        local_seen: Set[str] = set()

        while count < quota and attempt < quota * 30:
            attempt += 1
            base, category = rng.choice(templates)
            query = expand(base, rng) if attempt > len(templates) else base
            if attempt > len(templates) * 2:
                query = expand(f"{base}（场景{attempt}）", rng)

            if query in global_seen or query in local_seen:
                continue
            local_seen.add(query)
            global_seen.add(query)
            cases.append(
                {
                    "id": f"unseen_{len(cases)+1:04d}",
                    "query": query,
                    "relevant_doc_id": doc_id,
                    "category": category,
                    "split": "unseen_test",
                }
            )
            count += 1

        if count < quota:
            raise RuntimeError(f"文档 {doc_id} 仅生成 {count}/{quota} 条无重叠 query")

    rng.shuffle(cases)
    for idx, c in enumerate(cases):
        c["id"] = f"unseen_{idx+1:04d}"
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="data/benchmark_500_unseen.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    existing = load_existing_queries(root)
    cases = generate_unseen(args.count, args.seed, existing)

    overlap = [c["query"] for c in cases if c["query"] in existing]
    if overlap:
        raise RuntimeError(f"与现有数据重叠 {len(overlap)} 条，请调整 seed 或模板")

    out = root / args.output
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成 {len(cases)} 条全新测试用例 -> {out}")
    print(f"与现有 {len(existing)} 条 query 重叠: 0")
    from collections import Counter
    dist = Counter(c["relevant_doc_id"] for c in cases)
    print(f"文档覆盖 {len(dist)} 个, 每文档 {min(dist.values())}-{max(dist.values())} 条")


if __name__ == "__main__":
    main()
