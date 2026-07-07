"""生成 1000 条标准化检索 benchmark 测试集。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


# 每个主文档的 query 模板池（正式 / 口语 / 缩写 / 同义）
DOC_QUERY_TEMPLATES: Dict[str, List[dict]] = {
    "doc_001": [
        ("核心接口挂了最多能修多久", "colloquial"),
        ("SLA指标要求是什么", "abbreviation"),
        ("故障恢复时间RTO是多少", "abbreviation"),
        ("接口可用性要达到多少", "formal"),
        ("P99响应时间限制", "abbreviation"),
        ("服务等级协议怎么规定的", "formal"),
        ("交易系统SLA标准", "abbreviation"),
        ("最多停机多久必须恢复", "colloquial"),
        ("核心接口可用性指标", "formal"),
        ("RTO故障恢复时限", "abbreviation"),
    ],
    "doc_002": [
        ("买家不想买了怎么把钱退回去", "colloquial"),
        ("退货退款流程是什么", "formal"),
        ("原路退款要多久到账", "colloquial"),
        ("售后申请怎么提", "colloquial"),
        ("退款审批流程", "formal"),
        ("用户取消订单如何退款", "formal"),
        ("仓库签收后多久退款", "colloquial"),
        ("财务退款时效", "formal"),
        ("消费者退钱流程", "colloquial"),
        ("退货后资金退回方式", "formal"),
    ],
    "doc_003": [
        ("7821那台风控盒子是干嘛的", "colloquial"),
        ("SKU-A7821网关功能", "product_code"),
        ("A7821智能风控网关", "product_code"),
        ("风控网关日均处理量", "formal"),
        ("规则引擎和机器学习联合决策", "formal"),
        ("实时拦截高风险交易", "formal"),
        ("智能风控网关介绍", "formal"),
        ("7821产品规格", "product_code"),
        ("风控网关部署说明", "formal"),
        ("高风险交易拦截系统", "domain_synonym"),
    ],
    "doc_004": [
        ("怎么拦住刷单和黑产注册", "colloquial"),
        ("交易反欺诈系统原理", "formal"),
        ("如何识别盗刷和羊毛党", "domain_synonym"),
        ("设备指纹风控", "formal"),
        ("虚假注册检测", "formal"),
        ("行为序列反欺诈", "formal"),
        ("关联图谱风险识别", "formal"),
        ("羊毛党拦截方案", "colloquial"),
        ("盗刷交易识别", "colloquial"),
        ("反欺诈模型架构", "formal"),
    ],
    "doc_005": [
        ("线上配置能直接在后台改吗", "colloquial"),
        ("Apollo配置中心用法", "entity"),
        ("生产环境禁止改配置", "formal"),
        ("配置灰度发布流程", "formal"),
        ("微服务配置统一管理", "formal"),
        ("配置回滚怎么做", "colloquial"),
        ("多环境配置隔离", "formal"),
        ("Apollo灰度发布", "entity"),
        ("配置中心规范", "formal"),
        ("能不能直接改生产配置", "colloquial"),
    ],
    "doc_006": [
        ("没人买过东西的新客推什么货", "colloquial"),
        ("推荐系统冷启动策略", "domain_jargon"),
        ("新用户没有行为数据怎么推荐", "domain_jargon"),
        ("热门商品兜底推荐", "formal"),
        ("新商品曝光策略", "formal"),
        ("协同过滤冷启动", "formal"),
        ("新客推荐方案", "colloquial"),
        ("内容特征推荐", "formal"),
        ("冷启动热门兜底", "domain_jargon"),
        ("新用户推荐算法", "formal"),
    ],
    "doc_007": [
        ("列表里用户会不会点进去", "colloquial"),
        ("CTR点击率模型特征", "abbreviation"),
        ("点击率预估用什么结构", "formal"),
        ("Wide Deep模型特征", "abbreviation"),
        ("用户历史行为特征工程", "formal"),
        ("CTR模型训练", "abbreviation"),
        ("商品类目时段渠道特征", "formal"),
        ("点击率预测模型", "formal"),
        ("列表页点击预估", "colloquial"),
        ("CTR预估Wide&Deep", "abbreviation"),
    ],
    "doc_008": [
        ("公司账户打款要准备啥材料", "colloquial"),
        ("财务风控合规要求", "formal"),
        ("对公转账审批流程", "formal"),
        ("发票开具规范", "formal"),
        ("月度关账审计留痕", "formal"),
        ("对公打款合规", "colloquial"),
        ("财务合规制度", "formal"),
        ("企业转账需要什么", "colloquial"),
        ("发票和对公转账", "domain_synonym"),
        ("财务审计要求", "formal"),
    ],
    "doc_009": [
        ("攒的分能不能当现金用", "colloquial"),
        ("会员积分抵扣规则", "formal"),
        ("积分有效期多久", "colloquial"),
        ("100积分等于多少钱", "colloquial"),
        ("积分和促销叠加吗", "colloquial"),
        ("消费积分规则", "formal"),
        ("积分兑换比例", "formal"),
        ("会员积分制度", "formal"),
        ("积分能抵现吗", "colloquial"),
        ("积分过期时间", "colloquial"),
    ],
    "doc_010": [
        ("包裹发到哪了在哪查进度", "colloquial"),
        ("OMS订单管理系统功能", "abbreviation"),
        ("订单拆单规则", "formal"),
        ("物流跟踪在哪看", "colloquial"),
        ("履约路由说明", "formal"),
        ("异常订单处理流程", "formal"),
        ("OMS拆单物流", "abbreviation"),
        ("订单创建和履约", "formal"),
        ("查快递进度系统", "colloquial"),
        ("订单管理系统OMS", "abbreviation"),
    ],
    "doc_011": [
        ("搜索越来越慢是不是索引有问题", "colloquial"),
        ("Elasticsearch慢查询排查", "abbreviation"),
        ("ES索引分片规划", "abbreviation"),
        ("检索集群冷热分层", "formal"),
        ("搜索服务熔断限流", "formal"),
        ("ES集群运维规范", "abbreviation"),
        ("慢查询分析工具", "formal"),
        ("Elasticsearch运维", "abbreviation"),
        ("索引分片怎么规划", "colloquial"),
        ("搜索集群性能优化", "formal"),
    ],
    "doc_012": [
        ("日志里别把用户手机号全打出来", "colloquial"),
        ("数据脱敏规范", "formal"),
        ("手机号中间四位掩码", "formal"),
        ("身份证号脱敏规则", "formal"),
        ("日志禁止输出银行卡号", "formal"),
        ("敏感数据打码要求", "colloquial"),
        ("数据脱敏标准", "formal"),
        ("手机号掩码规则", "colloquial"),
        ("日志脱敏规范", "formal"),
        ("隐私数据脱敏", "formal"),
    ],
    "doc_013": [
        ("Kafka Topic命名规范", "abbreviation"),
        ("消息队列幂等处理", "formal"),
        ("Topic命名格式", "formal"),
        ("消费组幂等要求", "formal"),
        ("Kafka消息规范", "abbreviation"),
        ("业务域事件类型版本命名", "formal"),
        ("MQ消费幂等", "abbreviation"),
        ("消息队列Topic规范", "formal"),
        ("Kafka消费组规范", "abbreviation"),
        ("事件消息命名规则", "formal"),
    ],
    "doc_014": [
        ("新员工怎么开账号", "colloquial"),
        ("LDAP账号开通流程", "abbreviation"),
        ("入职IT指引", "formal"),
        ("VPN申请步骤", "colloquial"),
        ("终端安全软件安装", "formal"),
        ("Wiki和Jira权限申请", "formal"),
        ("新员工IT onboarding", "abbreviation"),
        ("入职需要申请哪些权限", "colloquial"),
        ("IT入职流程", "formal"),
        ("新员工系统权限", "formal"),
    ],
    "doc_015": [
        ("大促前要做容量评估吗", "colloquial"),
        ("大促压测扩容方案", "formal"),
        ("历史QPS容量评估", "formal"),
        ("转化率库存深度评估", "formal"),
        ("弹性扩容评审", "formal"),
        ("大促前14天压测", "formal"),
        ("促销容量规划", "colloquial"),
        ("大促流量预估", "colloquial"),
        ("双11容量准备", "colloquial"),
        ("大促扩容方案评审", "formal"),
    ],
}

# 口语变体后缀/前缀，用于扩充到 1000
PREFIXES = ["请问", "想了解", "帮忙查一下", "有没有文档说明", ""]
SUFFIXES = ["", "？", "吗", "怎么弄", "有哪些要求"]


def expand_query(base: str, rng: random.Random) -> str:
    p = rng.choice(PREFIXES)
    s = rng.choice(SUFFIXES)
    if p and not base.startswith(p):
        base = f"{p}{base}"
    if s and not base.endswith(s):
        base = f"{base}{s}"
    return base


def generate_cases(target: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    cases: List[dict] = []
    doc_ids = list(DOC_QUERY_TEMPLATES.keys())
    per_doc = target // len(doc_ids)
    remainder = target % len(doc_ids)

    for i, doc_id in enumerate(doc_ids):
        quota = per_doc + (1 if i < remainder else 0)
        templates = DOC_QUERY_TEMPLATES[doc_id]
        generated = 0
        attempt = 0
        seen = set()

        while generated < quota and attempt < quota * 10:
            attempt += 1
            base, category = rng.choice(templates)
            query = expand_query(base, rng) if attempt > len(templates) else base
            if query in seen:
                query = expand_query(f"{base}（变体{attempt}）", rng)
            if query in seen:
                continue
            seen.add(query)
            cases.append(
                {
                    "id": f"bench_{len(cases)+1:04d}",
                    "query": query,
                    "relevant_doc_id": doc_id,
                    "category": category,
                }
            )
            generated += 1

    rng.shuffle(cases)
    for idx, case in enumerate(cases):
        case["id"] = f"bench_{idx+1:04d}"
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/benchmark_1k.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    output = root / args.output
    cases = generate_cases(args.count, args.seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    from collections import Counter

    dist = Counter(c["relevant_doc_id"] for c in cases)
    print(f"生成 {len(cases)} 条测试用例 -> {output}")
    print(f"文档覆盖: {len(dist)} 个, 每文档 {min(dist.values())}-{max(dist.values())} 条")


if __name__ == "__main__":
    main()
