# 私有领域 Embedding 微调示例

> 完整实验记录见 **[docs/实验记录.md](docs/实验记录.md)**（含 1000 条 Benchmark、五模型对比、结论与复现命令）。

基于 **BAAI/bge-small-zh-v1.5** 的完整流程：基线检索评估 → 收集失败用例 → 微调 → 复测对比。

## 场景说明

模拟企业内部知识库（电商/FinTech），包含：

- 内部缩写：SLA、CTR、OMS、ES
- 产品编码：SKU-A7821
- 系统专名：Apollo 配置中心
- 口语 vs 正式表述："怎么退钱" vs "退货退款流程"
- 领域近义混淆："财务风控" vs "交易反欺诈"

通用开源 embedding 在这些场景下 Recall@1 往往偏低，需要通过 **query-正样本文档对** 做领域微调。

## 选用模型

| 项目 | 说明 |
|------|------|
| 模型 | [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |
| 维度 | 512 |
| 优势 | MTEB 中文检索榜单表现好，体积小，CPU 可跑 |
| 微调损失 | MultipleNegativesRankingLoss（in-batch negatives） |

## 目录结构

```
finetune/
├── data/
│   ├── corpus.json        # 私有知识库文档（15条）
│   ├── test_cases.json    # 检索测试用例（12条，含失败场景标注）
│   └── train_pairs.json   # 微调正样本对（由失败场景扩展）
├── src/
│   ├── evaluate.py        # Recall@K、MRR 评估
│   ├── baseline_test.py   # 基线测试 + 导出失败用例
│   ├── finetune.py        # 领域微调
│   └── compare_results.py # 微调前后对比
├── run_pipeline.py        # 一键运行
└── results/               # 评估指标输出
```

## 快速开始

```bash
pip install -r requirements.txt
python run_pipeline.py
```

分步运行：

```bash
python src/baseline_test.py
python src/finetune.py
python src/compare_results.py
```

## 实测结果（示例数据）

| 阶段 | Recall@1 | Recall@3 | MRR |
|------|----------|----------|-----|
| 基线 BGE | 66.67% | 75.00% | 0.729 |
| 普通微调 | 83.33% | 100% | 0.917 |
| **Hard neg 微调** | **100%** | **100%** | **1.000** |

基线 4 条失败用例修复路径：

| Query | 基线 rank | 普通微调 | Hard neg 微调 |
|-------|-----------|----------|---------------|
| 怎么拦住刷单和黑产注册 | 4 | 1 | 1 |
| 列表里用户会不会点进去 | 13 | 1 | 1 |
| 包裹发到哪了在哪查进度 | 12 | 2 | **1** |
| 搜索越来越慢是不是索引有问题 | 3 | 2 | **1** |

## 1000 条标准化 Benchmark

生成 1000 条测试 query 并对四模型做统一评测：

```bash
python run_benchmark.py
```

### 评测协议（保证可比性）

| 项目 | 设置 |
|------|------|
| 测试集 | `data/benchmark_1k.json`（1000 条，覆盖 15 个主文档，每文档 66~67 条） |
| 语料库 | 25 篇文档（含 10 篇干扰文档） |
| 相似度 | Cosine（向量 L2 归一化） |
| Batch size | 32（所有模型一致） |
| Warmup | 10 条 query |
| 设备 | CPU（同机同环境） |
| Qwen3 | query 使用 `prompt_name="query"`，document 无 prompt |

### 指标说明

- **Recall@K**：正确文档出现在 Top-K 的比例
- **MRR**：正确文档平均倒数排名
- **NDCG@10**：排序质量（单 relevant 文档标准 NDCG）
- **query_ms**：单条 query 编码耗时（均值 / P50 / P95）
- **QPS**：query 编码吞吐（1000 / 总编码秒数）
- **disk_MB / parameters**：模型磁盘占用与参数量

### 1000 条数据重新微调

用 `benchmark_1k.json` 中全部 1000 条 query-doc 对从基线重新训练：

```bash
python run_finetune_1k.py
```

训练数据：`data/train_1k.json`（1000 对）→ 模型：`models/bge-small-zh-1k`

> **注意**：若在同一 1000 条上评测，属于 in-distribution 测试（指标偏高）；独立难例见 `data/test_cases.json`（12 条 holdout）。

### 实测结果摘要（1000 queries, CPU）

| 模型 | 参数量 | 磁盘 | R@1 | R@3 | R@5 | MRR | NDCG@10 | ms/query | QPS |
|------|--------|------|-----|-----|-----|-----|---------|----------|-----|
| BGE-small 基线 | 24M | 92MB | 88.0% | 95.5% | 96.8% | 0.921 | 0.936 | 2.6 | 384 |
| BGE-small 领域微调 | 24M | 92MB | 94.6% | 99.5% | 99.5% | 0.970 | 0.977 | 2.5 | 406 |
| BGE-small Hard Neg (41条) | 24M | 92MB | 94.9% | 99.4% | 99.5% | 0.972 | 0.979 | 2.5 | 406 |
| **BGE-small (1000条微调)** | 24M | 92MB | **97.9%** | **100%** | **100%** | **0.990** | **0.992** | 2.6 | 388 |
| Qwen3-Embedding-0.6B | 596M | 1152MB | 96.4% | 99.7% | 100% | 0.980 | 0.985 | 878 | 1.1 |

### 真实泛化评测（500 条未见 query，与训练集 0 重叠）

```bash
python run_eval_unseen_500.py
```

| 模型 | Recall@1 | Recall@3 | Recall@5 | MRR | 说明 |
|------|----------|----------|----------|-----|------|
| BGE-small 基线 | 76.4% | 89.8% | 93.8% | 0.842 | 零样本 |
| 25条领域微调 | 82.0% | 94.8% | 96.6% | 0.885 | 25 条训练 |
| 41条 Hard Neg | 83.0% | 95.0% | 96.6% | 0.893 | 41 条训练 |
| 1000条微调 | 84.6% | 93.6% | 95.8% | 0.897 | 未见集 +8.2% vs 基线 |
| Qwen3-0.6B | **88.6%** | **98.2%** | **100%** | **0.934** | 未见集最高 |

详见 `docs/实验记录.md` §6.5。

| 模型 | 训练数据量 | Recall@1 |
|------|-----------|----------|
| BGE-small 基线 | 0 | 66.7% |
| 25条领域微调 | 25 | 83.3% |
| 41条 Hard Neg | 41 | **100%** |
| **1000条微调** | **1000** | **100%** |
| Qwen3-0.6B | 0 | 91.7% |

完整报告：`results/benchmark_1k_report.json` / `results/benchmark_1k_report.csv`

### Qwen3-Embedding-0.6B vs Hard neg 微调

对比更强通用模型与领域微调模型：

```bash
python run_compare_qwen3.py
```

> Qwen3 需要 Python 3.9+ 与 `transformers>=4.51`。脚本默认使用 `learn-loop` conda 环境；也可手动：
>
> ```bash
> conda activate learn-loop
> pip install "transformers>=4.51" "sentence-transformers>=3.2"
> python src/compare_qwen3.py
> ```

| 模型 | 参数量 | Recall@1 | MRR | 失败数 |
|------|--------|----------|-----|--------|
| 基线 BGE-small | ~33M | 66.67% | 0.729 | 4 |
| **Qwen3-Embedding-0.6B** | ~600M | **91.67%** | 0.938 | 1 |
| **BGE Hard neg 微调** | ~33M | **100%** | 1.000 | 0 |

结论（本私有领域测试集）：

- Qwen3 零样本强于 BGE-small 基线（+25% Recall@1），但仍会失败于口语 CTR 查询
- Hard neg 微调的小模型在领域检索上**超过**未微调的 Qwen3-0.6B
- 仅 Hard neg 能命中的 case：`列表里用户会不会点进去`（口语 CTR vs 用户画像混淆）

### Hard negative 微调（进阶）

从基线/普通微调失败用例自动构造三元组，并继续训练：

```bash
python run_hard_neg_pipeline.py
```

分步：

```bash
python src/build_hard_neg_data.py      # 生成 data/train_triplets.json
python src/finetune_hard_neg.py        # 在领域模型上继续 hard neg 微调
python src/compare_hard_neg.py         # 三阶段对比报告
```

> 国内下载模型需设置 `HF_ENDPOINT=https://hf-mirror.com`，`run_pipeline.py` 已默认配置。

## 典型失败场景（私有数据）

| 类别 | 示例 query | 难点 |
|------|-----------|------|
| abbreviation | SLA指标要求是什么 | 缩写与全称语义 gap |
| colloquial | 怎么退钱给消费者 | 口语与制度文档表述不一致 |
| product_code | A7821网关是干什么的 | 内部编码关联弱 |
| domain_synonym | 发票和对公转账合规要求 | 与"反欺诈"等近域文档混淆 |
| entity_disambiguation | 生产环境改配置走哪个系统 | Apollo 等内部系统名 |
| domain_jargon | 新用户没有行为数据怎么推荐 | 领域术语冷启动 |

## 微调数据构造建议

1. 从基线 **Recall@1 失败** 的 query 出发
2. 为每个 query 构造 1~2 条 `(anchor, positive)` 对
3. anchor 可包含：原始 query、口语变体、缩写形式
4. positive 为对应知识库段落原文
5. 样本量较小时 3~5 epoch 即可，注意过拟合

## 扩展到真实业务

- 将 `corpus.json` 替换为真实 Wiki/工单/产品文档
- 用线上搜索日志挖掘 `(query, clicked_doc)` 作为训练对
- 增加 hard negatives（误召回文档）可进一步提升区分度
- 数据量大时可换 `bge-base-zh-v1.5` 或 `bge-large-zh-v1.5`
