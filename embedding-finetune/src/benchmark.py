"""标准化 Embedding 检索 Benchmark：召回、延迟、模型大小。"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util


BATCH_SIZE = 32
WARMUP_QUERIES = 10
RECALL_KS = [1, 3, 5, 10]
NDCG_K = 10


@dataclass
class LatencyMetrics:
    corpus_encode_ms: float
    query_encode_total_ms: float
    query_encode_ms_per_query: float
    query_encode_p50_ms: float
    query_encode_p95_ms: float
    search_total_ms: float
    search_ms_per_query: float
    end_to_end_ms_per_query: float
    throughput_qps: float


@dataclass
class RetrievalMetrics:
    num_queries: int
    num_corpus: int
    recall_at_k: Dict[str, float]
    mrr: float
    ndcg_at_10: float
    failed_at_1: int
    success_rate_at_1: float


@dataclass
class ModelInfo:
    name: str
    label: str
    parameters: int
    disk_size_mb: float
    embedding_dim: int
    device: str


@dataclass
class BenchmarkResult:
    model: ModelInfo
    retrieval: RetrievalMetrics
    latency: LatencyMetrics
    config: Dict = field(default_factory=dict)


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_model_disk_size_mb(model_path: Path) -> float:
    if not model_path.exists():
        return 0.0
    total = 0
    for p in model_path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 2)


def count_parameters(model: SentenceTransformer) -> int:
    return sum(p.numel() for p in model.parameters())


def get_embedding_dim(model: SentenceTransformer) -> int:
    if hasattr(model, "get_embedding_dimension"):
        dim = model.get_embedding_dimension()
    else:
        dim = model.get_sentence_embedding_dimension()
    return int(dim) if dim else 0


def ndcg_at_k(rank: int, k: int = 10) -> float:
    if rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def encode_with_timing(
    model: SentenceTransformer,
    texts: List[str],
    encode_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, float, List[float]]:
    kwargs = {
        "batch_size": BATCH_SIZE,
        "show_progress_bar": False,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }
    if encode_kwargs:
        kwargs.update(encode_kwargs)

    # 逐条计时（用于 p50/p95）
    per_item_ms: List[float] = []
    all_embeddings = []
    t0 = time.perf_counter()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        bt0 = time.perf_counter()
        emb = model.encode(batch, **kwargs)
        bt1 = time.perf_counter()
        batch_ms = (bt1 - bt0) * 1000
        per_query = batch_ms / len(batch)
        per_item_ms.extend([per_query] * len(batch))
        all_embeddings.append(emb)

    total_ms = (time.perf_counter() - t0) * 1000
    return np.vstack(all_embeddings), total_ms, per_item_ms


def run_retrieval(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_ids: List[str],
    test_cases: List[dict],
    id_to_idx: Dict[str, int],
) -> Tuple[RetrievalMetrics, float]:
    recalls = {k: [] for k in RECALL_KS}
    reciprocal_ranks: List[float] = []
    ndcgs: List[float] = []
    failed = 0

    t0 = time.perf_counter()
    for idx, case in enumerate(test_cases):
        scores = util.cos_sim(query_embeddings[idx], doc_embeddings)[0].cpu().numpy()
        ranked = np.argsort(scores)[::-1]
        rel_idx = id_to_idx[case["relevant_doc_id"]]
        rank_pos = np.where(ranked == rel_idx)[0]
        rank = int(rank_pos[0]) + 1 if len(rank_pos) else len(doc_ids) + 1

        reciprocal_ranks.append(1.0 / rank)
        ndcgs.append(ndcg_at_k(rank, NDCG_K))
        if rank > 1:
            failed += 1
        for k in RECALL_KS:
            recalls[k].append(rank <= k)

    search_ms = (time.perf_counter() - t0) * 1000
    n = len(test_cases)
    metrics = RetrievalMetrics(
        num_queries=n,
        num_corpus=len(doc_ids),
        recall_at_k={f"recall@{k}": sum(recalls[k]) / n for k in RECALL_KS},
        mrr=float(np.mean(reciprocal_ranks)),
        ndcg_at_10=float(np.mean(ndcgs)),
        failed_at_1=failed,
        success_rate_at_1=(n - failed) / n,
    )
    return metrics, search_ms


def warmup(model: SentenceTransformer, encode_kwargs: Optional[dict] = None) -> None:
    dummy = ["warmup query"] * WARMUP_QUERIES
    kwargs = {"batch_size": BATCH_SIZE, "show_progress_bar": False, "normalize_embeddings": True}
    if encode_kwargs:
        kwargs.update(encode_kwargs)
    model.encode(dummy, **kwargs)


def benchmark_model(
    model: SentenceTransformer,
    label: str,
    model_path: Path,
    corpus: List[dict],
    test_cases: List[dict],
    query_encode_kwargs: Optional[dict] = None,
    doc_encode_kwargs: Optional[dict] = None,
) -> BenchmarkResult:
    doc_ids = [d["id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]
    queries = [c["query"] for c in test_cases]
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    warmup(model, query_encode_kwargs)
    warmup(model, doc_encode_kwargs)

    doc_emb, corpus_ms, _ = encode_with_timing(model, doc_texts, doc_encode_kwargs)
    query_emb, query_total_ms, query_per_ms = encode_with_timing(model, queries, query_encode_kwargs)
    retrieval, search_ms = run_retrieval(query_emb, doc_emb, doc_ids, test_cases, id_to_idx)

    per_query_encode = query_total_ms / len(queries)
    per_query_search = search_ms / len(queries)
    e2e_per_query = (corpus_ms / len(queries)) + per_query_encode + per_query_search

    latency = LatencyMetrics(
        corpus_encode_ms=round(corpus_ms, 2),
        query_encode_total_ms=round(query_total_ms, 2),
        query_encode_ms_per_query=round(per_query_encode, 4),
        query_encode_p50_ms=round(float(np.percentile(query_per_ms, 50)), 4),
        query_encode_p95_ms=round(float(np.percentile(query_per_ms, 95)), 4),
        search_total_ms=round(search_ms, 2),
        search_ms_per_query=round(per_query_search, 4),
        end_to_end_ms_per_query=round(e2e_per_query, 4),
        throughput_qps=round(1000.0 / per_query_encode, 2) if per_query_encode > 0 else 0,
    )

    device = str(next(model.parameters()).device)
    info = ModelInfo(
        name=str(model_path) if model_path.exists() else label,
        label=label,
        parameters=count_parameters(model),
        disk_size_mb=get_model_disk_size_mb(model_path) if model_path.exists() else 0.0,
        embedding_dim=get_embedding_dim(model),
        device=device,
    )

    return BenchmarkResult(
        model=info,
        retrieval=retrieval,
        latency=latency,
        config={
            "batch_size": BATCH_SIZE,
            "warmup_queries": WARMUP_QUERIES,
            "normalize_embeddings": True,
            "similarity": "cosine",
            "query_encode_kwargs": query_encode_kwargs or {},
            "doc_encode_kwargs": doc_encode_kwargs or {},
        },
    )


def resolve_model_path(path_or_name: str, root: Path) -> Path:
    local = root / path_or_name
    if local.exists():
        return local
    cache_name = path_or_name.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{cache_name}"
    if cache_root.exists():
        snapshots = sorted((cache_root / "snapshots").glob("*"))
        if snapshots:
            return snapshots[-1]
    return local


def load_st_model(path_or_name: str, root: Path, qwen3: bool = False) -> Tuple[SentenceTransformer, Path]:
    model_path = resolve_model_path(path_or_name, root)
    resolved = str(model_path) if model_path.exists() else path_or_name

    if qwen3:
        model = SentenceTransformer(
            resolved,
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"padding_side": "left"},
        )
    else:
        model = SentenceTransformer(resolved)
    disk_path = model_path if model_path.exists() else resolve_model_path(path_or_name, root)
    return model, disk_path


MODEL_SPECS = [
    {
        "key": "baseline",
        "label": "BGE-small (基线)",
        "path": "BAAI/bge-small-zh-v1.5",
        "qwen3": False,
    },
    {
        "key": "finetuned",
        "label": "BGE-small (领域微调)",
        "path": "models/bge-small-zh-domain",
        "qwen3": False,
    },
    {
        "key": "hard_neg",
        "label": "BGE-small (Hard Neg 微调)",
        "path": "models/bge-small-zh-hard-neg",
        "qwen3": False,
    },
    {
        "key": "finetuned_1k",
        "label": "BGE-small (1000条微调)",
        "path": "models/bge-small-zh-1k",
        "qwen3": False,
    },
    {
        "key": "qwen3",
        "label": "Qwen3-Embedding-0.6B",
        "path": "Qwen/Qwen3-Embedding-0.6B",
        "qwen3": True,
        "query_kwargs": {"prompt_name": "query"},
        "doc_kwargs": {},
    },
]


def print_comparison_table(results: List[BenchmarkResult]) -> None:
    headers = ["模型", "参数量", "磁盘MB", "维度", "R@1", "R@5", "R@10", "MRR", "NDCG@10", "ms/query", "QPS"]
    rows = []
    for r in results:
        rows.append(
            [
                r.model.label,
                f"{r.model.parameters/1e6:.1f}M",
                f"{r.model.disk_size_mb:.1f}",
                str(r.model.embedding_dim),
                f"{r.retrieval.recall_at_k['recall@1']:.2%}",
                f"{r.retrieval.recall_at_k['recall@5']:.2%}",
                f"{r.retrieval.recall_at_k['recall@10']:.2%}",
                f"{r.retrieval.mrr:.4f}",
                f"{r.retrieval.ndcg_at_10:.4f}",
                f"{r.latency.query_encode_ms_per_query:.2f}",
                f"{r.latency.throughput_qps:.1f}",
            ]
        )

    widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = " | "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="四模型标准化 Benchmark")
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--benchmark", default="data/benchmark_1k.json")
    parser.add_argument("--output", default="results/benchmark_1k_report.json")
    parser.add_argument("--models", nargs="*", default=None, help="指定模型 key: baseline finetuned hard_neg qwen3")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    corpus = load_json(root / args.corpus)
    test_cases = load_json(root / args.benchmark)

    specs = MODEL_SPECS
    if args.models:
        keys = set(args.models)
        specs = [s for s in MODEL_SPECS if s["key"] in keys]

    results: List[BenchmarkResult] = []
    for spec in specs:
        print(f"\n>>> Benchmark: {spec['label']}")
        model, model_path = load_st_model(spec["path"], root, qwen3=spec.get("qwen3", False))
        result = benchmark_model(
            model=model,
            label=spec["label"],
            model_path=model_path,
            corpus=corpus,
            test_cases=test_cases,
            query_encode_kwargs=spec.get("query_kwargs"),
            doc_encode_kwargs=spec.get("doc_kwargs"),
        )
        results.append(result)
        print(
            f"    R@1={result.retrieval.recall_at_k['recall@1']:.2%} "
            f"MRR={result.retrieval.mrr:.4f} "
            f"latency={result.latency.query_encode_ms_per_query:.2f}ms/q"
        )
        del model

    report = {
        "benchmark_config": {
            "num_queries": len(test_cases),
            "num_corpus": len(corpus),
            "batch_size": BATCH_SIZE,
            "device_note": results[0].model.device if results else "cpu",
            "metrics": ["recall@1/3/5/10", "mrr", "ndcg@10", "latency", "throughput", "model_size"],
        },
        "results": [asdict(r) for r in results],
        "ranking": {
            "by_recall_at_1": sorted(
                [(r.model.label, r.retrieval.recall_at_k["recall@1"]) for r in results],
                key=lambda x: x[1],
                reverse=True,
            ),
            "by_mrr": sorted(
                [(r.model.label, r.retrieval.mrr) for r in results],
                key=lambda x: x[1],
                reverse=True,
            ),
            "by_speed": sorted(
                [(r.model.label, r.latency.query_encode_ms_per_query) for r in results],
                key=lambda x: x[1],
            ),
        },
    }

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig") as f:
        f.write(
            "model,parameters_M,disk_MB,embedding_dim,recall@1,recall@3,recall@5,recall@10,"
            "mrr,ndcg@10,failed@1,query_ms,p50_ms,p95_ms,search_ms,qps,e2e_ms\n"
        )
        for r in results:
            f.write(
                f"{r.model.label},{r.model.parameters/1e6:.2f},{r.model.disk_size_mb},"
                f"{r.model.embedding_dim},"
                f"{r.retrieval.recall_at_k['recall@1']:.4f},"
                f"{r.retrieval.recall_at_k['recall@3']:.4f},"
                f"{r.retrieval.recall_at_k['recall@5']:.4f},"
                f"{r.retrieval.recall_at_k['recall@10']:.4f},"
                f"{r.retrieval.mrr:.4f},{r.retrieval.ndcg_at_10:.4f},"
                f"{r.retrieval.failed_at_1},"
                f"{r.latency.query_encode_ms_per_query:.4f},"
                f"{r.latency.query_encode_p50_ms:.4f},"
                f"{r.latency.query_encode_p95_ms:.4f},"
                f"{r.latency.search_ms_per_query:.4f},"
                f"{r.latency.throughput_qps:.2f},"
                f"{r.latency.end_to_end_ms_per_query:.4f}\n"
            )

    print("\n" + "=" * 90)
    print("Benchmark 对比总表 (1000 queries)")
    print("=" * 90)
    print_comparison_table(results)
    print(f"\n完整报告: {output}")
    print(f"CSV 对比表: {csv_path}")


if __name__ == "__main__":
    main()
