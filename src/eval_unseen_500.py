"""在 500 条全新未见测试集上对比各模型（不改动现有训练数据）。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from evaluate import evaluate_retrieval, print_metrics

MODEL_SPECS = [
    ("BGE-small (基线)", "BAAI/bge-small-zh-v1.5", False, None, None),
    ("BGE-small (25条领域微调)", "models/bge-small-zh-domain", False, None, None),
    ("BGE-small (41条 Hard Neg)", "models/bge-small-zh-hard-neg", False, None, None),
    ("BGE-small (1000条微调)", "models/bge-small-zh-1k", False, None, None),
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", True, {"prompt_name": "query"}, {}),
]


def load_model(path: str, root: Path, qwen3: bool) -> SentenceTransformer:
    p = root / path
    resolved = str(p) if p.exists() else path
    if qwen3:
        return SentenceTransformer(
            resolved,
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"padding_side": "left"},
        )
    return SentenceTransformer(resolved)


def print_table(results: List[dict]) -> None:
    headers = ["模型", "R@1", "R@3", "R@5", "MRR", "NDCG@10", "失败数", "ms/q"]
    rows = []
    for r in results:
        m = r["retrieval"]
        lat = r["latency"]
        rows.append([
            r["label"],
            f"{m['recall_at_k']['recall@1']:.2%}",
            f"{m['recall_at_k']['recall@3']:.2%}",
            f"{m['recall_at_k']['recall@5']:.2%}",
            f"{m['mrr']:.4f}",
            f"{m['ndcg_at_10']:.4f}",
            str(m["failed_at_1"]),
            f"{lat['query_encode_ms_per_query']:.2f}",
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = " | "
    print(sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * sum(widths) + "-" * (3 * (len(headers) - 1)))
    for row in rows:
        print(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark_500_unseen.json")
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--output", default="results/benchmark_500_unseen_report.json")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    root = Path(__file__).resolve().parent.parent
    corpus_path = root / args.corpus
    test_path = root / args.benchmark
    output_path = root / args.output

    if not test_path.exists():
        raise FileNotFoundError(f"请先生成测试集: python src/generate_benchmark_500_unseen.py")

    # 延迟指标（简化：仅 query 编码均值）
    from benchmark import benchmark_model, LatencyMetrics, RetrievalMetrics, ModelInfo

    results = []
    for label, path, qwen, qkw, dkw in MODEL_SPECS:
        print(f"\n>>> {label}")
        model = load_model(path, root, qwen)
        p = root / path
        disk_path = p if p.exists() else Path(path)
        from benchmark import resolve_model_path, count_parameters, get_model_disk_size_mb, get_embedding_dim

        br = benchmark_model(
            model=model,
            label=label,
            model_path=resolve_model_path(path, root) if not p.exists() else p,
            corpus=json.loads(corpus_path.read_text(encoding="utf-8")),
            test_cases=json.loads(test_path.read_text(encoding="utf-8")),
            query_encode_kwargs=qkw,
            doc_encode_kwargs=dkw,
        )
        print(
            f"    R@1={br.retrieval.recall_at_k['recall@1']:.2%} "
            f"MRR={br.retrieval.mrr:.4f} failed={br.retrieval.failed_at_1}"
        )
        results.append({"label": label, "retrieval": asdict(br.retrieval), "latency": asdict(br.latency), "model": asdict(br.model)})
        del model

    report = {
        "description": "500条全新未见测试集，与 train/benchmark_1k/test_cases 无 query 重叠",
        "test_file": str(args.benchmark),
        "num_queries": len(json.loads(test_path.read_text(encoding="utf-8"))),
        "note": "真实泛化评测：模型均未在此500条上训练",
        "results": results,
        "ranking_by_recall_at_1": sorted(
            [(r["label"], r["retrieval"]["recall_at_k"]["recall@1"]) for r in results],
            key=lambda x: x[1],
            reverse=True,
        ),
        "ranking_by_recall_at_3": sorted(
            [(r["label"], r["retrieval"]["recall_at_k"]["recall@3"]) for r in results],
            key=lambda x: x[1],
            reverse=True,
        ),
        "ranking_by_recall_at_5": sorted(
            [(r["label"], r["retrieval"]["recall_at_k"]["recall@5"]) for r in results],
            key=lambda x: x[1],
            reverse=True,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig") as f:
        f.write("model,recall@1,recall@3,recall@5,recall@10,mrr,ndcg@10,failed@1,query_ms,qps\n")
        for r in results:
            m, lat = r["retrieval"], r["latency"]
            f.write(
                f"{r['label']},{m['recall_at_k']['recall@1']:.4f},"
                f"{m['recall_at_k']['recall@3']:.4f},{m['recall_at_k']['recall@5']:.4f},"
                f"{m['recall_at_k']['recall@10']:.4f},{m['mrr']:.4f},{m['ndcg_at_10']:.4f},"
                f"{m['failed_at_1']},{lat['query_encode_ms_per_query']:.4f},{lat['throughput_qps']:.2f}\n"
            )

    print("\n" + "=" * 80)
    print("500 条未见测试集对比（真实泛化）")
    print("=" * 80)
    print_table(results)
    print(f"\nJSON: {output_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
