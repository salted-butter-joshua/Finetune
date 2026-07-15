"""BGE-M3 vs Qwen3-Embedding-0.6B 零样本对比（当前私有领域数据集）。

评测协议与现有 benchmark 一致：
- Cosine（L2 归一化）
- Batch size = 32
- Warmup = 10
- BGE-M3：dense embedding（SentenceTransformer）
- Qwen3：query 使用 prompt_name="query"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from benchmark import (
    benchmark_model,
    resolve_model_path,
)
from evaluate import evaluate_retrieval, print_metrics

BGE_M3_LOCAL = "models/bge-m3"
BGE_M3_HUB = "BAAI/bge-m3"
QWEN3 = "Qwen/Qwen3-Embedding-0.6B"


def resolve_bge_m3_path(root: Path) -> str:
    local = root / BGE_M3_LOCAL
    # SentenceTransformer 需要 config.json + 权重；权重常为 pytorch_model.bin / model.safetensors
    if (local / "config.json").exists() and (
        (local / "pytorch_model.bin").exists()
        or (local / "model.safetensors").exists()
        or (local / "onnx").exists()
    ):
        return BGE_M3_LOCAL
    return BGE_M3_HUB


def build_model_specs(root: Path) -> List[dict]:
    bge_path = resolve_bge_m3_path(root)
    return [
        {
            "key": "bge_m3",
            "label": "BGE-M3",
            "path": bge_path,
            "qwen3": False,
            "query_kwargs": None,
            "doc_kwargs": None,
            "note": "dense-only via SentenceTransformer; dim=1024",
        },
        {
            "key": "qwen3",
            "label": "Qwen3-Embedding-0.6B",
            "path": QWEN3,
            "qwen3": True,
            "query_kwargs": {"prompt_name": "query"},
            "doc_kwargs": {},
            "note": "query prompt_name=query; dim=1024",
        },
    ]

DATASETS = [
    {
        "key": "hard12",
        "label": "难例集 12 条",
        "path": "data/test_cases.json",
        "note": "与 train_pairs 重叠，仅作难例诊断，不作泛化结论",
    },
    {
        "key": "unseen500",
        "label": "未见泛化 500 条",
        "path": "data/benchmark_500_unseen.json",
        "note": "与全部训练/测试数据 0 query 重叠，推荐主结论",
    },
    {
        "key": "bench1k",
        "label": "Benchmark 1000 条",
        "path": "data/benchmark_1k.json",
        "note": "与 train_1k query 重叠，仅作 in-distribution 参考",
    },
]


def ensure_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_model(path: str, root: Path, qwen3: bool) -> SentenceTransformer:
    local = root / path
    resolved = str(local) if local.exists() else path
    if qwen3:
        return SentenceTransformer(
            resolved,
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"padding_side": "left"},
        )
    return SentenceTransformer(resolved)


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = ["数据集", "模型", "参数量", "维度", "R@1", "R@3", "R@5", "MRR", "NDCG@10", "失败@1", "ms/q", "QPS"]
    table = []
    for r in rows:
        m = r["retrieval"]
        lat = r["latency"]
        info = r["model"]
        table.append(
            [
                r["dataset_label"],
                r["label"],
                f"{info['parameters'] / 1e6:.0f}M",
                str(info["embedding_dim"]),
                f"{m['recall_at_k']['recall@1']:.2%}",
                f"{m['recall_at_k']['recall@3']:.2%}",
                f"{m['recall_at_k']['recall@5']:.2%}",
                f"{m['mrr']:.4f}",
                f"{m['ndcg_at_10']:.4f}",
                str(m["failed_at_1"]),
                f"{lat['query_encode_ms_per_query']:.1f}",
                f"{lat['throughput_qps']:.1f}",
            ]
        )
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    sep = " | "
    print(sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(headers) - 1)))
    for row in table:
        print(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))


def build_case_diff(
    left_failed: List[dict],
    right_failed: List[dict],
    test_cases: List[dict],
    left_name: str,
    right_name: str,
) -> List[dict]:
    left_map = {c["query"]: c for c in left_failed}
    right_map = {c["query"]: c for c in right_failed}
    rows = []
    for case in test_cases:
        q = case["query"]
        l = left_map.get(q)
        r = right_map.get(q)
        left_hit = l is None
        right_hit = r is None
        if left_hit and right_hit:
            winner = "both"
        elif left_hit:
            winner = left_name
        elif right_hit:
            winner = right_name
        else:
            winner = "neither"
        rows.append(
            {
                "query": q,
                "category": case.get("category", ""),
                "relevant_doc_id": case["relevant_doc_id"],
                f"{left_name}_hit": left_hit,
                f"{left_name}_rank": 1 if left_hit else l["rank"],
                f"{right_name}_hit": right_hit,
                f"{right_name}_rank": 1 if right_hit else r["rank"],
                "winner": winner,
            }
        )
    return rows


def run_dataset(
    root: Path,
    corpus: List[dict],
    dataset: dict,
    models_cache: Dict[str, SentenceTransformer],
    model_specs: List[dict],
) -> Tuple[List[dict], Optional[List[dict]]]:
    test_path = root / dataset["path"]
    if not test_path.exists():
        raise FileNotFoundError(f"缺少测试集: {test_path}")
    test_cases = json.loads(test_path.read_text(encoding="utf-8"))

    rows: List[dict] = []
    metrics_by_key: Dict[str, Any] = {}

    for spec in model_specs:
        print(f"\n>>> [{dataset['label']}] {spec['label']}")
        if spec["key"] not in models_cache:
            models_cache[spec["key"]] = load_model(spec["path"], root, spec["qwen3"])
        model = models_cache[spec["key"]]
        model_path = resolve_model_path(spec["path"], root)

        br = benchmark_model(
            model=model,
            label=spec["label"],
            model_path=model_path,
            corpus=corpus,
            test_cases=test_cases,
            query_encode_kwargs=spec["query_kwargs"],
            doc_encode_kwargs=spec["doc_kwargs"],
        )
        print(
            f"    R@1={br.retrieval.recall_at_k['recall@1']:.2%} "
            f"R@3={br.retrieval.recall_at_k['recall@3']:.2%} "
            f"R@5={br.retrieval.recall_at_k['recall@5']:.2%} "
            f"MRR={br.retrieval.mrr:.4f} "
            f"ms/q={br.latency.query_encode_ms_per_query:.1f}"
        )
        rows.append(
            {
                "dataset_key": dataset["key"],
                "dataset_label": dataset["label"],
                "dataset_note": dataset["note"],
                "label": spec["label"],
                "model_key": spec["key"],
                "model_note": spec["note"],
                "retrieval": asdict(br.retrieval),
                "latency": asdict(br.latency),
                "model": asdict(br.model),
            }
        )

        # 难例集补充 case-level 对比
        if dataset["key"] == "hard12":
            em = evaluate_retrieval(
                model=model,
                corpus_path=root / "data/corpus.json",
                test_cases_path=test_path,
                model_label=spec["label"],
                query_encode_kwargs=spec["query_kwargs"],
                doc_encode_kwargs=spec["doc_kwargs"],
            )
            print_metrics(em, title=f"{spec['label']} @ 难例集")
            metrics_by_key[spec["key"]] = em

    case_diff = None
    if dataset["key"] == "hard12" and len(metrics_by_key) == 2:
        case_diff = build_case_diff(
            metrics_by_key["bge_m3"].failed_cases,
            metrics_by_key["qwen3"].failed_cases,
            test_cases,
            "bge_m3",
            "qwen3",
        )
    return rows, case_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="BGE-M3 vs Qwen3-Embedding-0.6B")
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["hard12", "unseen500", "bench1k"],
        help="hard12 unseen500 bench1k",
    )
    parser.add_argument("--output", default="results/comparison_bge_m3_vs_qwen3.json")
    args = parser.parse_args()

    ensure_hf_mirror()
    root = Path(__file__).resolve().parent.parent
    corpus = json.loads((root / args.corpus).read_text(encoding="utf-8"))
    model_specs = build_model_specs(root)
    print(f"BGE-M3 path: {model_specs[0]['path']}")

    selected = [d for d in DATASETS if d["key"] in set(args.datasets)]
    if not selected:
        raise ValueError(f"未选中任何数据集: {args.datasets}")

    models_cache: Dict[str, SentenceTransformer] = {}
    all_rows: List[dict] = []
    hard12_diff: Optional[List[dict]] = None

    for dataset in selected:
        rows, case_diff = run_dataset(root, corpus, dataset, models_cache, model_specs)
        all_rows.extend(rows)
        if case_diff is not None:
            hard12_diff = case_diff

    # 按数据集汇总胜负
    summary = []
    for dataset in selected:
        subset = [r for r in all_rows if r["dataset_key"] == dataset["key"]]
        if len(subset) != 2:
            continue
        a, b = subset[0], subset[1]
        ra = a["retrieval"]["recall_at_k"]["recall@1"]
        rb = b["retrieval"]["recall_at_k"]["recall@1"]
        if abs(ra - rb) < 1e-9:
            winner = "tie"
        else:
            winner = a["label"] if ra > rb else b["label"]
        summary.append(
            {
                "dataset": dataset["label"],
                "dataset_key": dataset["key"],
                "winner_by_recall@1": winner,
                a["label"]: {
                    "recall@1": ra,
                    "recall@3": a["retrieval"]["recall_at_k"]["recall@3"],
                    "recall@5": a["retrieval"]["recall_at_k"]["recall@5"],
                    "mrr": a["retrieval"]["mrr"],
                    "query_ms": a["latency"]["query_encode_ms_per_query"],
                },
                b["label"]: {
                    "recall@1": rb,
                    "recall@3": b["retrieval"]["recall_at_k"]["recall@3"],
                    "recall@5": b["retrieval"]["recall_at_k"]["recall@5"],
                    "mrr": b["retrieval"]["mrr"],
                    "query_ms": b["latency"]["query_encode_ms_per_query"],
                },
            }
        )

    report = {
        "description": "BGE-M3 vs Qwen3-Embedding-0.6B 零样本对比（当前私有领域语料）",
        "protocol": {
            "similarity": "cosine (L2 normalize)",
            "batch_size": 32,
            "device": "cpu",
            "bge_m3": "dense embedding only via sentence-transformers",
            "qwen3": "query prompt_name=query",
        },
        "models": [
            {"key": s["key"], "label": s["label"], "path": s["path"], "note": s["note"]}
            for s in model_specs
        ],
        "datasets": [
            {"key": d["key"], "label": d["label"], "path": d["path"], "note": d["note"]}
            for d in selected
        ],
        "summary": summary,
        "results": all_rows,
        "hard12_case_diff": hard12_diff,
    }

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "model",
                "parameters_M",
                "embedding_dim",
                "recall@1",
                "recall@3",
                "recall@5",
                "recall@10",
                "mrr",
                "ndcg@10",
                "failed@1",
                "query_ms",
                "qps",
            ]
        )
        for r in all_rows:
            m, lat, info = r["retrieval"], r["latency"], r["model"]
            writer.writerow(
                [
                    r["dataset_label"],
                    r["label"],
                    f"{info['parameters'] / 1e6:.2f}",
                    info["embedding_dim"],
                    f"{m['recall_at_k']['recall@1']:.4f}",
                    f"{m['recall_at_k']['recall@3']:.4f}",
                    f"{m['recall_at_k']['recall@5']:.4f}",
                    f"{m['recall_at_k']['recall@10']:.4f}",
                    f"{m['mrr']:.4f}",
                    f"{m['ndcg_at_10']:.4f}",
                    m["failed_at_1"],
                    f"{lat['query_encode_ms_per_query']:.4f}",
                    f"{lat['throughput_qps']:.2f}",
                ]
            )

    print("\n" + "=" * 100)
    print("BGE-M3 vs Qwen3-Embedding-0.6B")
    print("=" * 100)
    print_table(all_rows)
    print("\n按数据集胜负（Recall@1）:")
    for s in summary:
        print(f"  - {s['dataset']}: {s['winner_by_recall@1']}")
    print(f"\nJSON: {out}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
