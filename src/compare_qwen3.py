"""对比 Qwen3-Embedding-0.6B 与 Hard neg 微调 BGE 模型。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from evaluate import evaluate_retrieval, print_metrics, save_metrics

QWEN3_MODEL = "Qwen/Qwen3-Embedding-0.6B"
HARD_NEG_MODEL = "models/bge-small-zh-hard-neg"
BASELINE_BGE = "BAAI/bge-small-zh-v1.5"


def ensure_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_model(path_or_name: str, root: Path, qwen3: bool = False) -> SentenceTransformer:
    model_path = root / path_or_name
    resolved = str(model_path) if model_path.exists() else path_or_name
    if qwen3:
        return SentenceTransformer(
            resolved,
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"padding_side": "left"},  # ST 3.x 仍接受此参数
        )
    return SentenceTransformer(resolved)


def qwen3_encode_kwargs() -> Dict[str, dict]:
    return {
        "query": {"prompt_name": "query"},
        "doc": {},
    }


def build_case_table(
    qwen3_all: List[Dict],
    hard_all: List[Dict],
    test_cases: List[dict],
) -> List[Dict]:
    qwen3_failed = {c["query"]: c for c in qwen3_all}
    hard_failed = {c["query"]: c for c in hard_all}

    rows = []
    for case in test_cases:
        q = case["query"]
        q_fail = qwen3_failed.get(q)
        h_fail = hard_failed.get(q)
        rows.append(
            {
                "query": q,
                "category": case.get("category", ""),
                "relevant_doc_id": case["relevant_doc_id"],
                "qwen3_hit": q_fail is None,
                "qwen3_rank": q_fail["rank"] if q_fail else 1,
                "qwen3_predicted": q_fail["predicted_doc_id"] if q_fail else case["relevant_doc_id"],
                "hard_neg_hit": h_fail is None,
                "hard_neg_rank": h_fail["rank"] if h_fail else 1,
                "hard_neg_predicted": h_fail["predicted_doc_id"] if h_fail else case["relevant_doc_id"],
                "winner": (
                    "both"
                    if q_fail is None and h_fail is None
                    else "qwen3"
                    if q_fail is None
                    else "hard_neg"
                    if h_fail is None
                    else "neither"
                ),
            }
        )
    return rows


def summarize(metrics: Dict) -> Dict:
    return {
        "model": metrics["model_name"],
        "recall_at_1": metrics["recall_at_1"],
        "recall_at_3": metrics["recall_at_3"],
        "recall_at_5": metrics["recall_at_5"],
        "mrr": metrics["mrr"],
        "avg_relevant_score": metrics["avg_relevant_score"],
        "failed_count": len(metrics["failed_cases"]),
        "failed_queries": [c["query"] for c in metrics["failed_cases"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 vs Hard neg 对比")
    parser.add_argument("--qwen3-model", default=QWEN3_MODEL)
    parser.add_argument("--hard-neg-model", default=HARD_NEG_MODEL)
    parser.add_argument("--baseline-model", default=BASELINE_BGE)
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--test-cases", default="data/test_cases.json")
    parser.add_argument("--output", default="results/comparison_qwen3_vs_hard_neg.json")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    ensure_hf_mirror()
    root = Path(__file__).resolve().parent.parent
    corpus_path = root / args.corpus
    test_cases_path = root / args.test_cases
    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
    output_path = root / args.output

    qwen_kwargs = qwen3_encode_kwargs()

    print(f"加载 Qwen3: {args.qwen3_model}")
    qwen3_model = load_model(args.qwen3_model, root, qwen3=True)
    qwen3_metrics = evaluate_retrieval(
        model=qwen3_model,
        corpus_path=corpus_path,
        test_cases_path=test_cases_path,
        model_label=args.qwen3_model,
        query_encode_kwargs=qwen_kwargs["query"],
        doc_encode_kwargs=qwen_kwargs["doc"],
    )
    print_metrics(qwen3_metrics, title="Qwen3-Embedding-0.6B")
    save_metrics(qwen3_metrics, root / "results/qwen3_metrics.json")

    print(f"\n加载 Hard neg 微调模型: {args.hard_neg_model}")
    hard_neg_model = load_model(args.hard_neg_model, root)
    hard_neg_metrics = evaluate_retrieval(
        model=hard_neg_model,
        corpus_path=corpus_path,
        test_cases_path=test_cases_path,
        model_label=args.hard_neg_model,
    )
    print_metrics(hard_neg_metrics, title="BGE Hard neg 微调")
    save_metrics(hard_neg_metrics, root / "results/hard_neg_metrics.json")

    baseline_summary = None
    if not args.skip_baseline:
        print(f"\n加载基线 BGE: {args.baseline_model}")
        baseline_model = load_model(args.baseline_model, root)
        baseline_metrics = evaluate_retrieval(
            model=baseline_model,
            corpus_path=corpus_path,
            test_cases_path=test_cases_path,
            model_label=args.baseline_model,
        )
        print_metrics(baseline_metrics, title="基线 BGE-small (参考)")
        save_metrics(baseline_metrics, root / "results/baseline_metrics.json")
        baseline_summary = summarize(asdict(baseline_metrics))

    qwen_data = asdict(qwen3_metrics)
    hard_data = asdict(hard_neg_metrics)
    case_table = build_case_table(
        qwen_data["failed_cases"],
        hard_data["failed_cases"],
        test_cases,
    )

    qwen_failed = {c["query"] for c in qwen_data["failed_cases"]}
    hard_failed = {c["query"] for c in hard_data["failed_cases"]}

    report = {
        "baseline_bge_small": baseline_summary,
        "qwen3_embedding_0.6b": summarize(qwen_data),
        "bge_hard_neg_finetuned": summarize(hard_data),
        "head_to_head": {
            "qwen3_only_wins": sorted(hard_failed - qwen_failed),
            "hard_neg_only_wins": sorted(qwen_failed - hard_failed),
            "both_fail": sorted(qwen_failed & hard_failed),
            "both_hit": sum(1 for r in case_table if r["winner"] == "both"),
        },
        "delta_qwen3_minus_hard_neg": {
            "recall_at_1": qwen_data["recall_at_1"] - hard_data["recall_at_1"],
            "mrr": qwen_data["mrr"] - hard_data["mrr"],
        },
        "per_case": case_table,
        "notes": {
            "qwen3_query_encoding": "encode(..., prompt_name='query')",
            "qwen3_doc_encoding": "encode(...) without prompt",
            "hard_neg_model": "BAAI/bge-small-zh-v1.5 + domain finetune + hard neg finetune",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Qwen3-0.6B vs Hard neg 微调 BGE")
    print("=" * 60)
    if baseline_summary:
        print(
            f"基线 BGE-small:     Recall@1 {baseline_summary['recall_at_1']:.2%} | "
            f"MRR {baseline_summary['mrr']:.4f} | 失败 {baseline_summary['failed_count']}"
        )
    print(
        f"Qwen3-0.6B:         Recall@1 {qwen_data['recall_at_1']:.2%} | "
        f"MRR {qwen_data['mrr']:.4f} | 失败 {len(qwen_data['failed_cases'])}"
    )
    print(
        f"Hard neg BGE:       Recall@1 {hard_data['recall_at_1']:.2%} | "
        f"MRR {hard_data['mrr']:.4f} | 失败 {len(hard_data['failed_cases'])}"
    )
    print(
        f"\nQwen3 - Hard neg:   Recall@1 {report['delta_qwen3_minus_hard_neg']['recall_at_1']:+.2%} | "
        f"MRR {report['delta_qwen3_minus_hard_neg']['mrr']:+.4f}"
    )

    if report["head_to_head"]["qwen3_only_wins"]:
        print("\n仅 Qwen3 命中:")
        for q in report["head_to_head"]["qwen3_only_wins"]:
            print(f"  [Qwen3] {q}")
    if report["head_to_head"]["hard_neg_only_wins"]:
        print("\n仅 Hard neg 命中:")
        for q in report["head_to_head"]["hard_neg_only_wins"]:
            print(f"  [Hard neg] {q}")
    if report["head_to_head"]["both_fail"]:
        print("\n两者均失败:")
        for q in report["head_to_head"]["both_fail"]:
            print(f"  [FAIL] {q}")

    print(f"\n报告: {output_path}")


if __name__ == "__main__":
    main()
