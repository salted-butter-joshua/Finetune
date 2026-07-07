"""对比基线模型与微调模型在原有失败用例上的表现。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

from evaluate import evaluate_retrieval, print_metrics, save_metrics

DEFAULT_BASELINE = "BAAI/bge-small-zh-v1.5"
DEFAULT_FINETUNED = "models/bge-small-zh-domain"


def compare_metrics(baseline: Dict, finetuned: Dict) -> Dict:
    delta = {
        "recall_at_1": finetuned["recall_at_1"] - baseline["recall_at_1"],
        "recall_at_3": finetuned["recall_at_3"] - baseline["recall_at_3"],
        "recall_at_5": finetuned["recall_at_5"] - baseline["recall_at_5"],
        "mrr": finetuned["mrr"] - baseline["mrr"],
        "avg_relevant_score": finetuned["avg_relevant_score"] - baseline["avg_relevant_score"],
    }

    baseline_failed_queries = {c["query"] for c in baseline["failed_cases"]}
    finetuned_failed_queries = {c["query"] for c in finetuned["failed_cases"]}
    recovered = sorted(baseline_failed_queries - finetuned_failed_queries)
    still_failed = sorted(baseline_failed_queries & finetuned_failed_queries)
    regressed = sorted(finetuned_failed_queries - baseline_failed_queries)

    return {
        "baseline_recall_at_1": baseline["recall_at_1"],
        "finetuned_recall_at_1": finetuned["recall_at_1"],
        "delta": delta,
        "baseline_failed_count": len(baseline["failed_cases"]),
        "finetuned_failed_count": len(finetuned["failed_cases"]),
        "recovered_from_baseline_failures": recovered,
        "still_failed_from_baseline": still_failed,
        "new_failures_after_finetune": regressed,
    }


def per_case_comparison(baseline: Dict, finetuned: Dict) -> List[Dict]:
    baseline_map = {c["query"]: c for c in baseline.get("failed_cases", [])}
    finetuned_all = {c["query"]: c for c in finetuned.get("failed_cases", [])}

    rows = []
    for query, base_case in baseline_map.items():
        fin_case = finetuned_all.get(query)
        rows.append(
            {
                "query": query,
                "category": base_case["category"],
                "baseline_rank": base_case["rank"],
                "baseline_predicted": base_case["predicted_doc_id"],
                "finetuned_rank": fin_case["rank"] if fin_case else 1,
                "finetuned_predicted": fin_case["predicted_doc_id"] if fin_case else base_case["relevant_doc_id"],
                "recovered": query not in finetuned_all,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="微调前后对比评估")
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE)
    parser.add_argument("--finetuned-model", default=DEFAULT_FINETUNED)
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--test-cases", default="data/test_cases.json")
    parser.add_argument("--baseline-metrics", default="results/baseline_metrics.json")
    parser.add_argument("--output", default="results/comparison.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    corpus_path = root / args.corpus
    test_cases_path = root / args.test_cases
    baseline_metrics_path = root / args.baseline_metrics
    output_path = root / args.output
    finetuned_metrics_path = root / "results/finetuned_metrics.json"

    with baseline_metrics_path.open(encoding="utf-8") as f:
        baseline_data = json.load(f)

    print(f"加载微调模型: {args.finetuned_model}")
    finetuned_model = SentenceTransformer(str(root / args.finetuned_model))
    finetuned_metrics = evaluate_retrieval(
        model=finetuned_model,
        corpus_path=corpus_path,
        test_cases_path=test_cases_path,
        model_label=args.finetuned_model,
    )
    print_metrics(finetuned_metrics, title="微调模型评估结果")
    save_metrics(finetuned_metrics, finetuned_metrics_path)

    from dataclasses import asdict

    finetuned_data = asdict(finetuned_metrics)

    comparison = compare_metrics(baseline_data, finetuned_data)
    comparison["per_case_on_baseline_failures"] = per_case_comparison(baseline_data, finetuned_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("微调前后对比")
    print("=" * 60)
    print(f"Recall@1: {comparison['baseline_recall_at_1']:.2%} -> {comparison['finetuned_recall_at_1']:.2%} "
          f"(Δ {comparison['delta']['recall_at_1']:+.2%})")
    print(f"MRR:      {baseline_data['mrr']:.4f} -> {finetuned_data['mrr']:.4f} "
          f"(Δ {comparison['delta']['mrr']:+.4f})")
    print(f"基线失败 -> 微调后仍失败: {len(comparison['still_failed_from_baseline'])}")
    print(f"基线失败 -> 微调后修复:   {len(comparison['recovered_from_baseline_failures'])}")
    if comparison["recovered_from_baseline_failures"]:
        print("\n已修复的基线失败用例:")
        for q in comparison["recovered_from_baseline_failures"]:
            print(f"  [OK] {q}")
    if comparison["still_failed_from_baseline"]:
        print("\n仍未修复的基线失败用例:")
        for q in comparison["still_failed_from_baseline"]:
            print(f"  [FAIL] {q}")
    print(f"\n对比报告: {output_path}")


if __name__ == "__main__":
    main()
