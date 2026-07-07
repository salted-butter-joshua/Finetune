"""对比：基线 vs 普通微调 vs Hard negative 微调。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

from evaluate import evaluate_retrieval, print_metrics, save_metrics

DEFAULT_BASELINE = "BAAI/bge-small-zh-v1.5"
DEFAULT_STANDARD = "models/bge-small-zh-domain"
DEFAULT_HARD_NEG = "models/bge-small-zh-hard-neg"


def load_metrics(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def compare_two(before: Dict, after: Dict, before_label: str, after_label: str) -> Dict:
    before_failed = {c["query"] for c in before.get("failed_cases", [])}
    after_failed = {c["query"] for c in after.get("failed_cases", [])}
    return {
        "before": before_label,
        "after": after_label,
        "recall_at_1_before": before["recall_at_1"],
        "recall_at_1_after": after["recall_at_1"],
        "mrr_before": before["mrr"],
        "mrr_after": after["mrr"],
        "delta_recall_at_1": after["recall_at_1"] - before["recall_at_1"],
        "delta_mrr": after["mrr"] - before["mrr"],
        "recovered": sorted(before_failed - after_failed),
        "still_failed": sorted(before_failed & after_failed),
        "new_failures": sorted(after_failed - before_failed),
    }


def per_case_rows(baseline: Dict, model_metrics: Dict) -> List[Dict]:
    baseline_failed = {c["query"]: c for c in baseline.get("failed_cases", [])}
    model_failed = {c["query"]: c for c in model_metrics.get("failed_cases", [])}
    rows = []
    for query, base in baseline_failed.items():
        cur = model_failed.get(query)
        rows.append(
            {
                "query": query,
                "category": base["category"],
                "baseline_rank": base["rank"],
                "baseline_predicted": base["predicted_doc_id"],
                "current_rank": cur["rank"] if cur else 1,
                "current_predicted": cur["predicted_doc_id"] if cur else base["relevant_doc_id"],
                "recovered": query not in model_failed,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard neg 微调效果对比")
    parser.add_argument("--baseline-metrics", default="results/baseline_metrics.json")
    parser.add_argument("--standard-metrics", default="results/finetuned_metrics.json")
    parser.add_argument("--hard-neg-model", default=DEFAULT_HARD_NEG)
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--test-cases", default="data/test_cases.json")
    parser.add_argument("--output", default="results/comparison_hard_neg.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    baseline = load_metrics(root / args.baseline_metrics)
    standard = load_metrics(root / args.standard_metrics)

    print(f"加载 Hard neg 模型: {args.hard_neg_model}")
    model = SentenceTransformer(str(root / args.hard_neg_model))
    hard_neg_metrics = evaluate_retrieval(
        model=model,
        corpus_path=root / args.corpus,
        test_cases_path=root / args.test_cases,
        model_label=args.hard_neg_model,
    )
    print_metrics(hard_neg_metrics, title="Hard negative 微调评估结果")

    hard_neg_path = root / "results/hard_neg_metrics.json"
    save_metrics(hard_neg_metrics, hard_neg_path)
    hard_neg_data = asdict(hard_neg_metrics)

    report = {
        "baseline": {
            "recall_at_1": baseline["recall_at_1"],
            "recall_at_3": baseline["recall_at_3"],
            "mrr": baseline["mrr"],
            "failed_count": len(baseline["failed_cases"]),
        },
        "standard_finetune": {
            "recall_at_1": standard["recall_at_1"],
            "recall_at_3": standard["recall_at_3"],
            "mrr": standard["mrr"],
            "failed_count": len(standard["failed_cases"]),
        },
        "hard_neg_finetune": {
            "recall_at_1": hard_neg_data["recall_at_1"],
            "recall_at_3": hard_neg_data["recall_at_3"],
            "mrr": hard_neg_data["mrr"],
            "failed_count": len(hard_neg_data["failed_cases"]),
        },
        "standard_vs_baseline": compare_two(baseline, standard, "baseline", "standard"),
        "hard_neg_vs_baseline": compare_two(baseline, hard_neg_data, "baseline", "hard_neg"),
        "hard_neg_vs_standard": compare_two(standard, hard_neg_data, "standard", "hard_neg"),
        "per_case_hard_neg_vs_baseline": per_case_rows(baseline, hard_neg_data),
    }

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("三阶段对比 (Recall@1 / MRR)")
    print("=" * 60)
    print(f"基线:           {baseline['recall_at_1']:.2%} / {baseline['mrr']:.4f}")
    print(f"普通微调:       {standard['recall_at_1']:.2%} / {standard['mrr']:.4f}")
    print(f"Hard neg 微调:  {hard_neg_data['recall_at_1']:.2%} / {hard_neg_data['mrr']:.4f}")

    extra = report["hard_neg_vs_standard"]
    print(f"\n相对普通微调: Recall@1 {extra['delta_recall_at_1']:+.2%}, MRR {extra['delta_mrr']:+.4f}")
    if extra["recovered"]:
        print("Hard neg 额外修复:")
        for q in extra["recovered"]:
            print(f"  [OK] {q}")
    if extra["still_failed"]:
        print("仍未修复:")
        for q in extra["still_failed"]:
            print(f"  [FAIL] {q}")
    print(f"\n报告: {output_path}")


if __name__ == "__main__":
    main()
