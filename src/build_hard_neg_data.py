"""从基线失败用例自动生成带 hard negative 的训练三元组。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def corpus_map(corpus: List[dict]) -> Dict[str, str]:
    return {item["id"]: item["text"] for item in corpus}


def collect_hard_neg_ids(case: dict) -> List[str]:
    """优先误召回 Top1，再补充 Top3 中的其他干扰文档。"""
    seen: Set[str] = set()
    ordered: List[str] = []

    def add(doc_id: str) -> None:
        if doc_id and doc_id != case["relevant_doc_id"] and doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)

    add(case.get("predicted_doc_id", ""))
    for doc_id in case.get("top3_docs", []):
        add(doc_id)
    return ordered


def build_triplets_from_failures(
    failed_cases: List[dict],
    docs: Dict[str, str],
    max_hard_negs_per_query: int = 2,
) -> List[dict]:
    triplets: List[dict] = []
    for case in failed_cases:
        query = case["query"]
        positive = docs[case["relevant_doc_id"]]
        hard_ids = collect_hard_neg_ids(case)[:max_hard_negs_per_query]

        for hard_id in hard_ids:
            triplets.append(
                {
                    "anchor": query,
                    "positive": positive,
                    "hard_negative": docs[hard_id],
                    "hard_negative_id": hard_id,
                    "relevant_doc_id": case["relevant_doc_id"],
                    "source": "baseline_failure",
                    "category": case.get("category", ""),
                }
            )
    return triplets


def augment_with_train_variants(
    triplets: List[dict],
    train_pairs: List[dict],
    failed_queries: Set[str],
) -> List[dict]:
    """为失败 query 的口语变体也挂上相同 hard negative。"""
    positive_to_templates: Dict[str, List[dict]] = {}
    for t in triplets:
        positive_to_templates.setdefault(t["positive"], []).append(t)

    extra: List[dict] = []
    for pair in train_pairs:
        if pair["anchor"] in failed_queries:
            continue
        for tmpl in positive_to_templates.get(pair["positive"], []):
            extra.append(
                {
                    **tmpl,
                    "anchor": pair["anchor"],
                    "source": "train_pair_variant",
                }
            )
    return triplets + extra


def merge_failed_cases(*case_lists: List[dict]) -> List[dict]:
    """合并多份失败用例，按 query 聚合 hard neg 来源。"""
    by_query: Dict[str, dict] = {}
    for cases in case_lists:
        for case in cases:
            q = case["query"]
            if q not in by_query:
                by_query[q] = dict(case)
                by_query[q]["top3_docs"] = list(case.get("top3_docs", []))
                continue
            existing = by_query[q]
            for doc_id in [case.get("predicted_doc_id")] + case.get("top3_docs", []):
                if doc_id and doc_id not in existing["top3_docs"]:
                    existing["top3_docs"].append(doc_id)
            if case.get("predicted_doc_id"):
                existing["predicted_doc_id"] = case["predicted_doc_id"]
    return list(by_query.values())


def merge_training_data(
    train_pairs: List[dict],
    triplets: List[dict],
) -> List[dict]:
    """合并普通正样本对与三元组，去重 (anchor, positive, hard_negative)。"""
    merged: List[dict] = []
    seen: Set[tuple] = set()

    for pair in train_pairs:
        key = (pair["anchor"], pair["positive"], "")
        if key not in seen:
            seen.add(key)
            merged.append({"anchor": pair["anchor"], "positive": pair["positive"]})

    for triplet in triplets:
        key = (triplet["anchor"], triplet["positive"], triplet["hard_negative"])
        if key not in seen:
            seen.add(key)
            merged.append(
                {
                    "anchor": triplet["anchor"],
                    "positive": triplet["positive"],
                    "hard_negative": triplet["hard_negative"],
                    "hard_negative_id": triplet.get("hard_negative_id"),
                    "source": triplet.get("source", ""),
                }
            )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 hard negative 训练数据")
    parser.add_argument("--failed-cases", default="results/baseline_failed_cases.json")
    parser.add_argument(
        "--extra-failed-cases",
        default="results/finetuned_metrics.json",
        help="可选：普通微调后的失败用例（从 metrics 文件的 failed_cases 字段读取）",
    )
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--train-pairs", default="data/train_pairs.json")
    parser.add_argument("--output", default="data/train_triplets.json")
    parser.add_argument("--max-hard-negs", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    failed_path = root / args.failed_cases
    corpus_path = root / args.corpus
    train_pairs_path = root / args.train_pairs
    output_path = root / args.output

    failed_cases = load_json(failed_path)
    extra_failed_path = root / args.extra_failed_cases
    if extra_failed_path.exists():
        extra_payload = load_json(extra_failed_path)
        extra_failed = (
            extra_payload.get("failed_cases", extra_payload)
            if isinstance(extra_payload, dict)
            else extra_payload
        )
        if isinstance(extra_failed, list) and extra_failed:
            failed_cases = merge_failed_cases(failed_cases, extra_failed)
            print(f"已合并普通微调失败用例: {len(extra_failed)} 条")
    docs = corpus_map(load_json(corpus_path))
    train_pairs = load_json(train_pairs_path)

    triplets = build_triplets_from_failures(
        failed_cases, docs, max_hard_negs_per_query=args.max_hard_negs
    )
    failed_queries = {c["query"] for c in failed_cases}
    triplets = augment_with_train_variants(triplets, train_pairs, failed_queries)
    merged = merge_training_data(train_pairs, triplets)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    triplet_count = sum(1 for x in merged if "hard_negative" in x)
    pair_count = len(merged) - triplet_count
    print(f"训练样本总数: {len(merged)} (正样本对: {pair_count}, 含 hard neg 三元组: {triplet_count})")
    print(f"已保存: {output_path}")

    print("\nHard negative 三元组摘要:")
    for item in merged:
        if "hard_negative" not in item:
            continue
        hid = item.get("hard_negative_id", "?")
        print(f"  [{item.get('source', '')}] {item['anchor'][:20]}... -> neg={hid}")


if __name__ == "__main__":
    main()
