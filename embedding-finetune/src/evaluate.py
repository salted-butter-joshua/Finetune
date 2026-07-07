"""Embedding 检索评估工具。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer, util


@dataclass
class CaseResult:
    query: str
    relevant_doc_id: str
    predicted_doc_id: str
    rank: int
    top1_score: float
    relevant_score: float
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    category: str
    description: str
    top3_docs: List[str]


@dataclass
class EvalMetrics:
    model_name: str
    num_cases: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    avg_relevant_score: float
    avg_top1_score: float
    failed_cases: List[Dict]


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_corpus_index(corpus: Sequence[dict]) -> Tuple[List[str], List[str], Dict[str, int]]:
    doc_ids = [item["id"] for item in corpus]
    doc_texts = [item["text"] for item in corpus]
    id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    return doc_ids, doc_texts, id_to_idx


def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    encode_kwargs: Optional[dict] = None,
) -> np.ndarray:
    kwargs = {
        "batch_size": 32,
        "show_progress_bar": False,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }
    if encode_kwargs:
        kwargs.update(encode_kwargs)
    return model.encode(list(texts), **kwargs)


def evaluate_retrieval(
    model: SentenceTransformer,
    corpus_path: Path,
    test_cases_path: Path,
    model_label: Optional[str] = None,
    top_k: int = 5,
    query_encode_kwargs: Optional[dict] = None,
    doc_encode_kwargs: Optional[dict] = None,
) -> EvalMetrics:
    corpus = load_json(corpus_path)
    test_cases = load_json(test_cases_path)
    doc_ids, doc_texts, id_to_idx = build_corpus_index(corpus)

    doc_embeddings = encode_texts(model, doc_texts, doc_encode_kwargs)
    queries = [case["query"] for case in test_cases]
    query_embeddings = encode_texts(model, queries, query_encode_kwargs)

    case_results: List[CaseResult] = []
    reciprocal_ranks: List[float] = []

    for idx, case in enumerate(test_cases):
        scores = util.cos_sim(query_embeddings[idx], doc_embeddings)[0].cpu().numpy()
        full_ranked_indices = np.argsort(scores)[::-1]
        ranked_indices = full_ranked_indices[:top_k]

        relevant_idx = id_to_idx[case["relevant_doc_id"]]
        rank_matches = np.where(full_ranked_indices == relevant_idx)[0]
        rank = int(rank_matches[0]) + 1 if len(rank_matches) else len(doc_ids) + 1
        reciprocal_ranks.append(1.0 / rank)

        top3_docs = [doc_ids[i] for i in ranked_indices[:3]]
        case_results.append(
            CaseResult(
                query=case["query"],
                relevant_doc_id=case["relevant_doc_id"],
                predicted_doc_id=doc_ids[ranked_indices[0]],
                rank=rank,
                top1_score=float(scores[ranked_indices[0]]),
                relevant_score=float(scores[relevant_idx]),
                hit_at_1=rank == 1,
                hit_at_3=rank <= 3,
                hit_at_5=rank <= 5,
                category=case.get("category", "unknown"),
                description=case.get("description", ""),
                top3_docs=top3_docs,
            )
        )

    failed = [r for r in case_results if not r.hit_at_1]
    metrics = EvalMetrics(
        model_name=model_label or getattr(model, "model_card_data", None) and model.model_card_data.model_name or "unknown",
        num_cases=len(case_results),
        recall_at_1=sum(r.hit_at_1 for r in case_results) / len(case_results),
        recall_at_3=sum(r.hit_at_3 for r in case_results) / len(case_results),
        recall_at_5=sum(r.hit_at_5 for r in case_results) / len(case_results),
        mrr=float(np.mean(reciprocal_ranks)),
        avg_relevant_score=float(np.mean([r.relevant_score for r in case_results])),
        avg_top1_score=float(np.mean([r.top1_score for r in case_results])),
        failed_cases=[asdict(r) for r in failed],
    )
    return metrics


def save_metrics(metrics: EvalMetrics, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, ensure_ascii=False, indent=2)


def print_metrics(metrics: EvalMetrics, title: str = "") -> None:
    if title:
        print(f"\n{'=' * 60}")
        print(title)
        print("=" * 60)
    print(f"模型: {metrics.model_name}")
    print(f"用例数: {metrics.num_cases}")
    print(f"Recall@1: {metrics.recall_at_1:.2%}")
    print(f"Recall@3: {metrics.recall_at_3:.2%}")
    print(f"Recall@5: {metrics.recall_at_5:.2%}")
    print(f"MRR:      {metrics.mrr:.4f}")
    print(f"相关文档平均相似度: {metrics.avg_relevant_score:.4f}")
    print(f"Top1 平均相似度:     {metrics.avg_top1_score:.4f}")
    print(f"Recall@1 失败数: {len(metrics.failed_cases)}")

    if metrics.failed_cases:
        print("\n失败用例摘要:")
        for case in metrics.failed_cases:
            print(
                f"  [{case['category']}] {case['query']} -> "
                f"期望 {case['relevant_doc_id']}, 实际 {case['predicted_doc_id']} "
                f"(rank={case['rank']}, top3={case['top3_docs']})"
            )
