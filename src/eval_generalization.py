"""严格划分 train/test（query 不重叠），评估真实泛化能力。"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import List

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from evaluate import evaluate_retrieval, print_metrics

DEFAULT_BASE = "BAAI/bge-small-zh-v1.5"


def load_cases(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pairs(cases: List[dict], corpus: dict) -> List[dict]:
    return [
        {"anchor": c["query"], "positive": corpus[c["relevant_doc_id"]], "relevant_doc_id": c["relevant_doc_id"]}
        for c in cases
    ]


def finetune(train_pairs: List[dict], output: Path, epochs: int = 3, batch_size: int = 32) -> None:
    model = SentenceTransformer(DEFAULT_BASE)
    examples = [InputExample(texts=[p["anchor"], p["positive"]]) for p in train_pairs]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = max(20, int(len(loader) * epochs * 0.1))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": 2e-5},
        show_progress_bar=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark_1k.json")
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/generalization")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    root = Path(__file__).resolve().parent.parent
    cases = load_cases(root / args.benchmark)
    corpus = {d["id"]: d["text"] for d in load_cases(root / args.corpus)}

    rng = random.Random(args.seed)
    rng.shuffle(cases)
    n_test = int(len(cases) * args.test_ratio)
    test_cases = cases[:n_test]
    train_cases = cases[n_test:]

    train_queries = {c["query"] for c in train_cases}
    test_queries = {c["query"] for c in test_cases}
    assert not (train_queries & test_queries), "train/test query 存在重叠"

    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_800.json").write_text(json.dumps(train_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "test_200.json").write_text(json.dumps(test_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    train_pairs = build_pairs(train_cases, corpus)
    (out_dir / "train_pairs_800.json").write_text(
        json.dumps(train_pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 写入临时测试文件供 evaluate 使用
    test_path = out_dir / "test_200_eval.json"
    test_path.write_text(json.dumps(test_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus_path = root / args.corpus

    models = [
        ("基线 BGE-small", DEFAULT_BASE, None),
    ]

    finetuned_path = out_dir / "model_800"
    print(f"\n>>> 在 {len(train_cases)} 条未见测试集的 query 上微调...")
    finetune(train_pairs, finetuned_path, epochs=3)
    models.append(("800条划分微调", str(finetuned_path), None))

    # 对比：曾在全量1000上训练的模型（存在泄漏）
    full_1k = root / "models/bge-small-zh-1k"
    if full_1k.exists():
        models.append(("1000条全量微调(有泄漏)", str(full_1k), None))

    report = {"split": {"train": len(train_cases), "test": len(test_cases), "seed": args.seed}, "results": []}

    for label, path, _ in models:
        print(f"\n>>> 评测: {label}（仅在 {len(test_cases)} 条未见 query 上）")
        model = SentenceTransformer(path)
        metrics = evaluate_retrieval(model, corpus_path, test_path, model_label=label)
        print_metrics(metrics)
        report["results"].append(asdict(metrics))
        del model

    report_path = out_dir / "generalization_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n泛化评测报告: {report_path}")


if __name__ == "__main__":
    main()
