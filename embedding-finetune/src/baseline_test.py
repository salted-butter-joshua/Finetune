"""基线模型测试：评估开源 embedding 并收集失败用例。"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from evaluate import evaluate_retrieval, print_metrics, save_metrics

# BGE 中文小模型：MTEB 中文榜单表现好，体积小，适合 CPU 演示
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def main() -> None:
    parser = argparse.ArgumentParser(description="基线 embedding 检索评估")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--test-cases", default="data/test_cases.json")
    parser.add_argument("--output", default="results/baseline_metrics.json")
    parser.add_argument("--failed-output", default="results/baseline_failed_cases.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    corpus_path = root / args.corpus
    test_cases_path = root / args.test_cases
    output_path = root / args.output
    failed_output_path = root / args.failed_output

    print(f"加载基线模型: {args.model}")
    model = SentenceTransformer(args.model)

    metrics = evaluate_retrieval(
        model=model,
        corpus_path=corpus_path,
        test_cases_path=test_cases_path,
        model_label=args.model,
    )
    print_metrics(metrics, title="基线模型评估结果")

    save_metrics(metrics, output_path)

    failed_output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with failed_output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics.failed_cases, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {output_path}")
    print(f"失败用例已保存: {failed_output_path}")


if __name__ == "__main__":
    main()
