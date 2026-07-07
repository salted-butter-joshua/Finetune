"""从 benchmark_1k 构建 (query, positive) 训练对。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark_1k.json")
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--output", default="data/train_1k.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cases = json.loads((root / args.benchmark).read_text(encoding="utf-8"))
    corpus = {d["id"]: d["text"] for d in json.loads((root / args.corpus).read_text(encoding="utf-8"))}

    pairs = []
    for case in cases:
        doc_id = case["relevant_doc_id"]
        pairs.append(
            {
                "anchor": case["query"],
                "positive": corpus[doc_id],
                "relevant_doc_id": doc_id,
                "category": case.get("category", ""),
                "source_id": case.get("id", ""),
            }
        )

    out = root / args.output
    out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练对: {len(pairs)} 条 -> {out}")


if __name__ == "__main__":
    main()
