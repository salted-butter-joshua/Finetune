"""使用 1000 条 query-doc 正样本对从基线模型重新微调。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def load_pairs(path: Path) -> List[InputExample]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [InputExample(texts=[r["anchor"], r["positive"]]) for r in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="1000 条数据 embedding 微调")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", default="data/train_1k.json")
    parser.add_argument("--output", default="models/bge-small-zh-1k")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_path = root / args.train_data
    output_path = root / args.output
    model_path = root / args.model if (root / args.model).exists() else args.model

    print(f"基座模型: {model_path}")
    model = SentenceTransformer(str(model_path))

    examples = load_pairs(train_path)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = max(50, int(len(loader) * args.epochs * 0.1))

    print(f"训练样本: {len(examples)}, epochs: {args.epochs}, batch: {args.batch_size}, warmup: {warmup}")

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"模型已保存: {output_path}")


if __name__ == "__main__":
    main()
