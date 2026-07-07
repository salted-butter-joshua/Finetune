"""使用失败用例对应的 query-doc 正样本对微调 embedding 模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def load_train_pairs(path: Path) -> List[InputExample]:
    with path.open(encoding="utf-8") as f:
        pairs = json.load(f)
    return [InputExample(texts=[item["anchor"], item["positive"]]) for item in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description="领域 embedding 微调")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-pairs", default="data/train_pairs.json")
    parser.add_argument("--output", default="models/bge-small-zh-domain")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_path = root / args.train_pairs
    output_path = root / args.output

    print(f"加载基座模型: {args.model}")
    model = SentenceTransformer(args.model)

    train_examples = load_train_pairs(train_path)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(10, int(len(train_dataloader) * args.epochs * 0.1))
    print(f"训练样本: {len(train_examples)}, epochs: {args.epochs}, warmup: {warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"微调模型已保存: {output_path}")


if __name__ == "__main__":
    main()
