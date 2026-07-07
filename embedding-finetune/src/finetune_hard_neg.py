"""使用 (anchor, positive, hard_negative) 三元组微调 embedding。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_BASE_MODEL = "models/bge-small-zh-domain"


def load_training_data(path: Path) -> List[InputExample]:
    with path.open(encoding="utf-8") as f:
        records = json.load(f)

    examples: List[InputExample] = []
    for item in records:
        if item.get("hard_negative"):
            examples.append(
                InputExample(texts=[item["anchor"], item["positive"], item["hard_negative"]])
            )
        else:
            examples.append(InputExample(texts=[item["anchor"], item["positive"]]))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard negative embedding 微调")
    parser.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help="基座模型；默认在已有领域微调模型上继续训练",
    )
    parser.add_argument("--train-data", default="data/train_triplets.json")
    parser.add_argument("--output", default="models/bge-small-zh-hard-neg")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_path = root / args.train_data
    output_path = root / args.output
    model_path = root / args.model if (root / args.model).exists() else args.model

    print(f"加载基座模型: {model_path}")
    model = SentenceTransformer(str(model_path))

    train_examples = load_training_data(train_path)
    triplet_n = sum(1 for ex in train_examples if len(ex.texts) == 3)
    print(f"训练样本: {len(train_examples)} (其中三元组: {triplet_n})")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = max(10, int(len(train_dataloader) * args.epochs * 0.1))
    print(f"epochs: {args.epochs}, batch_size: {args.batch_size}, warmup: {warmup_steps}, lr: {args.lr}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"Hard neg 微调模型已保存: {output_path}")


if __name__ == "__main__":
    main()
