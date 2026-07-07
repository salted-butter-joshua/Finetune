"""在独立 holdout 集（12条难例）上对比各模型，避免与 1000 条训练集重叠。"""
from __future__ import annotations
import json
import os
from dataclasses import asdict
from pathlib import Path
from sentence_transformers import SentenceTransformer
from evaluate import evaluate_retrieval, print_metrics

MODELS = [
    ("BGE-small (基线)", "BAAI/bge-small-zh-v1.5", False, None, None),
    ("BGE-small (25条领域微调)", "models/bge-small-zh-domain", False, None, None),
    ("BGE-small (41条 Hard Neg)", "models/bge-small-zh-hard-neg", False, None, None),
    ("BGE-small (1000条微调)", "models/bge-small-zh-1k", False, None, None),
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", True, {"prompt_name": "query"}, {}),
]

def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    root = Path(__file__).resolve().parent.parent
    rows = []
    for label, path, qwen, qkw, dkw in MODELS:
        p = root / path
        resolved = str(p) if p.exists() else path
        kw = {"model_kwargs": {"trust_remote_code": True}, "tokenizer_kwargs": {"padding_side": "left"}} if qwen else {}
        model = SentenceTransformer(resolved, **kw)
        m = evaluate_retrieval(model, root/"data/corpus.json", root/"data/test_cases.json", label,
                               query_encode_kwargs=qkw, doc_encode_kwargs=dkw)
        rows.append({"label": label, "recall_at_1": m.recall_at_1, "mrr": m.mrr,
                     "failed": len(m.failed_cases)})
        print_metrics(m, title=label)
        del model
    out = root / "results/holdout_12_report.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHoldout 报告: {out}")

if __name__ == "__main__":
    main()
