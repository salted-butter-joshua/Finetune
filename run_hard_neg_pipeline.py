"""Hard negative 流程：构建数据 -> 微调 -> 三阶段对比。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def run(script: str, extra: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / "src" / script)]
    if extra:
        cmd.extend(extra)
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=root)


def main() -> None:
    ensure_hf_mirror()
    root = Path(__file__).resolve().parent

    baseline_metrics = root / "results" / "baseline_metrics.json"
    standard_model = root / "models" / "bge-small-zh-domain"

    if not baseline_metrics.exists():
        print("未找到基线评估结果，先运行 baseline_test.py ...")
        run("baseline_test.py")

    if not standard_model.exists():
        print("未找到普通微调模型，先运行 finetune.py ...")
        run("finetune.py")
        run("compare_results.py")

    run("build_hard_neg_data.py")
    run("finetune_hard_neg.py")
    run("compare_hard_neg.py")
    print("\nHard negative pipeline 完成。查看 results/comparison_hard_neg.json")


if __name__ == "__main__":
    main()
