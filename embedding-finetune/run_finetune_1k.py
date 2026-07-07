"""1000 条数据微调 + 全模型 benchmark 对比。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LEARN_LOOP_PYTHON = Path(r"D:\software\anaconda\envs\learn-loop\python.exe")


def ensure_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def pick_python() -> str:
    return str(LEARN_LOOP_PYTHON) if LEARN_LOOP_PYTHON.exists() else sys.executable


def run(py: str, script: str, cwd: Path) -> None:
    cmd = [py, str(cwd / "src" / script)]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def main() -> None:
    ensure_hf_mirror()
    root = Path(__file__).resolve().parent
    py = pick_python()
    print(f"Python: {py}")

    run(py, "build_train_1k.py", root)
    run(py, "finetune_1k.py", root)
    run(py, "benchmark.py", root)
    print("\n完成。对比报告: results/benchmark_1k_report.json")


if __name__ == "__main__":
    main()
