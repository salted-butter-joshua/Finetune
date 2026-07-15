"""一键运行：BGE-M3 vs Qwen3-Embedding-0.6B 零样本对比。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LEARN_LOOP_PYTHON = Path(r"D:\software\anaconda\envs\learn-loop\python.exe")


def ensure_hf_mirror() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def pick_python() -> str:
    if LEARN_LOOP_PYTHON.exists():
        return str(LEARN_LOOP_PYTHON)
    return sys.executable


def main() -> None:
    ensure_hf_mirror()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    root = Path(__file__).resolve().parent
    py = pick_python()
    script = root / "src" / "compare_bge_m3_qwen3.py"
    print(f"Python: {py}")
    print(f">>> {py} {script}")
    subprocess.check_call([py, "-u", str(script)], cwd=root)
    print("\n完成。查看 results/comparison_bge_m3_vs_qwen3.json")


if __name__ == "__main__":
    main()
