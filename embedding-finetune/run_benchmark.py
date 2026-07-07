"""生成 1000 条测试集并运行四模型标准化 Benchmark。"""

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


def run(py: str, script: str, cwd: Path) -> None:
    cmd = [py, str(cwd / "src" / script)]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def main() -> None:
    ensure_hf_mirror()
    root = Path(__file__).resolve().parent
    py = pick_python()
    print(f"Benchmark Python: {py}")

    run(py, "generate_benchmark_1k.py", root)
    run(py, "benchmark.py", root)
    print("\n完成。查看 results/benchmark_1k_report.json")


if __name__ == "__main__":
    main()
