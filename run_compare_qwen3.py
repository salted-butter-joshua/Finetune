"""对比 Qwen3-Embedding-0.6B 与 Hard neg 微调 BGE（自动选择可用 Python）。"""

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
    root = Path(__file__).resolve().parent
    py = pick_python()
    script = root / "src" / "compare_qwen3.py"
    print(f"使用 Python: {py}")
    subprocess.check_call([py, str(script)], cwd=root)


if __name__ == "__main__":
    main()
