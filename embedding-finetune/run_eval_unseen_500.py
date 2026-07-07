"""生成 500 条未见测试集 + 五模型泛化对比（不改动现有数据）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PY = Path(r"D:\software\anaconda\envs\learn-loop\python.exe")


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    root = Path(__file__).resolve().parent
    py = str(PY) if PY.exists() else sys.executable
    for script in ("generate_benchmark_500_unseen.py", "eval_unseen_500.py"):
        cmd = [py, str(root / "src" / script)]
        print(">>>", " ".join(cmd))
        subprocess.check_call(cmd, cwd=root)


if __name__ == "__main__":
    main()
