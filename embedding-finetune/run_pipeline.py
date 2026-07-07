"""一键运行：基线评估 -> 微调 -> 对比评估。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_hf_mirror() -> None:
    # 国内环境默认使用镜像，可通过 HF_ENDPOINT 覆盖
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def run_step(label: str, script: str, extra_args: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / "src" / script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n>>> Step: {label}")
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=root)


def main() -> None:
    ensure_hf_mirror()
    run_step("基线评估", "baseline_test.py")
    run_step("领域微调", "finetune.py")
    run_step("微调后对比", "compare_results.py")
    print("\nPipeline 完成。查看 results/ 目录获取指标与对比报告。")


if __name__ == "__main__":
    main()
