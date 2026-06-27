#!/usr/bin/env python3
"""Intel oneAPI 快速入门：检测 oneMKL/TBB 并跑数值基准（对齐 oneAPI PDF）。"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.oneapi_runtime import apply_oneapi_env, detect_oneapi_stack, print_oneapi_summary, vtune_profile_hint


def _bench(label: str, fn, repeats: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = time.perf_counter() - start
    mean_ms = elapsed / repeats * 1000
    print(f"  {label}: {mean_ms:.2f} ms/次 ({repeats} 次)")
    return mean_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="oneAPI / oneMKL 环境检测与基准")
    parser.add_argument("--threads", type=int, default=None, help="MKL/OpenMP 线程数")
    parser.add_argument("--size", type=int, default=2048, help="矩阵基准维度")
    parser.add_argument("--vtune-hint", action="store_true", help="打印 VTune  profiling 命令提示")
    args = parser.parse_args()

    if args.vtune_hint:
        print(vtune_profile_hint())
        print()

    stack = apply_oneapi_env(num_threads=args.threads)
    if not stack["oneapi_ready"]:
        print("提示: 安装 oneMKL 后可获得更好 NumPy 线性代数性能")
        print(
            "  pip install -r requirements-oneapi.txt "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
        )

    n = args.size
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    audio = np.random.randn(32000 * 4).astype(np.float32)

    print(f"\n=== oneMKL 数值基准 (n={n}) ===")
    _bench("矩阵乘 @ (SGEMM)", lambda: a @ b)
    _bench("FFT (实数)", lambda: np.fft.rfft(audio))
    _bench("RMS (SSS 路径)", lambda: float(np.sqrt(np.mean(audio ** 2))))

    print_oneapi_summary(detect_oneapi_stack())


if __name__ == "__main__":
    main()
