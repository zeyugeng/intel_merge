#!/usr/bin/env python3
"""对比启用/未启用 oneAPI 环境时 SSS 音频处理与 NumPy 运算耗时。"""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.oneapi_runtime import apply_oneapi_env, detect_oneapi_stack
from core.sss_reader import normalize_for_birdnet, read_growing_pcm_tail


def _bench(label: str, fn, repeats: int = 30, warmup: int = 5) -> list[float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    mean_ms = statistics.mean(times) * 1000
    print(f"  {label}: mean={mean_ms:.2f} ms")
    return times


def _sss_pipeline_bench(raw_path: Path, repeats: int) -> None:
    if not raw_path.is_file():
        print(f"跳过 SSS 基准: 无文件 {raw_path}")
        return

    def run_once():
        audio = read_growing_pcm_tail(
            raw_path,
            sample_rate=32000,
            hop_size=512,
            n_channels=4,
            duration_sec=3.0,
        )
        if audio is not None:
            normalize_for_birdnet(audio)

    _bench("SSS 读取+归一化", run_once, repeats=repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description="oneAPI SSS/NumPy benchmark")
    parser.add_argument(
        "--sss-raw",
        type=Path,
        default=ROOT.parent / "odas" / "separated.raw",
        help="ODAS separated.raw 路径",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--skip-oneapi-env", action="store_true")
    args = parser.parse_args()

    print("=== 未设置 oneAPI 环境 ===")
    stack = detect_oneapi_stack()
    print(f"oneMKL: {stack.get('mkl') or '未安装'}")

    audio = np.random.randn(32000 * 3).astype(np.float32)
    matrix = np.random.randn(1024, 1024).astype(np.float32)
    _bench("FFT 96k samples", lambda: np.fft.rfft(audio), repeats=args.repeats)
    _bench("MatMul 1024", lambda: matrix @ matrix, repeats=args.repeats)
    _sss_pipeline_bench(args.sss_raw, args.repeats)

    if not args.skip_oneapi_env and stack["oneapi_ready"]:
        print("\n=== 启用 oneAPI 环境 (MKL 线程) ===")
        apply_oneapi_env(num_threads=args.threads, verbose=True)
        _bench("FFT 96k samples", lambda: np.fft.rfft(audio), repeats=args.repeats)
        _bench("MatMul 1024", lambda: matrix @ matrix, repeats=args.repeats)
        _sss_pipeline_bench(args.sss_raw, args.repeats)
    elif not stack["oneapi_ready"]:
        print("\n安装 oneMKL 后可对比加速效果:")
        print(
            "  pip install -r requirements-oneapi.txt "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
        )


if __name__ == "__main__":
    main()
