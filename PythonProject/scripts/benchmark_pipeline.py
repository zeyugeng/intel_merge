#!/usr/bin/env python3
"""
端到端性能测试：YOLO / BirdNET / SSS + CPU、内存、RAPL 功耗。

示例:
  python scripts/benchmark_pipeline.py --all
  python scripts/benchmark_pipeline.py --vision openvino --oneapi --save
  sudo python scripts/benchmark_pipeline.py --all   # 可读 RAPL 功耗
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import VisualConfig
from core.openvino_runtime import resolve_yolo_openvino_dir
from core.paths import OUTPUT_DIR, YOLO_MODEL_PATH
from core.performance_monitor import PerformanceMonitor, print_report, rapl_available
from core.sss_reader import normalize_for_birdnet, read_growing_pcm_tail


def _find_birdnet_clip() -> Path | None:
    clips = sorted((OUTPUT_DIR / "birdnet_clips").glob("sss_*.wav"), key=lambda p: p.stat().st_mtime)
    return clips[-1] if clips else None


def _find_sss_raw() -> Path | None:
    for p in (ROOT.parent / "odas" / "separated.raw", ROOT.parent / "odas" / "postfiltered.raw"):
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def bench_yolo(backend: str, frame: np.ndarray, repeats: int, warmup: int, ov_device: str):
    from core.visual_detector import VisualDetector

    cfg = VisualConfig(model_path=str(YOLO_MODEL_PATH), backend=backend, ov_device=ov_device)
    if backend == "openvino":
        resolve_yolo_openvino_dir(YOLO_MODEL_PATH, auto_export=True)
    det = VisualDetector(cfg)
    h, w = frame.shape[:2]

    def once():
        det.detect(frame, w, h)

    monitor = PerformanceMonitor(interval=0.25)
    for _ in range(warmup):
        once()
    monitor.start()
    latencies = []
    start = time.perf_counter()
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        latencies.append(time.perf_counter() - t0)
    duration = time.perf_counter() - start
    samples = monitor.stop()
    return PerformanceMonitor.summarize(
        f"YOLO ({backend})",
        samples,
        duration_s=duration,
        iterations=repeats,
        latencies=latencies,
    )


def bench_birdnet(wav: Path, repeats: int, warmup: int):
    from core.birdnet_infer import predict_audio

    def once():
        predict_audio(wav, confidence_threshold=0.15, top_k=3)

    monitor = PerformanceMonitor(interval=0.25)
    for _ in range(warmup):
        once()
    monitor.start()
    latencies = []
    start = time.perf_counter()
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        latencies.append(time.perf_counter() - t0)
    duration = time.perf_counter() - start
    samples = monitor.stop()
    return PerformanceMonitor.summarize(
        "BirdNET (TensorFlow CPU)",
        samples,
        duration_s=duration,
        iterations=repeats,
        latencies=latencies,
        notes=str(wav.name),
    )


def bench_sss(raw: Path, repeats: int, warmup: int):
    def once():
        result = read_growing_pcm_tail(raw, 32000, 512, 4, 3.0)
        if result is not None:
            audio, _ch = result
            normalize_for_birdnet(audio)

    monitor = PerformanceMonitor(interval=0.25)
    for _ in range(warmup):
        once()
    monitor.start()
    latencies = []
    start = time.perf_counter()
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        latencies.append(time.perf_counter() - t0)
    duration = time.perf_counter() - start
    samples = monitor.stop()
    return PerformanceMonitor.summarize(
        "SSS 读取+归一化",
        samples,
        duration_s=duration,
        iterations=repeats,
        latencies=latencies,
        notes=str(raw.name),
    )


def bench_fusion_frame_loop(frame: np.ndarray, backend: str, repeats: int, warmup: int, ov_device: str):
    from core.visual_detector import VisualDetector

    cfg = VisualConfig(model_path=str(YOLO_MODEL_PATH), backend=backend, ov_device=ov_device, conf=0.3)
    if backend == "openvino":
        resolve_yolo_openvino_dir(YOLO_MODEL_PATH, auto_export=True)
    det = VisualDetector(cfg)
    h, w = frame.shape[:2]
    overlay = "sound x=+0.12 y=-0.05 z=+0.98 E=0.85"

    def once():
        fc = frame.copy()
        dets = det.detect(frame, w, h)
        det.draw(fc, dets, highlight_idx=0 if dets else -1)
        cv2.putText(fc, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        _ = cv2.resize(fc, (1280, int(h * 1280 / w)))

    monitor = PerformanceMonitor(interval=0.25)
    for _ in range(warmup):
        once()
    monitor.start()
    latencies = []
    start = time.perf_counter()
    for _ in range(repeats):
        t0 = time.perf_counter()
        once()
        latencies.append(time.perf_counter() - t0)
    duration = time.perf_counter() - start
    samples = monitor.stop()
    fps = repeats / duration if duration > 0 else 0
    return PerformanceMonitor.summarize(
        f"Fusion 帧循环模拟 ({backend})",
        samples,
        duration_s=duration,
        iterations=repeats,
        latencies=latencies,
        notes=f"约 {fps:.1f} fps",
    )


def _print_tune_hints(reports) -> None:
    yolo_pt = next((r for r in reports if r.label == "YOLO (pytorch)"), None)
    yolo_ov = next((r for r in reports if r.label == "YOLO (openvino)"), None)
    if yolo_pt and yolo_ov and yolo_pt.latency_ms_mean and yolo_ov.latency_ms_mean:
        if yolo_ov.latency_ms_mean < yolo_pt.latency_ms_mean:
            speedup = yolo_pt.latency_ms_mean / yolo_ov.latency_ms_mean
            print(f"  • 视觉: OpenVINO 比 PyTorch 约快 {speedup:.1f}x → 推荐 --vision-backend openvino")
        else:
            ratio = yolo_ov.latency_ms_mean / yolo_pt.latency_ms_mean
            print(f"  • 视觉: 本次 PyTorch 更快 ({ratio:.1f}x)，可对比多次或加大 --repeats")

    fusion_ov = next((r for r in reports if "Fusion" in r.label and "openvino" in r.label), None)
    if fusion_ov and fusion_ov.latency_ms_mean:
        fps = 1000.0 / fusion_ov.latency_ms_mean
        print(f"  • Fusion 帧率约 {fps:.1f} fps；若卡顿可降 --conf 或减小预览分辨率")
        if fusion_ov.cpu_peak > 80:
            print("  • CPU 峰值较高 → 可加 --oneapi；BirdNET 冷却保持 --birdnet-cooldown 6")

    print("  • 云台太灵敏: --activity 0.05 --trigger-interval 3.0")
    print("  • BirdNET 误报多: --birdnet-conf 0.25")
    print("  • 调参预设: python scripts/tune_params.py --list")
    print("  • 实地: python scripts/run_sound_ptz_all.py --vision-backend openvino --oneapi")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline CPU/内存/功耗 benchmark")
    parser.add_argument("--all", action="store_true", help="跑全部子项")
    parser.add_argument("--vision", choices=("pytorch", "openvino", "both"), default="both")
    parser.add_argument("--oneapi", action="store_true")
    parser.add_argument("--ov-device", default="CPU")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--save", action="store_true", help="保存 JSON 到 output/performance/")
    parser.add_argument("--birdnet-wav", type=Path, default=None)
    parser.add_argument("--sss-raw", type=Path, default=None)
    parser.add_argument(
        "--rapl-sudo",
        action="store_true",
        help="通过 sudo cat 读 RAPL 功耗（用普通用户运行，会提示输入 sudo 密码）",
    )
    args = parser.parse_args()

    if os.geteuid() == 0:
        print("错误: 不要用 sudo 运行本脚本，OpenVINO 在 root 下会 import 失败。")
        print("请改为: python scripts/benchmark_pipeline.py --all --rapl-sudo")
        raise SystemExit(1)

    if not args.all and args.vision == "both":
        args.all = True

    if args.oneapi:
        from core.oneapi_runtime import apply_oneapi_env

        apply_oneapi_env(verbose=True)

    if args.rapl_sudo:
        from core.performance_monitor import enable_rapl_sudo

        if enable_rapl_sudo():
            print("  RAPL: 已通过 sudo cat 启用")
        else:
            print("  RAPL: sudo cat 仍失败，请检查 sudo 权限")

    import psutil

    print("=== 系统信息 ===")
    print(f"  CPU 逻辑核: {psutil.cpu_count(logical=True)}")
    print(f"  内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  RAPL 功耗可读: {rapl_available()}")
    if not rapl_available():
        print("  提示: 加 --rapl-sudo 可读 CPU package 功耗（勿 sudo 整个脚本）")

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    reports = []

    backends: list[str] = []
    if args.all or args.vision in ("pytorch", "both"):
        backends.append("pytorch")
    if args.all or args.vision in ("openvino", "both"):
        backends.append("openvino")

    for backend in backends:
        reports.append(bench_yolo(backend, frame, args.repeats, args.warmup, args.ov_device))
        if args.all:
            reports.append(
                bench_fusion_frame_loop(frame, backend, min(args.repeats, 15), args.warmup, args.ov_device)
            )

    if args.all:
        wav = args.birdnet_wav or _find_birdnet_clip()
        if wav and wav.is_file():
            reports.append(bench_birdnet(wav, min(args.repeats, 5), 1))
        else:
            print("\n跳过 BirdNET: 未找到 output/birdnet_clips/sss_*.wav")

        raw = args.sss_raw or _find_sss_raw()
        if raw and raw.is_file():
            reports.append(bench_sss(raw, args.repeats, args.warmup))
        else:
            print("\n跳过 SSS: 未找到 odas/separated.raw")

    for r in reports:
        print_report(r)

    print("\n=== Tune 建议（基于 benchmark）===")
    _print_tune_hints(reports)

    if args.save:
        out_dir = OUTPUT_DIR / "performance"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"benchmark_{stamp}.json"
        payload = {
            "timestamp": stamp,
            "oneapi": args.oneapi,
            "rapl_available": rapl_available(),
            "reports": [r.to_dict() for r in reports],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
