#!/usr/bin/env python3
"""对比 PyTorch YOLO 与 OpenVINO YOLO 的推理延迟与 CPU 占用。"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import VisualConfig
from core.openvino_runtime import print_device_summary, resolve_yolo_openvino_dir
from core.paths import YOLO_MODEL_PATH
from core.visual_detector import VisualDetector


def _sample_frame(imgsz: int, image: Path | None) -> np.ndarray:
    if image and image.exists():
        frame = cv2.imread(str(image))
        if frame is None:
            raise SystemExit(f"无法读取图像: {image}")
        return frame
    return np.zeros((imgsz, imgsz, 3), dtype=np.uint8)


def _bench_detector(detector: VisualDetector, frame: np.ndarray, repeats: int, warmup: int) -> list[float]:
    h, w = frame.shape[:2]
    for _ in range(warmup):
        detector.detect(frame, w, h)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        detector.detect(frame, w, h)
        timings.append(time.perf_counter() - start)
    return timings


def _format_stats(label: str, timings: list[float]) -> str:
    mean_ms = statistics.mean(timings) * 1000
    p95_ms = statistics.quantiles(timings, n=20)[18] * 1000 if len(timings) >= 20 else max(timings) * 1000
    return f"{label}: mean={mean_ms:.1f} ms, p95≈{p95_ms:.1f} ms, fps≈{1.0 / statistics.mean(timings):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO PyTorch vs OpenVINO benchmark")
    parser.add_argument("--model", type=Path, default=YOLO_MODEL_PATH)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ov-device", default="CPU", help="OpenVINO 设备: CPU, GPU, NPU")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-openvino", action="store_true")
    args = parser.parse_args()

    print_device_summary()
    frame = _sample_frame(args.imgsz, args.image)
    h, w = frame.shape[:2]
    print(f"输入帧: {w}x{h}, repeats={args.repeats}")

    if not args.skip_pytorch:
        pt_det = VisualDetector(VisualConfig(model_path=str(args.model), backend="pytorch", device="cpu"))
        pt_times = _bench_detector(pt_det, frame, args.repeats, args.warmup)
        print(_format_stats("PyTorch CPU", pt_times))

    if not args.skip_openvino:
        resolve_yolo_openvino_dir(args.model, auto_export=True, imgsz=args.imgsz)
        ov_det = VisualDetector(
            VisualConfig(
                model_path=str(args.model),
                backend="openvino",
                ov_device=args.ov_device,
            )
        )
        ov_times = _bench_detector(ov_det, frame, args.repeats, args.warmup)
        print(_format_stats(f"OpenVINO {args.ov_device.upper()}", ov_times))

    try:
        import psutil

        proc = psutil.Process()
        print(f"进程 CPU% (瞬时): {proc.cpu_percent(interval=0.5):.1f}")
        print(f"进程内存: {proc.memory_info().rss / (1024 * 1024):.1f} MB")
    except ImportError:
        print("提示: pip install psutil 可显示 CPU/内存占用")


if __name__ == "__main__":
    main()
