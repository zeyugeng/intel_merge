#!/usr/bin/env python3
"""OpenVINO 快速入门：列出设备并对 YOLO IR 做一次推理（对齐 Intel OpenVINO PDF quickstart）。"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.openvino_runtime import (
    compile_model,
    find_openvino_xml,
    print_device_summary,
    resolve_yolo_openvino_dir,
)
from core.paths import YOLO_MODEL_PATH, YOLO_OPENVINO_DIR


def _load_bgr_image(path: Path | None, imgsz: int) -> np.ndarray:
    if path and path.exists():
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"无法读取图像: {path}")
        return img
    return np.zeros((imgsz, imgsz, 3), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVINO YOLO 推理演示")
    parser.add_argument("--model-pt", type=Path, default=YOLO_MODEL_PATH)
    parser.add_argument("--model-ov", type=Path, default=YOLO_OPENVINO_DIR)
    parser.add_argument("--device", default="CPU", help="OpenVINO 设备: CPU, GPU, NPU, AUTO")
    parser.add_argument("--image", type=Path, default=None, help="测试图片；缺省用空白图")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--auto-export", action="store_true", help="无 IR 时从 .pt 自动导出")
    args = parser.parse_args()

    print_device_summary()

    ov_dir = args.model_ov
    if find_openvino_xml(ov_dir) is None:
        if args.auto_export:
            ov_dir = resolve_yolo_openvino_dir(args.model_pt, auto_export=True, imgsz=args.imgsz)
        else:
            raise SystemExit(
                f"未找到 {ov_dir}，请先运行: python scripts/export_yolo_openvino.py"
            )

    xml_path = find_openvino_xml(ov_dir)
    compiled = compile_model(xml_path, device=args.device)

    img = _load_bgr_image(args.image, args.imgsz)
    resized = cv2.resize(img, (args.imgsz, args.imgsz))
    blob = resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    start = time.perf_counter()
    outputs = compiled(blob)
    elapsed_ms = (time.perf_counter() - start) * 1000

    out_list = list(outputs.values()) if hasattr(outputs, "values") else [outputs]
    shapes = [tuple(o.shape) for o in out_list]
    print(f"模型: {xml_path}")
    print(f"设备: {args.device}")
    print(f"输出 shape: {shapes}")
    print(f"推理耗时: {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
