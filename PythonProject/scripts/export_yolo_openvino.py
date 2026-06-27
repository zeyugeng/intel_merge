#!/usr/bin/env python3
"""将 YOLO .pt 导出为 OpenVINO IR（对齐 Ultralytics / OpenVINO 文档）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.openvino_runtime import export_yolo_to_openvino, find_openvino_xml, print_device_summary
from core.paths import YOLO_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 YOLO 为 OpenVINO IR")
    parser.add_argument("--model", type=Path, default=YOLO_MODEL_PATH, help="输入 .pt 权重")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 导出")
    parser.add_argument("--int8", action="store_true", help="INT8 量化（需校准数据）")
    parser.add_argument("--force", action="store_true", help="覆盖已有导出")
    parser.add_argument("--list-devices", action="store_true", help="列出 OpenVINO 可用设备")
    args = parser.parse_args()

    if args.list_devices:
        print_device_summary()
        return

    out_dir = export_yolo_to_openvino(
        args.model,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        force=args.force,
    )
    xml_path = find_openvino_xml(out_dir)
    print(f"导出完成: {xml_path}")
    print_device_summary()


if __name__ == "__main__":
    main()
