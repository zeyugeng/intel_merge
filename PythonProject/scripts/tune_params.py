#!/usr/bin/env python3
"""参数调优向导：根据场景给出 run_sound_ptz_all 推荐参数。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PRESETS = {
    "demo": {
        "desc": "比赛演示 / Intel 栈全开",
        "args": [
            "--vision-backend", "openvino",
            "--oneapi",
            "--activity", "0.01",
            "--trigger-interval", "2.0",
            "--birdnet-conf", "0.15",
        ],
    },
    "field": {
        "desc": "实地监测：减少云台乱转、降低误报",
        "args": [
            "--vision-backend", "openvino",
            "--oneapi",
            "--activity", "0.05",
            "--trigger-interval", "3.0",
            "--birdnet-conf", "0.25",
            "--birdnet-cooldown", "8.0",
        ],
    },
    "quiet": {
        "desc": "室内安静环境：更高触发阈值",
        "args": [
            "--vision-backend", "openvino",
            "--activity", "0.08",
            "--trigger-interval", "4.0",
            "--birdnet-conf", "0.30",
        ],
    },
    "perf": {
        "desc": "优先帧率：OpenVINO + oneAPI，略降 YOLO 置信度",
        "args": [
            "--vision-backend", "openvino",
            "--oneapi",
            "--conf", "0.25",
            "--activity", "0.05",
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="fusion 参数调优预设")
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS.keys()),
        default="demo",
        help="场景预设",
    )
    parser.add_argument("--list", action="store_true", help="列出所有预设")
    args = parser.parse_args()

    if args.list:
        for name, cfg in PRESETS.items():
            print(f"  {name}: {cfg['desc']}")
        return

    cfg = PRESETS[args.preset]
    cmd = ["python", "scripts/run_sound_ptz_all.py", *cfg["args"]]
    print(f"# {cfg['desc']} ({args.preset})")
    print(" ".join(cmd))
    print()
    print("调参说明:")
    print("  --activity          声源强度阈值，越大越不敏感")
    print("  --trigger-interval  云台两次转向最小间隔(秒)")
    print("  --birdnet-conf      BirdNET 置信度，越大误报越少")
    print("  --conf              YOLO 检测阈值，越大框越少、越快")
    print("  --oneapi-threads N  限制 MKL 线程，CPU 占用高时可设为 4")
    print()
    print("性能测试:")
    print("  python scripts/benchmark_pipeline.py --all --oneapi --rapl-sudo --save")
    print("  python scripts/monitor_system.py --duration 120 --rapl-sudo --save")


if __name__ == "__main__":
    main()
