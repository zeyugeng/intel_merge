#!/usr/bin/env python3
"""云台回正：水平/俯仰回到 0°。"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import PTZConfig
from core.ptz_camera import PTZCameraController
from core.serial_ptz import SerialPTZConfig, SerialPanTiltBackend


def main() -> int:
    parser = argparse.ArgumentParser(description="云台回正（pan=0°, tilt=0°）")
    parser.add_argument(
        "--ptz-backend",
        choices=("serial", "onvif"),
        default="serial",
        help="serial=串口舵机（默认）; onvif=网络云台",
    )
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="串口设备路径")
    parser.add_argument("--serial-baud", type=int, default=115200, help="串口波特率")
    parser.add_argument("--move-time-ms", type=int, default=1000, help="转动耗时（毫秒）")
    parser.add_argument("--ip", default=PTZConfig.ip, help="ONVIF 云台 IP")
    parser.add_argument("--port", type=int, default=PTZConfig.port, help="ONVIF 端口")
    parser.add_argument("--user", default=PTZConfig.user, help="ONVIF 用户名")
    parser.add_argument("--password", default=PTZConfig.password, help="ONVIF 密码")
    args = parser.parse_args()

    if args.ptz_backend == "serial":
        ptz = SerialPanTiltBackend(
            SerialPTZConfig(
                port=args.serial_port,
                baud=args.serial_baud,
                default_time_ms=args.move_time_ms,
            )
        )
    else:
        ptz = PTZCameraController(
            PTZConfig(ip=args.ip, port=args.port, user=args.user, password=args.password)
        )

    if not ptz.connect():
        return 1

    try:
        before = ptz.get_current_angle()
        print(f"回正前角度: pan={before[0]:.2f}°, tilt={before[1]:.2f}°")
        ptz.center(args.move_time_ms)
        wait_s = max(0.2, args.move_time_ms / 1000.0 + 0.2)
        time.sleep(wait_s)
        after = ptz.get_current_angle()
        print(f"回正完成: pan={after[0]:.2f}°, tilt={after[1]:.2f}°")
    finally:
        ptz.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
