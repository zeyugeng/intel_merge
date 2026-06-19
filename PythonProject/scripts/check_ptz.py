#!/usr/bin/env python3
"""Check ONVIF PTZ connectivity without starting ODAS or moving the camera."""

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import PTZConfig
from core.ptz_camera import PTZCameraController


def tcp_reachable(ip: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError as exc:
        print(f"TCP 连接失败: {ip}:{port} ({exc})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="检测 ONVIF 云台连接")
    parser.add_argument("--ip", default=PTZConfig.ip, help="云台 IP")
    parser.add_argument("--port", type=int, default=PTZConfig.port, help="ONVIF 端口")
    parser.add_argument("--user", default=PTZConfig.user, help="用户名")
    parser.add_argument("--password", default=PTZConfig.password, help="密码")
    parser.add_argument("--timeout", type=float, default=3.0, help="TCP 连接超时秒数")
    args = parser.parse_args()

    config = PTZConfig(
        ip=args.ip,
        port=args.port,
        user=args.user,
        password=args.password,
    )

    print(f"检测云台: {config.ip}:{config.port}")
    if not tcp_reachable(config.ip, config.port, args.timeout):
        return 1
    print("TCP 端口可达")

    ptz = PTZCameraController(config)
    if not ptz.connect():
        return 2

    print(f"ONVIF 连接成功，ProfileToken={ptz.profile_token}")
    stream_uri = ptz.get_stream_uri()
    if stream_uri:
        print(f"RTSP 地址: {stream_uri}")
    else:
        print("未获取到 RTSP 地址")

    print("云台连接检测通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
