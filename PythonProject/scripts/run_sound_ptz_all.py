#!/usr/bin/env python3
"""一键启动：声源桥接 + ODAS + 云台跟踪（单终端）。"""

import argparse
import sys
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import ODASConfig, PTZConfig, PTZTrackConfig, SoundConfig
from core.ptz_camera import PTZCameraController
from core.ptz_tracker import MainThreadCameraPreview, SoundPTZTracker
from core.serial_ptz import SerialPTZConfig, SerialPanTiltBackend
from core.sound_pipeline import SoundPipeline


def build_configs(args):
    odas_config = ODASConfig(config_path=args.odas_cfg)

    sound_config = SoundConfig(
        host=odas_config.host,
        port=odas_config.python_port,
    )
    if args.energy is not None:
        sound_config.energy_threshold = args.energy
    if args.invert_x is not None:
        sound_config.invert_x = args.invert_x

    track_config = PTZTrackConfig(show_preview=not args.no_preview)
    if args.kp_pan is not None:
        track_config.kp_pan = args.kp_pan
    if args.kp_tilt is not None:
        track_config.kp_tilt = args.kp_tilt
    if args.invert_y is not None:
        track_config.invert_y = args.invert_y
    if args.deadzone is not None:
        track_config.deadzone = args.deadzone
    if args.max_speed is not None:
        track_config.max_speed = args.max_speed
    if args.control_interval is not None:
        track_config.control_interval = args.control_interval

    return odas_config, sound_config, track_config


def main() -> None:
    parser = argparse.ArgumentParser(description="单终端：ODAS + 桥接 + 云台声源跟踪")
    parser.add_argument("--no-preview", action="store_true", help="不打开 RTSP 预览")
    parser.add_argument("--energy", type=float, default=None, help="声源能量阈值")
    parser.add_argument("--kp-pan", type=float, default=None, help="水平增益")
    parser.add_argument("--kp-tilt", type=float, default=None, help="俯仰增益")
    parser.add_argument(
        "--invert-x",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否反转声源 x 到云台水平转向，默认使用配置值",
    )
    parser.add_argument(
        "--invert-y",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否反转声源 y 到云台俯仰转向，默认使用配置值",
    )
    parser.add_argument("--deadzone", type=float, default=None, help="声源坐标死区，默认 0.08")
    parser.add_argument("--max-speed", type=float, default=None, help="云台最大速度，默认 0.6")
    parser.add_argument(
        "--control-interval",
        type=float,
        default=None,
        help="云台控制刷新间隔秒数，默认 0.1",
    )
    parser.add_argument(
        "--odas-cfg",
        type=Path,
        default=ODASConfig().config_path,
        help="ODAS 配置文件",
    )
    parser.add_argument("--show-odas-log", action="store_true", help="ODAS 输出打到当前终端")
    parser.add_argument(
        "--ptz-backend",
        choices=("serial", "onvif"),
        default="serial",
        help="云台后端，默认 serial",
    )
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="串口云台设备，默认 /dev/ttyUSB0")
    parser.add_argument("--serial-baud", type=int, default=115200, help="串口波特率，默认 115200")
    parser.add_argument(
        "--angle-step",
        type=float,
        default=8.0,
        help="串口云台每次速度控制对应的角度步长，默认 8 度",
    )
    parser.add_argument(
        "--move-time-ms",
        type=int,
        default=120,
        help="串口云台单步移动时间，默认 120ms",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="不自动结束旧的 odas_bridge/odaslive（默认会先清理）",
    )
    parser.add_argument(
        "--mic-check",
        action="store_true",
        help="启动前用 arecord 预检麦克风（默认跳过，避免占用设备）",
    )
    args = parser.parse_args()

    odas_config, sound_config, track_config = build_configs(args)
    preview = MainThreadCameraPreview() if track_config.show_preview else None

    print("=== 单终端声源跟踪 ===")

    def sound_worker() -> None:
        pipeline = SoundPipeline(odas_config, quiet_odas=not args.show_odas_log)
        try:
            pipeline.start(clean_stale=not args.no_clean, mic_check=args.mic_check)
            print("等待 ODAS 稳定输出...")
            if not pipeline.wait_odas_ready(timeout=3.0):
                return

            if args.ptz_backend == "serial":
                ptz = SerialPanTiltBackend(
                    SerialPTZConfig(
                        port=args.serial_port,
                        baud=args.serial_baud,
                        default_time_ms=args.move_time_ms,
                        angle_step_per_speed=args.angle_step,
                    )
                )
            else:
                ptz = PTZCameraController(PTZConfig())

            if not ptz.connect():
                print("云台未连接，摄像头预览和声源坐标继续运行；云台将不会转动。")

            SoundPTZTracker(
                ptz,
                sound_config=sound_config,
                track_config=track_config,
                preview=preview,
            ).run()
        except RuntimeError as exc:
            print(f"启动失败: {exc}")
        except KeyboardInterrupt:
            print("\n已退出")
        finally:
            pipeline.stop()
            if preview is not None:
                preview.stop()

    worker = Thread(target=sound_worker, daemon=True)
    worker.start()

    try:
        if preview is not None:
            preview.run_forever()
        else:
            worker.join()
    except KeyboardInterrupt:
        print("\n已退出")
        if preview is not None:
            preview.stop()
    finally:
        if preview is not None:
            preview.stop()
        worker.join(timeout=3.0)


if __name__ == "__main__":
    main()
