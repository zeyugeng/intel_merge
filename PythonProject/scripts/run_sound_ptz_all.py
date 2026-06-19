#!/usr/bin/env python3
"""一键启动：声源桥接 + ODAS + 云台跟踪（单终端）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import ODASConfig, PTZConfig, PTZTrackConfig, SoundConfig
from core.ptz_camera import PTZCameraController
from core.sound_pipeline import SoundPipeline


def build_configs(args):
    odas_config = ODASConfig(config_path=args.odas_cfg)

    sound_config = SoundConfig(
        host=odas_config.host,
        port=odas_config.python_port,
    )
    if args.energy is not None:
        sound_config.energy_threshold = args.energy

    track_config = PTZTrackConfig(show_preview=not args.no_preview)
    if args.kp_pan is not None:
        track_config.kp_pan = args.kp_pan
    if args.kp_tilt is not None:
        track_config.kp_tilt = args.kp_tilt

    return odas_config, sound_config, track_config


def main() -> None:
    parser = argparse.ArgumentParser(description="单终端：ODAS + 桥接 + 云台声源跟踪")
    parser.add_argument("--no-preview", action="store_true", help="不打开 RTSP 预览")
    parser.add_argument("--energy", type=float, default=None, help="声源能量阈值")
    parser.add_argument("--kp-pan", type=float, default=None, help="水平增益")
    parser.add_argument("--kp-tilt", type=float, default=None, help="俯仰增益")
    parser.add_argument(
        "--odas-cfg",
        type=Path,
        default=ODASConfig().config_path,
        help="ODAS 配置文件",
    )
    parser.add_argument("--show-odas-log", action="store_true", help="ODAS 输出打到当前终端")
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
    pipeline = SoundPipeline(odas_config, quiet_odas=not args.show_odas_log)

    print("=== 单终端声源跟踪 ===")
    try:
        pipeline.start(clean_stale=not args.no_clean, mic_check=args.mic_check)
        print("等待 ODAS 稳定输出...")
        if not pipeline.wait_odas_ready(timeout=3.0):
            return

        ptz = PTZCameraController(PTZConfig())
        if not ptz.connect():
            return

        ptz.track_with_sound(sound_config=sound_config, track_config=track_config)
    except RuntimeError as exc:
        print(f"启动失败: {exc}")
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
