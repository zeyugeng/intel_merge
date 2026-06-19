"""声源坐标实时驱动云台转动。

一键启动（推荐，单终端）:
  python scripts/run_sound_ptz_all.py

或手动分步（桥接 + ODAS 需另开终端）:
  python scripts/run_ptz_track.py
  python scripts/run_ptz_track.py --no-preview
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import PTZConfig, PTZTrackConfig, SoundConfig
from core.ptz_camera import PTZCameraController


def main():
    parser = argparse.ArgumentParser(description="声源坐标驱动云台跟踪")
    parser.add_argument("--no-preview", action="store_true", help="不打开 RTSP 预览窗口")
    parser.add_argument("--energy", type=float, default=None, help="声源能量阈值")
    parser.add_argument("--kp-pan", type=float, default=None, help="水平增益")
    parser.add_argument("--kp-tilt", type=float, default=None, help="俯仰增益")
    args = parser.parse_args()

    sound_config = SoundConfig()
    if args.energy is not None:
        sound_config.energy_threshold = args.energy

    track_config = PTZTrackConfig(show_preview=not args.no_preview)
    if args.kp_pan is not None:
        track_config.kp_pan = args.kp_pan
    if args.kp_tilt is not None:
        track_config.kp_tilt = args.kp_tilt

    ptz = PTZCameraController(PTZConfig())
    if not ptz.connect():
        return

    ptz.track_with_sound(sound_config=sound_config, track_config=track_config)


if __name__ == "__main__":
    main()
