"""云台控制 + RTSP 预览 + 声源数据。"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import PTZConfig, SoundConfig
from core.ptz_camera import PTZCameraController


def main():
    ptz = PTZCameraController(PTZConfig())
    if not ptz.connect():
        return

    ptz.move_ptz(pan_speed=0.3, tilt_speed=0.1)
    time.sleep(0.2)
    ptz.stop_ptz()
    ptz.preview_with_sound(SoundConfig())


if __name__ == "__main__":
    main()
