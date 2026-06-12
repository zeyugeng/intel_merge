"""声视融合：ODAS 声源 + YOLO26 视觉。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import CameraConfig, SoundConfig, VisualConfig
from core.fusion import AudioVisualFusion


def main():
    fusion = AudioVisualFusion(
        camera_config=CameraConfig(),
        sound_config=SoundConfig(),
        visual_config=VisualConfig(),
    )
    fusion.run()


if __name__ == "__main__":
    main()
