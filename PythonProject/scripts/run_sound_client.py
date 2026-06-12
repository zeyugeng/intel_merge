"""调试 ODAS 声源 TCP 数据流。"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SoundConfig
from core.sound_client import SoundSourceClient


def main():
    client = SoundSourceClient(SoundConfig())
    client.start()
    print("监听声源数据，Ctrl+C 退出")
    try:
        while True:
            valid, xyz = client.parse_latest()
            if valid and xyz:
                print(f"x={xyz[0]:.3f}, y={xyz[1]:.3f}, E={xyz[2]:.3f}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        client.stop()


if __name__ == "__main__":
    main()
