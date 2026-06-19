#!/usr/bin/env python3
"""ODAS -> Python sound bridge CLI.

ODAS connects to ``odas_port`` and writes streaming JSON. Python clients connect
to ``python_port`` and receive flattened newline-delimited sound-source JSON.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.odas_bridge import OdasBridge


def main() -> None:
    parser = argparse.ArgumentParser(description="ODAS 到 Python 声源桥接")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--odas-port", type=int, default=9001)
    parser.add_argument("--python-port", type=int, default=5000)
    args = parser.parse_args()

    bridge = OdasBridge(args.host, args.odas_port, args.python_port)
    bridge.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop()
        print("桥接服务已退出")


if __name__ == "__main__":
    main()
