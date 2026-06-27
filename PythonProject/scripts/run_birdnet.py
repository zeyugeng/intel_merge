"""BirdNET 鸟类声音识别。

用法:
  python scripts/run_birdnet.py
  python scripts/run_birdnet.py data/audio/testau.wav
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.birdnet_infer import DEFAULT_LOCALE, format_predictions, predict_audio
from core.paths import DEFAULT_AUDIO_PATH


def main():
    parser = argparse.ArgumentParser(description="BirdNET 离线识别")
    parser.add_argument("audio", nargs="?", default=str(DEFAULT_AUDIO_PATH), help="wav 路径")
    parser.add_argument(
        "--locale",
        default=DEFAULT_LOCALE,
        help="鸟类名称语言，默认 zh",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_absolute():
        audio_path = ROOT / audio_path
    print(f"分析音频: {audio_path}")
    predictions = predict_audio(audio_path)
    print(format_predictions(predictions, locale=args.locale))


if __name__ == "__main__":
    main()
