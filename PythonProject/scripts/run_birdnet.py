"""BirdNET 鸟类声音识别。

用法:
  python scripts/run_birdnet.py
  python scripts/run_birdnet.py data/audio/testau.wav
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.birdnet_infer import format_predictions, predict_audio
from core.paths import DEFAULT_AUDIO_PATH


def main():
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO_PATH
    if not audio_path.is_absolute():
        audio_path = ROOT / audio_path
    print(f"分析音频: {audio_path}")
    predictions = predict_audio(audio_path)
    print(format_predictions(predictions))


if __name__ == "__main__":
    main()
