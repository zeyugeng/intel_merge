import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import soundfile as sf

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("birdnet").setLevel(logging.ERROR)

from .paths import DEFAULT_AUDIO_PATH


def _ensure_mono_audio(audio_path: Path) -> Path:
    info = sf.info(audio_path)
    if info.channels == 1:
        return audio_path

    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    mono_path = Path(tempfile.gettempdir()) / f"birdnet_mono_{audio_path.stem}.wav"
    sf.write(mono_path, data, sr)
    return mono_path


def _predict_modern(audio_path: Path):
    import birdnet

    # 0.2.x 对双声道兼容性更好，但统一转 mono 可避免边界问题
    mono_path = _ensure_mono_audio(audio_path)
    model = birdnet.load("acoustic", "2.4", "tf")
    return model.predict(str(mono_path))


def _predict_legacy(audio_path: Path) -> List[dict]:
    from birdnet import predict_species_within_audio_file

    mono_path = _ensure_mono_audio(audio_path)
    rows: List[dict] = []
    for (start, end), species_map in predict_species_within_audio_file(
        mono_path,
        silent=True,
    ):
        for species, confidence in species_map.items():
            rows.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "species_name": species,
                    "confidence": confidence,
                }
            )
    return rows


def predict_audio(audio_path: Union[str, Path] = DEFAULT_AUDIO_PATH):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    import birdnet

    if hasattr(birdnet, "load"):
        return _predict_modern(audio_path)
    return _predict_legacy(audio_path)3


def format_predictions(predictions: Any) -> str:
    if hasattr(predictions, "to_structured_array"):
        return str(predictions.to_structured_array())

    lines = ["start_time\tend_time\tspecies_name\tconfidence"]
    for row in predictions:
        lines.append(
            f"{row['start_time']:.2f}\t{row['end_time']:.2f}\t"
            f"{row['species_name']}\t{row['confidence']:.6f}"
        )
    return "\n".join(lines)


def benchmark_session(audio_path: Union[str, Path] = DEFAULT_AUDIO_PATH, repeats: int = 3):
    import time

    import birdnet

    if not hasattr(birdnet, "load"):
        raise RuntimeError("benchmark_session 需要 birdnet>=0.2（Python>=3.11）")

    audio_path = Path(audio_path)
    model = birdnet.load("acoustic", "2.4", "tf")
    timings = []
    results = []
    with model.predict_session() as session:
        for i in range(repeats):
            start = time.time()
            result = session.run([str(audio_path)])
            elapsed = time.time() - start
            timings.append(elapsed)
            results.append(result.to_structured_array())
            print(f"第 {i + 1} 次推理: {elapsed:.2f}s")
    return timings, results
