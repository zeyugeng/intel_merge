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
from .birdnet_labels import format_species_display, localize_prediction_rows

DEFAULT_LOCALE = "zh"

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        import birdnet

        _MODEL = birdnet.load("acoustic", "2.4", "tf")
    return _MODEL


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


def _predict_modern(audio_path: Path, confidence_threshold: float = 0.15, top_k: int = 5):
    mono_path = _ensure_mono_audio(audio_path)
    model = _get_model()
    return model.predict(
        str(mono_path),
        default_confidence_threshold=confidence_threshold,
        top_k=top_k,
    )


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


def predict_audio(
    audio_path: Union[str, Path] = DEFAULT_AUDIO_PATH,
    confidence_threshold: float = 0.15,
    top_k: int = 5,
):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    import birdnet

    if hasattr(birdnet, "load"):
        return _predict_modern(audio_path, confidence_threshold, top_k)
    return _predict_legacy(audio_path)


def summarize_predictions(
    predictions: Any,
    top_k: int = 5,
    locale: str = DEFAULT_LOCALE,
) -> list[dict[str, float | str]]:
    """Extract top species rows for logging / storage."""
    rows: list[dict[str, float | str]] = []

    if hasattr(predictions, "to_structured_array"):
        arr = predictions.to_structured_array()
        names = getattr(arr.dtype, "names", None)
        for item in arr:
            if names and "species_name" in names and "confidence" in names:
                rows.append(
                    {
                        "species": str(item["species_name"]),
                        "confidence": float(item["confidence"]),
                    }
                )
            else:
                rows.append(
                    {
                        "species": str(item[3] if len(item) > 3 else item[-2]),
                        "confidence": float(item[4] if len(item) > 4 else item[-1]),
                    }
                )
    elif isinstance(predictions, list):
        for row in predictions:
            rows.append(
                {
                    "species": str(row["species_name"]),
                    "confidence": float(row["confidence"]),
                }
            )

    rows.sort(key=lambda r: r["confidence"], reverse=True)
    rows = rows[:top_k]
    return localize_prediction_rows(rows, locale=locale)


def format_predictions(predictions: Any, locale: str = DEFAULT_LOCALE) -> str:
    if hasattr(predictions, "to_structured_array"):
        arr = predictions.to_structured_array()
        lines = ["start_time\tend_time\tspecies_name\tconfidence"]
        names = getattr(arr.dtype, "names", None)
        for item in arr:
            if names and "species_name" in names:
                species = format_species_display(str(item["species_name"]), locale=locale)
                conf = float(item["confidence"])
                start = float(item["start_time"])
                end = float(item["end_time"])
            else:
                species = format_species_display(str(item[3]), locale=locale)
                conf = float(item[4] if len(item) > 4 else item[-1])
                start = float(item[1])
                end = float(item[2])
            lines.append(f"{start:.2f}\t{end:.2f}\t{species}\t{conf:.6f}")
        return "\n".join(lines)

    lines = ["start_time\tend_time\tspecies_name\tconfidence"]
    for row in predictions:
        species = format_species_display(str(row["species_name"]), locale=locale)
        lines.append(
            f"{row['start_time']:.2f}\t{row['end_time']:.2f}\t"
            f"{species}\t{row['confidence']:.6f}"
        )
    return "\n".join(lines)


def is_non_bird_species(species: str) -> bool:
    raw = species
    if "（" in species:
        raw = species.split("（", 1)[0]
    markers = ("human", "non-vocal", "engine", "motor vehicle", "vehicle", "人类")
    text = raw.lower()
    if any(m in text for m in markers):
        return True
    return any(m in species for m in ("人类非语言", "人类语音", "人类哨声"))


def bird_species_only(
    rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Drop Human / engine / non-vocal rows from BirdNET live output."""
    return [
        row
        for row in rows
        if not is_non_bird_species(str(row.get("species_raw", row["species"])))
    ]


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
