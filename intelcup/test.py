import time
from typing import Any, Iterable, Optional

import numpy as np

from MicrophoneArray import SoundSource, odasconnecter, sssprocess, sstprocess
from SoundPredict import SoundPredict


ACTIVE_THRESHOLD = 0.1
PREDICT_INTERVAL_SEC = 3.0
MIN_AUDIO_RMS = 0.001
MIN_CONFIDENCE = 0.1
AUDIO_PATH = "birdnet_input.wav"

SPECIES_FIELDS = (
    "common_name",
    "common name",
    "Common name",
    "Common Name",
    "label",
    "species_name",
    "Species name",
    "Species Name",
    "species",
    "class",
    "class_name",
)
SCIENTIFIC_FIELDS = (
    "scientific_name",
    "scientific name",
    "Scientific name",
    "Scientific Name",
)
CONFIDENCE_FIELDS = (
    "confidence",
    "Confidence",
    "score",
    "Score",
    "probability",
    "Probability",
)


def get_active_source(frame) -> Optional[SoundSource]:
    if sstprocess.is_silent_frame(frame, ACTIVE_THRESHOLD):
        return None

    active_sources = [
        src for src in frame.sources if src.activity > ACTIVE_THRESHOLD
    ]
    if not active_sources:
        return None
    return max(active_sources, key=lambda src: src.activity)


def result_rows(result: Any) -> Iterable[Any]:
    if result is None:
        return []

    if hasattr(result, "to_structured_array"):
        result = result.to_structured_array()

    if hasattr(result, "to_dict") and hasattr(result, "columns"):
        return result.to_dict("records")

    if isinstance(result, dict):
        for key in ("predictions", "detections", "results"):
            value = result.get(key)
            if isinstance(value, (list, tuple)):
                return value
        return [result]

    if isinstance(result, (list, tuple)):
        return result

    try:
        arr = np.asarray(result)
    except ValueError:
        return [result]

    if arr.shape == ():
        return [arr.item()]
    return arr.reshape(-1)


def field_value(row: Any, names) -> Any:
    if isinstance(row, dict):
        lookup = {str(key).lower().replace(" ", "_"): value for key, value in row.items()}
        for name in names:
            value = lookup.get(str(name).lower().replace(" ", "_"))
            if value is not None:
                return value
        return None

    dtype = getattr(row, "dtype", None)
    if dtype is not None and dtype.names:
        lookup = {name.lower().replace(" ", "_"): name for name in dtype.names}
        for name in names:
            real_name = lookup.get(str(name).lower().replace(" ", "_"))
            if real_name is not None:
                return row[real_name]
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def max_audio_rms(sss: sssprocess, sec: int = 3) -> Optional[float]:
    audio = sss.get_last(sec)
    if len(audio) < sss.sr // 2:
        return None

    rms = np.sqrt(np.mean(audio * audio, axis=0))
    return float(np.max(rms))


def format_species_name(species: str) -> str:
    if "_" not in species:
        return species

    scientific, common = species.split("_", 1)
    return f"{common} ({scientific})"


def best_bird_prediction(result: Any) -> Optional[dict]:
    best = None
    best_score = float("-inf")

    for row in result_rows(result):
        species = field_value(row, SPECIES_FIELDS)
        if species is None:
            continue

        confidence = to_float(field_value(row, CONFIDENCE_FIELDS))
        if confidence is not None and confidence < MIN_CONFIDENCE:
            continue

        score = confidence if confidence is not None else 0.0
        if best is None or score > best_score:
            best = {
                "species": format_species_name(str(species)),
                "scientific": field_value(row, SCIENTIFIC_FIELDS),
                "confidence": confidence,
            }
            best_score = score

    return best


def print_detection(
    prediction: Optional[dict],
    source: SoundSource,
    time_stamp: int,
    audio_rms: float,
) -> None:
    location = (
        f"id={source.id}, x={source.x:.3f}, y={source.y:.3f}, "
        f"z={source.z:.3f}, activity={source.activity:.3f}"
    )

    if prediction is None:
        print(
            f"[BirdNET] no bird >= {MIN_CONFIDENCE:.2f} | "
            f"rms={audio_rms:.5f} | {location} | timeStamp={time_stamp}"
        )
        return

    species = prediction["species"]
    scientific = prediction.get("scientific")
    confidence = prediction.get("confidence")

    if scientific is not None:
        species = f"{species} ({scientific})"
    if confidence is not None:
        species = f"{species}, confidence={confidence:.3f}"

    print(
        f"[BirdNET] bird={species} | rms={audio_rms:.5f} | "
        f"{location} | timeStamp={time_stamp}"
    )


if __name__ == "__main__":
    odas = odasconnecter()
    sst = sstprocess()
    sss = sssprocess(sr=32000)
    predictor = SoundPredict()

    last_predict = time.monotonic() - PREDICT_INTERVAL_SEC

    try:
        sst.connect(wait=False)
        sss.start()
        predictor.load_model()
        time.sleep(0.2)
        odas.open_odas()

        while True:
            frame = sst.get_latest()
            source = get_active_source(frame)

            now = time.monotonic()
            if source is not None and now - last_predict >= PREDICT_INTERVAL_SEC:
                audio_rms = max_audio_rms(sss)
                if audio_rms is None or audio_rms < MIN_AUDIO_RMS:
                    last_predict = now
                    continue

                audio_file = sss.save_last_3s_wav(AUDIO_PATH)
                last_predict = now

                if audio_file is not None:
                    result = predictor.predict(audio_file)
                    prediction = best_bird_prediction(result)
                    print_detection(prediction, source, frame.time_stamp, audio_rms)

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Program stopped")
    finally:
        predictor.close()
        sst.close()
        sss.close()
        odas.close_odas()
