import math
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2

from Camera import Camera
from MicrophoneArray import MicrophoneArray, SoundSource
from Picturepredic import Detection, PicturePredict
from Ptz import PTZ
from SoundPredict import SoundPredict


MIC_HOST = os.getenv("MIC_HOST", "0.0.0.0")
MIC_PORT = int(os.getenv("MIC_PORT", "5000"))
SSS_DIR = Path(os.getenv("SSS_DIR", "sss_output"))
SPECIES_LIST = os.getenv("SPECIES_LIST", "species_list.txt")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt")
PTZ_PORT = os.getenv("PTZ_PORT", "COM3")
PTZ_BAUD = int(os.getenv("PTZ_BAUD", "115200"))

SOUND_ACTIVITY_THRESHOLD = float(os.getenv("SOUND_ACTIVITY_THRESHOLD", "0.35"))
BIRD_CONF_THRESHOLD = float(os.getenv("BIRD_CONF_THRESHOLD", "0.25"))
PTZ_SOUND_TIME_MS = int(os.getenv("PTZ_SOUND_TIME_MS", "1000"))
PTZ_TRACK_TIME_MS = int(os.getenv("PTZ_TRACK_TIME_MS", "200"))
PAN_DEG_PER_PIXEL = float(os.getenv("PAN_DEG_PER_PIXEL", "0.04"))
TILT_DEG_PER_PIXEL = float(os.getenv("TILT_DEG_PER_PIXEL", "0.04"))
TRACK_DEAD_ZONE = float(os.getenv("TRACK_DEAD_ZONE", "0.08"))

# Empty means any BirdNET species in SPECIES_LIST is treated as important.
IMPORTANT_SPECIES = {
    item.strip().lower()
    for item in os.getenv("IMPORTANT_SPECIES", "").split(",")
    if item.strip()
}


def find_latest_audio(folder: Path, after_mtime: float = 0.0) -> Optional[Path]:
    if not folder.exists():
        return None

    candidates = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
        and path.stat().st_mtime > after_mtime
    ]
    if not candidates:
        return None

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    # Give the C process a short moment to finish writing the SSS file.
    size = latest.stat().st_size
    time.sleep(0.1)
    if latest.exists() and latest.stat().st_size == size:
        return latest
    return None


def strongest_active_source(sources: Iterable[SoundSource]) -> Optional[SoundSource]:
    active_sources = [
        source for source in sources if source.activity >= SOUND_ACTIVITY_THRESHOLD
    ]
    if not active_sources:
        return None
    return max(active_sources, key=lambda source: source.activity)


def source_to_ptz_angle(source: SoundSource) -> tuple[float, float]:
    pan = math.degrees(math.atan2(source.y, source.x))
    horizontal = math.hypot(source.x, source.y)
    tilt = math.degrees(math.atan2(source.z, horizontal))
    return pan, tilt


def iter_birdnet_rows(result) -> Iterable[dict]:
    if hasattr(result, "to_structured_array"):
        rows = result.to_structured_array()
    else:
        rows = result

    for row in rows:
        if hasattr(row, "dtype") and row.dtype.names:
            yield {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in row.dtype.names}
        elif isinstance(row, dict):
            yield row


def row_text(row: dict) -> str:
    parts = []
    for key in ("species", "scientific_name", "common_name", "label", "class", "name"):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def row_confidence(row: dict) -> float:
    for key in ("confidence", "score", "probability", "prob", "logit"):
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def is_important_bird(result) -> bool:
    for row in iter_birdnet_rows(result):
        if row_confidence(row) < BIRD_CONF_THRESHOLD:
            continue

        text = row_text(row)
        if not IMPORTANT_SPECIES or any(species in text for species in IMPORTANT_SPECIES):
            print(f"BirdNET hit: {row}")
            return True
    return False


def wait_for_important_sound(mic: MicrophoneArray, predictor: SoundPredict, ptz: PTZ) -> None:
    print("Stage 1: waiting for active sound source and important bird call...")
    last_audio_mtime = 0.0
    last_frame_stamp = None

    while True:
        frame = mic.get_latest()
        if frame is None or frame.time_stamp == last_frame_stamp:
            time.sleep(0.05)
            continue

        last_frame_stamp = frame.time_stamp
        source = strongest_active_source(frame.sources)
        if source is None:
            continue

        audio_file = find_latest_audio(SSS_DIR, last_audio_mtime)
        if audio_file is None:
            continue

        last_audio_mtime = audio_file.stat().st_mtime
        print(f"Active source detected, predicting {audio_file}...")
        result = predictor.predict(audio_file)
        if not is_important_bird(result):
            continue

        pan, tilt = source_to_ptz_angle(source)
        print(f"Important bird detected. Moving PTZ to pan={pan:.1f}, tilt={tilt:.1f}")
        ptz.move_angle(pan, tilt, PTZ_SOUND_TIME_MS)
        return


def draw_detection(frame, detection: Detection) -> None:
    cv2.rectangle(
        frame,
        (int(detection.x1), int(detection.y1)),
        (int(detection.x2), int(detection.y2)),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"{detection.label} {detection.confidence:.2f}",
        (int(detection.x1), max(20, int(detection.y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def track_bird_with_camera(camera: Camera, ptz: PTZ, picture_predictor: PicturePredict) -> None:
    print("Stage 2: tracking bird with camera. Press q to quit.")
    camera.get_camera()

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        height, width = frame.shape[:2]
        detection = picture_predictor.predict_bird(frame)

        if detection is not None:
            center_x, center_y = detection.center
            dx = center_x - width / 2
            dy = center_y - height / 2

            if abs(dx) > width * TRACK_DEAD_ZONE or abs(dy) > height * TRACK_DEAD_ZONE:
                pan, tilt = ptz.get_current_angle()
                target_pan = pan + dx * PAN_DEG_PER_PIXEL
                target_tilt = tilt - dy * TILT_DEG_PER_PIXEL
                ptz.move_angle(target_pan, target_tilt, PTZ_TRACK_TIME_MS)

            draw_detection(frame, detection)

        cv2.imshow("Bird tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    mic = MicrophoneArray(host=MIC_HOST, port=MIC_PORT)
    ptz = PTZ(port=PTZ_PORT, baud=PTZ_BAUD)
    camera = Camera()
    picture_predictor = PicturePredict(model_path=YOLO_MODEL, confidence=BIRD_CONF_THRESHOLD)

    try:
        mic.connect(wait=False)
        with SoundPredict(species_list=SPECIES_LIST) as predictor:
            while True:
                wait_for_important_sound(mic, predictor, ptz)
                track_bird_with_camera(camera, ptz, picture_predictor)
    finally:
        mic.close()
        ptz.close()
        if camera.cap is not None:
            camera.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
