import math
import re
import threading
import time
from typing import Any, Iterable, Optional

import numpy as np

from MicrophoneArray import SoundSource, odasconnecter, sssprocess, sstprocess
from Ptz import PTZ


ACTIVE_THRESHOLD = 0.1
PREDICT_INTERVAL_SEC = 3.0
MIN_AUDIO_RMS = 0.001
MIN_CONFIDENCE = 0.1
AUDIO_PATH = "birdnet_input.wav"
PTZ_MOVE_TIME_MS = 1000
PTZ_TURN_COOLDOWN_SEC = 0.5
CAMERA_ID = 0
YOLO_MODEL_PATH = "yolo26n.pt"
YOLO_PREVIEW_WIDTH = 1280
YOLO_PREVIEW_HEIGHT = 720
AUDIO_BIND_RETRY_COUNT = 5
AUDIO_BIND_RETRY_DELAY_SEC = 1.0

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

# BirdNET results are usually bird labels. These words are used only as a
# blacklist for obvious non-bird sounds so we do not point the PTZ at people.
REJECTED_SOUND_KEYWORDS = (
    "human",
    "homo sapiens",
    "speech",
    "voice",
    "conversation",
    "talk",
    "talking",
    "shout",
    "laugh",
    "cough",
    "sneeze",
    "cry",
    "music",
    "vehicle",
    "traffic",
    "engine",
    "motor",
    "siren",
    "alarm",
    "wind",
    "rain",
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
        lookup = {
            str(key).lower().replace(" ", "_"): value
            for key, value in row.items()
        }
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


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def rejected_sound_reason(*values: Any) -> Optional[str]:
    text = " ".join(normalize_label(value) for value in values if value is not None)
    padded_text = f" {text} "

    for keyword in REJECTED_SOUND_KEYWORDS:
        normalized_keyword = normalize_label(keyword)
        if f" {normalized_keyword} " in padded_text:
            return keyword
    return None


def format_species_name(species: str) -> str:
    if "_" not in species:
        return species

    scientific, common = species.split("_", 1)
    return f"{common} ({scientific})"


def best_prediction(result: Any) -> Optional[dict]:
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
            scientific = field_value(row, SCIENTIFIC_FIELDS)
            reason = rejected_sound_reason(species, scientific)
            best = {
                "species": format_species_name(str(species)),
                "scientific": scientific,
                "confidence": confidence,
                "rejected_reason": reason,
            }
            best_score = score

    return best


def source_to_ptz_angles(source: SoundSource) -> tuple[float, float]:
    pan_angle = math.degrees(math.atan2(source.x, source.z))
    horizontal_distance = math.sqrt(source.x * source.x + source.z * source.z)
    tilt_angle = math.degrees(math.atan2(source.y, horizontal_distance))
    return -pan_angle, -tilt_angle


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
            f"[BirdNET] no valid result >= {MIN_CONFIDENCE:.2f} | "
            f"rms={audio_rms:.5f} | {location} | timeStamp={time_stamp}"
        )
        return

    species = prediction["species"]
    scientific = prediction.get("scientific")
    confidence = prediction.get("confidence")
    rejected_reason = prediction.get("rejected_reason")

    if scientific is not None:
        species = f"{species} ({scientific})"
    if confidence is not None:
        species = f"{species}, confidence={confidence:.3f}"

    if rejected_reason is not None:
        print(
            f"[BirdNET] ignored non-bird sound={species}, reason={rejected_reason} | "
            f"rms={audio_rms:.5f} | {location} | timeStamp={time_stamp}"
        )
        return

    print(
        f"[BirdNET] bird={species} | rms={audio_rms:.5f} | "
        f"{location} | timeStamp={time_stamp}"
    )


def turn_ptz_to_source(ptz: PTZ, source: SoundSource) -> None:
    pan_angle, tilt_angle = source_to_ptz_angles(source)
    print(f"[PTZ] turn to pan={pan_angle:.2f}, tilt={tilt_angle:.2f}")
    ptz.move_angle(-pan_angle, -tilt_angle, t=PTZ_MOVE_TIME_MS)


def start_audio_socket_with_retry(name, port, start_func, close_func) -> None:
    last_exc = None

    for attempt in range(1, AUDIO_BIND_RETRY_COUNT + 1):
        try:
            start_func()
            return
        except OSError as exc:
            last_exc = exc
            close_func()
            if attempt < AUDIO_BIND_RETRY_COUNT:
                print(
                    f"[audio] {name} port {port} is busy, "
                    f"retry {attempt}/{AUDIO_BIND_RETRY_COUNT}"
                )
                time.sleep(AUDIO_BIND_RETRY_DELAY_SEC)

    raise RuntimeError(
        f"{name} port {port} is already in use, "
        "please close the old main.py/ODAS process"
    ) from last_exc


def resize_for_preview(frame: np.ndarray, cv2_module) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(YOLO_PREVIEW_WIDTH / w, YOLO_PREVIEW_HEIGHT / h)
    return cv2_module.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2_module.INTER_AREA,
    )


def setup_camera_yolo_preview():
    camera = None
    window = "YOLO26n realtime"

    try:
        from Camera import Camera
        import cv2
        from ultralytics import YOLO

        camera = Camera(camera_id=CAMERA_ID)
        camera.set_zoom(1000)
        print(f"[YOLO] loading model: {YOLO_MODEL_PATH}")
        model = YOLO(YOLO_MODEL_PATH)

        if not camera.get_camera():
            print("[Camera] open failed, YOLO preview disabled")
            camera.release()
            return None

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, YOLO_PREVIEW_WIDTH, YOLO_PREVIEW_HEIGHT)
        print("[Camera] YOLO preview ready, press q in the window to stop")
        return camera, model, cv2, window
    except Exception as exc:
        print(f"[Camera] YOLO preview disabled: {exc}")
        if camera is not None:
            camera.release()
        return None


def run_camera_yolo_preview(camera_ctx, stop_event: threading.Event) -> None:
    camera, model, cv2, window = camera_ctx

    try:
        while not stop_event.is_set():
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            preview = resize_for_preview(frame, cv2)
            results = model(preview, verbose=False)
            annotated = results[0].plot()
            cv2.imshow(window, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break

            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()
                    break
            except cv2.error:
                pass
    except Exception as exc:
        print(f"[Camera] YOLO preview stopped: {exc}")
    finally:
        camera.release()


def run_audio_tracking(stop_event: threading.Event) -> None:
    from SoundPredict import SoundPredict

    odas = odasconnecter()
    sst = sstprocess()
    sss = sssprocess(sr=32000)
    predictor = SoundPredict()
    ptz = None

    last_predict = time.monotonic() - PREDICT_INTERVAL_SEC
    last_turn = 0.0

    try:
        odas.release_odas()
        predictor.load_model()
        ptz = PTZ()

        start_audio_socket_with_retry(
            "SST",
            sst.port,
            lambda: sst.connect(wait=False),
            sst.close,
        )
        start_audio_socket_with_retry("SSS", sss.port, sss.start, sss.close)

        time.sleep(0.2)
        odas.open_odas()
        print("[main] ready")

        while not stop_event.is_set():
            frame = sst.get_latest()
            source = get_active_source(frame)
            now = time.monotonic()

            if source is None:
                time.sleep(0.05)
                continue

            if now - last_predict < PREDICT_INTERVAL_SEC:
                time.sleep(0.05)
                continue

            audio_rms = max_audio_rms(sss)
            if audio_rms is None or audio_rms < MIN_AUDIO_RMS:
                last_predict = now
                time.sleep(0.05)
                continue

            audio_file = sss.save_last_3s_wav(AUDIO_PATH)
            last_predict = now
            if audio_file is None:
                time.sleep(0.05)
                continue

            result = predictor.predict(audio_file)
            prediction = best_prediction(result)
            print_detection(prediction, source, frame.time_stamp, audio_rms)

            if prediction is None or prediction.get("rejected_reason") is not None:
                time.sleep(0.05)
                continue

            turn_now = time.monotonic()
            if turn_now - last_turn >= PTZ_TURN_COOLDOWN_SEC:
                turn_ptz_to_source(ptz, source)
                last_turn = turn_now

            time.sleep(0.05)
    except Exception as exc:
        print(f"[audio] stopped: {exc}")
        stop_event.set()
    finally:
        predictor.close()
        sst.close()
        sss.close()
        odas.close_odas()
        if ptz is not None:
            ptz.close()


def main() -> None:
    stop_event = threading.Event()
    camera_ctx = setup_camera_yolo_preview()
    audio_thread = threading.Thread(
        target=run_audio_tracking,
        args=(stop_event,),
        daemon=True,
    )

    try:
        audio_thread.start()
        if camera_ctx is None:
            print("[main] running audio tracking without camera preview")
            while not stop_event.is_set():
                time.sleep(0.2)
        else:
            run_camera_yolo_preview(camera_ctx, stop_event)
    except KeyboardInterrupt:
        print("[main] stopped")
    finally:
        stop_event.set()
        audio_thread.join(timeout=5)

#fuser -k 5000/tcp 10010/tcp
if __name__ == "__main__":
    main()
