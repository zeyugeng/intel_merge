"""Sound-source driven PTZ tracking."""

from __future__ import annotations

import glob
import importlib.util
import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional, Protocol, Tuple


def _ensure_gui_display() -> None:
    """补齐 Cursor/终端环境下 OpenCV 窗口需要的显示变量。"""
    uid = os.getuid()
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"
    if os.path.isdir(runtime):
        os.environ.setdefault("XDG_RUNTIME_DIR", runtime)

    if not os.environ.get("DISPLAY") and os.path.isdir("/tmp/.X11-unix"):
        os.environ.setdefault("DISPLAY", ":0")

    if not os.environ.get("XAUTHORITY"):
        candidates = sorted(glob.glob(f"{runtime}/.mutter-Xwaylandauth.*"))
        if candidates:
            os.environ["XAUTHORITY"] = candidates[-1]

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")


_ensure_gui_display()

import cv2

from .config import PTZTrackConfig, SoundConfig
from .sound_client import SoundSourceClient


INTELCUP_CAMERA_PATH = Path(__file__).resolve().parents[2] / "intelcup" / "Camera.py"
_camera_spec = importlib.util.spec_from_file_location("intelcup_camera", INTELCUP_CAMERA_PATH)
if _camera_spec and _camera_spec.loader:
    _camera_module = importlib.util.module_from_spec(_camera_spec)
    _camera_spec.loader.exec_module(_camera_module)
    LegacyUSBCamera = _camera_module.Camera
else:
    LegacyUSBCamera = None

PREOPENED_USB_CAMERA = None
PREVIEW_WINDOW_NAME = "Camera 实时预览 (按 q 退出)"
PREVIEW_W = 1280
PREVIEW_H = 720


def preopen_usb_preview() -> bool:
    """Open the exact USB camera implementation from intelcup/Camera.py."""
    global PREOPENED_USB_CAMERA

    if PREOPENED_USB_CAMERA is not None:
        return True
    if LegacyUSBCamera is None:
        print(f"无法加载摄像头文件: {INTELCUP_CAMERA_PATH}")
        return False

    camera_id = int(os.getenv("PTZ_PREVIEW_CAMERA_ID", "0"))
    width = int(os.getenv("PTZ_PREVIEW_WIDTH", "1920"))
    height = int(os.getenv("PTZ_PREVIEW_HEIGHT", "1080"))

    camera = LegacyUSBCamera(camera_id=camera_id)
    camera.WIDTH = width
    camera.HEIGHT = height
    if not camera.get_camera():
        return False

    PREOPENED_USB_CAMERA = camera
    real_w = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = camera.cap.get(cv2.CAP_PROP_FPS)
    print(f"已按 intelcup/Camera.py 打开摄像头: /dev/video{camera_id} {real_w}x{real_h} FPS={real_fps}")
    return True


def _read_camera_frame(camera, attempts: int = 10):
    frame = None
    for _ in range(attempts):
        frame = camera.get_frame()
        if frame is not None and frame.mean() > 1.0:
            return frame
        cv2.waitKey(1)
    return frame


def _show_camera_frame(
    camera,
    sound_xyz: Optional[Tuple[float, float, float, float]],
    window_name: str,
) -> bool:
    frame = _read_camera_frame(camera)
    if frame is None:
        print("USB 摄像头读取失败")
        return False

    if sound_xyz:
        sx, sy, sz, energy = sound_xyz
        text = f"USB | x={sx:+.2f} y={sy:+.2f} z={sz:+.2f} E={energy:.2f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    h, w = frame.shape[:2]
    scale = min(PREVIEW_W / w, PREVIEW_H / h)
    frame_show = cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, PREVIEW_W, PREVIEW_H)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    cv2.imshow(window_name, frame_show)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


class CameraPreviewThread:
    """Continuously show intelcup/Camera.py frames while sound/PTZ runs separately."""

    def __init__(self, window_name: str = PREVIEW_WINDOW_NAME):
        self.window_name = window_name
        self.camera = None
        self.latest_sound: Optional[Tuple[float, float, float, float]] = None
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._running = False

    def start(self) -> bool:
        if self._running:
            return True
        if LegacyUSBCamera is None:
            print(f"无法加载摄像头文件: {INTELCUP_CAMERA_PATH}")
            return False

        camera_id = int(os.getenv("PTZ_PREVIEW_CAMERA_ID", "0"))
        width = int(os.getenv("PTZ_PREVIEW_WIDTH", "1920"))
        height = int(os.getenv("PTZ_PREVIEW_HEIGHT", "1080"))

        self.camera = LegacyUSBCamera(camera_id=camera_id)
        self.camera.WIDTH = width
        self.camera.HEIGHT = height
        if not self.camera.get_camera():
            self.camera = None
            return False

        real_w = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = self.camera.cap.get(cv2.CAP_PROP_FPS)
        print(f"摄像头线程已启动: /dev/video{camera_id} {real_w}x{real_h} FPS={real_fps}")
        print(f"若看不到窗口，请 Alt+Tab 切换到「{self.window_name}」")

        self._stop_event.clear()
        self._running = True
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def update_sound(self, sound_xyz: Optional[Tuple[float, float, float, float]]) -> None:
        with self._lock:
            self.latest_sound = sound_xyz

    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self._running = False

    def _run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, PREVIEW_W, PREVIEW_H)
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

        try:
            _read_camera_frame(self.camera, attempts=30)
            while not self._stop_event.is_set():
                with self._lock:
                    sound_xyz = self.latest_sound
                if not _show_camera_frame(self.camera, sound_xyz, self.window_name):
                    self._stop_event.set()
                    break
        finally:
            self._running = False


class MainThreadCameraPreview:
    """Run the intelcup/Camera.py display loop on the main thread."""

    def __init__(self, window_name: str = PREVIEW_WINDOW_NAME):
        self.window_name = window_name
        self.camera = None
        self.latest_sound: Optional[Tuple[float, float, float, float]] = None
        self._lock = Lock()
        self._stop_event = Event()

    def update_sound(self, sound_xyz: Optional[Tuple[float, float, float, float]]) -> None:
        with self._lock:
            self.latest_sound = sound_xyz

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def run_forever(self) -> bool:
        if LegacyUSBCamera is None:
            print(f"无法加载摄像头文件: {INTELCUP_CAMERA_PATH}")
            return False

        camera_id = int(os.getenv("PTZ_PREVIEW_CAMERA_ID", "0"))
        width = int(os.getenv("PTZ_PREVIEW_WIDTH", "1920"))
        height = int(os.getenv("PTZ_PREVIEW_HEIGHT", "1080"))

        self.camera = LegacyUSBCamera(camera_id=camera_id)
        self.camera.WIDTH = width
        self.camera.HEIGHT = height
        if not self.camera.get_camera():
            return False

        window = self.window_name
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, PREVIEW_W, PREVIEW_H)
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

        real_w = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"实时预览已启动: {real_w}x{real_h} -> 窗口 {PREVIEW_W}x{PREVIEW_H}")
        print(f"若看不到窗口，请 Alt+Tab 切换到「{window}」")

        try:
            while not self._stop_event.is_set():
                ret, frame = self.camera.cap.read()
                if not ret:
                    print("读取失败")
                    break

                h, w = frame.shape[:2]
                scale = min(PREVIEW_W / w, PREVIEW_H / h)
                frame_show = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

                with self._lock:
                    sound_xyz = self.latest_sound
                if sound_xyz:
                    sx, sy, sz, energy = sound_xyz
                    cv2.putText(
                        frame_show,
                        f"x={sx:+.2f} y={sy:+.2f} z={sz:+.2f} E={energy:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow(window, frame_show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("f"):
                    try:
                        top = cv2.getWindowProperty(window, cv2.WND_PROP_TOPMOST)
                        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 0 if top else 1)
                    except cv2.error:
                        pass
        finally:
            self._stop_event.set()
            if self.camera is not None:
                self.camera.release()
                self.camera = None
        return True


class PTZBackend(Protocol):
    stream_uri: Optional[str]

    def move_ptz(
        self,
        pan_speed: float = 0.0,
        tilt_speed: float = 0.0,
        zoom_speed: float = 0.0,
    ) -> None: ...

    def stop_ptz(self, stop_zoom: bool = True) -> None: ...

    def get_stream_uri(self) -> Optional[str]: ...


class SoundToVelocityController:
    """Maps sound coordinates into PTZ continuous-move speeds."""

    def __init__(self, sound_config: SoundConfig, track_config: PTZTrackConfig):
        self.sound_config = sound_config
        self.track_config = track_config

    def compute(self, sound_x: float, sound_y: float, energy: float) -> Tuple[float, float, float]:
        if energy <= self.sound_config.energy_threshold:
            return 0.0, 0.0, 0.0

        pan = 0.0
        tilt = 0.0
        if abs(sound_x) > self.track_config.deadzone:
            pan = self.track_config.kp_pan * sound_x
        if abs(sound_y) > self.track_config.deadzone:
            y = -sound_y if self.track_config.invert_y else sound_y
            tilt = self.track_config.kp_tilt * y

        zoom = 0.0
        if self.track_config.enable_zoom and energy > self.track_config.zoom_energy:
            zoom = self.track_config.kp_zoom * min(energy, 1.0)

        max_speed = self.track_config.max_speed
        pan = max(-max_speed, min(max_speed, pan))
        tilt = max(-max_speed, min(max_speed, tilt))
        zoom = max(-max_speed, min(max_speed, zoom))
        return pan, tilt, zoom


class RTSPPreview:
    """Optional preview window with latest sound coordinates overlaid."""

    def __init__(self, backend: PTZBackend, window_name: str = PREVIEW_WINDOW_NAME):
        self.backend = backend
        self.window_name = window_name
        self.cap = None
        self.usb_camera = None
        self.source_name = "USB"
        self.preview_w = PREVIEW_W
        self.preview_h = PREVIEW_H

    def open(self) -> bool:
        if self._open_usb_camera():
            return True

        print("本地 USB 摄像头打开失败，尝试打开 RTSP ...")
        if not self.backend.stream_uri:
            self.backend.get_stream_uri()
        self.cap = cv2.VideoCapture(self.backend.stream_uri, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.source_name = "RTSP"
            print("摄像头预览已启动: RTSP")
            self._prepare_window()
            return True

        print(f"无法打开 RTSP: {self.backend.stream_uri}")
        print("无法打开摄像头画面，仅声源跟踪")
        self.cap.release()
        self.cap = None
        return False

    def _open_usb_camera(self) -> bool:
        global PREOPENED_USB_CAMERA

        if LegacyUSBCamera is None:
            print(f"无法加载摄像头文件: {INTELCUP_CAMERA_PATH}")
            return False

        if PREOPENED_USB_CAMERA is None and not preopen_usb_preview():
            return False

        self.usb_camera = PREOPENED_USB_CAMERA
        PREOPENED_USB_CAMERA = None

        self.source_name = "USB"
        self._prepare_window()
        real_w = int(self.usb_camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.usb_camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _read_camera_frame(self.usb_camera, attempts=30)
        print(f"实时预览已启动: {real_w}x{real_h} -> 窗口 {self.preview_w}x{self.preview_h}")
        print(f"若看不到窗口，请 Alt+Tab 切换到「{self.window_name}」")
        return True

    def _prepare_window(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.preview_w, self.preview_h)
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    def show(self, sound_xyz: Optional[Tuple[float, float, float, float]]) -> bool:
        if self.source_name == "USB" and self.usb_camera is not None:
            return _show_camera_frame(self.usb_camera, sound_xyz, self.window_name)
        elif self.cap is None:
            return True
        else:
            ret, frame = self.cap.read()
            if not ret:
                if self.source_name == "RTSP" and self._open_usb_camera():
                    frame = self.usb_camera.get_frame()
                if frame is None:
                    print("摄像头读取失败")
                    return False

        if sound_xyz:
            sx, sy, sz, energy = sound_xyz
            text = f"{self.source_name} | x={sx:+.2f} y={sy:+.2f} z={sz:+.2f} E={energy:.2f}"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        if self.source_name == "USB":
            h, w = frame.shape[:2]
            scale = min(self.preview_w / w, self.preview_h / h)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(self.window_name, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        if self.usb_camera is not None:
            self.usb_camera.release()
            self.usb_camera = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            cv2.destroyAllWindows()


class SoundPTZTracker:
    """Coordinates sound input, PTZ movement, and optional preview."""

    def __init__(
        self,
        backend: PTZBackend,
        sound_config: Optional[SoundConfig] = None,
        track_config: Optional[PTZTrackConfig] = None,
        preview: Optional[CameraPreviewThread] = None,
    ):
        self.backend = backend
        self.sound_config = sound_config or SoundConfig()
        self.track_config = track_config or PTZTrackConfig()
        self.controller = SoundToVelocityController(self.sound_config, self.track_config)
        self.preview = preview

    def run(self) -> None:
        sound = SoundSourceClient(self.sound_config)
        sound.start()

        preview = self.preview
        owns_preview = False
        if self.track_config.show_preview and preview is None:
            preview = CameraPreviewThread()
            owns_preview = preview.start()

        print("声源跟踪已启动：对着麦克风发声，云台将转向声源方向")
        print("按 q 退出预览窗口（无预览时 Ctrl+C）")
        last_control = 0.0
        moving = False
        latest_sound = None

        try:
            while True:
                now = time.monotonic()
                valid, sound_xyz = sound.parse_latest()
                if valid and sound_xyz:
                    latest_sound = sound_xyz
                    if preview is not None:
                        preview.update_sound(latest_sound)

                if valid and sound_xyz and now - last_control >= self.track_config.control_interval:
                    sx, sy, sz, energy = sound_xyz
                    pan, tilt, zoom = self.controller.compute(sx, sy, energy)
                    if pan or tilt or zoom:
                        self.backend.move_ptz(pan_speed=pan, tilt_speed=tilt, zoom_speed=zoom)
                        moving = True
                        print(
                            f"跟踪 pan={pan:+.2f} tilt={tilt:+.2f} zoom={zoom:+.2f} "
                            f"| 声源 x={sx:+.2f} y={sy:+.2f} z={sz:+.2f} E={energy:.2f}"
                        )
                    elif moving:
                        self.backend.stop_ptz()
                        moving = False
                    last_control = now

                if preview is not None and not preview.is_running():
                    break
                if preview is None:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n已停止跟踪")
            raise
        finally:
            if moving:
                self.backend.stop_ptz()
            sound.stop()
            if preview is not None and owns_preview:
                preview.stop()
