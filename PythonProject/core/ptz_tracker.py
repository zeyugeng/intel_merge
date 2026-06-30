"""Sound-source driven PTZ tracking."""

from __future__ import annotations

import glob
import importlib.util
import math
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
from .bird_sound_gate import BirdSoundGate
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
        _draw_sound_target(frame, sx, sy)

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


def _draw_sound_target(frame, sx: float, sy: float) -> None:
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    target = (
        int(center[0] + max(-1.0, min(1.0, sx)) * width * 0.45),
        int(center[1] - max(-1.0, min(1.0, sy)) * height * 0.45),
    )

    cv2.circle(frame, center, 8, (255, 255, 255), 2)
    cv2.circle(frame, target, 12, (0, 255, 255), 2)
    cv2.arrowedLine(frame, center, target, (0, 255, 255), 3, tipLength=0.18)


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
                    _draw_sound_target(frame_show, sx, sy)

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
                if self.camera.cap is not None:
                    self.camera.cap.release()
                    self.camera.cap = None
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


def sound_xyz_to_angles(x: float, y: float, z: float) -> Tuple[float, float]:
    """intelcup/main.py status_1: 3D 声源坐标 → pan/tilt 角度（度）。"""
    pan_angle = math.degrees(math.atan2(x, z))
    horizontal_distance = math.sqrt(x * x + z * z)
    tilt_angle = math.degrees(math.atan2(y, horizontal_distance))
    return pan_angle, tilt_angle


class SoundToVelocityController:
    """Maps sound coordinates into PTZ continuous-move speeds."""

    def __init__(self, sound_config: SoundConfig, track_config: PTZTrackConfig):
        self.sound_config = sound_config
        self.track_config = track_config

    def compute(self, sound_x: float, sound_y: float, energy: float) -> Tuple[float, float, float]:
        if energy <= self.track_config.activity_threshold:
            return 0.0, 0.0, 0.0

        pan = 0.0
        tilt = 0.0
        if abs(sound_x) > self.track_config.deadzone:
            pan_sign = -1.0 if self.track_config.invert_pan else 1.0
            pan = pan_sign * self.track_config.kp_pan * sound_x
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
        headless: bool = False,
    ):
        self.backend = backend
        self.sound_config = sound_config or SoundConfig()
        self.track_config = track_config or PTZTrackConfig()
        self.controller = SoundToVelocityController(self.sound_config, self.track_config)
        self.preview = preview
        self.headless = headless

    def _move_absolute(self, pan_angle: float, tilt_angle: float) -> bool:
        move_angle = getattr(self.backend, "move_angle", None)
        if move_angle is None:
            print("当前云台后端不支持绝对角度控制，请使用 --ptz-backend serial 或 --tracking-mode velocity")
            return False
        move_angle(pan_angle, tilt_angle, self.track_config.move_time_ms)
        return True

    def _run_absolute(self, sound: SoundSourceClient, preview: Optional[CameraPreviewThread]) -> bool:
        """intelcup/main.py status_1: 检测到声源后转到绝对角度，间隔 trigger_interval 再触发。"""
        moving = False
        threshold = self.track_config.activity_threshold

        while True:
            valid, sound_xyz = sound.parse_latest()
            if preview is not None:
                preview.update_sound(sound_xyz if valid else None)

            if not valid or sound_xyz is None:
                if preview is not None and not preview.is_running():
                    return moving
                time.sleep(0.02)
                continue

            sx, sy, sz, energy = sound_xyz
            if energy <= threshold:
                if preview is not None and not preview.is_running():
                    return moving
                time.sleep(0.02)
                continue

            pan_angle, tilt_angle = sound_xyz_to_angles(sx, sy, sz)
            if self.track_config.invert_pan:
                pan_angle = -pan_angle
            print(
                f"检测到声源: activity={energy:.3f}, "
                f"x={sx:.3f}, y={sy:.3f}, z={sz:.3f}"
            )
            print(f"云台转向: pan={pan_angle:.2f}, tilt={tilt_angle:.2f}")

            if self._move_absolute(pan_angle, tilt_angle):
                moving = True

            if preview is not None and not preview.is_running():
                return moving

            time.sleep(self.track_config.trigger_interval)

    def _run_velocity(self, sound: SoundSourceClient, preview: Optional[CameraPreviewThread]) -> bool:
        last_control = 0.0
        moving = False

        while True:
            now = time.monotonic()
            valid, sound_xyz = sound.parse_latest()
            if valid and sound_xyz:
                if preview is not None:
                    preview.update_sound(sound_xyz)

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
                return moving
            time.sleep(0.02)

    def run(self) -> None:
        sound = SoundSourceClient(self.sound_config)
        sound.start()

        preview = self.preview
        owns_preview = False
        if self.track_config.show_preview and preview is None:
            preview = CameraPreviewThread()
            owns_preview = preview.start()

        mode = self.track_config.tracking_mode
        if not self.headless:
            if mode == "absolute":
                print("声源跟踪已启动（absolute / intelcup/main.py）：对着麦克风发声，云台将转向声源方向")
            else:
                print("声源跟踪已启动（velocity）：对着麦克风发声，云台将连续跟随声源")
            if preview is not None:
                print("按 q 退出预览窗口")
            else:
                print("无预览时 Ctrl+C 退出")
        elif preview is None:
            print("声源云台后台运行中（fusion 窗口按 q 退出）")

        moving = False
        try:
            if mode == "absolute":
                moving = self._run_absolute(sound, preview)
            else:
                moving = self._run_velocity(sound, preview)
        except KeyboardInterrupt:
            print("\n已停止跟踪")
            raise
        finally:
            if moving and mode == "velocity":
                self.backend.stop_ptz()
            sound.stop()
            if preview is not None and owns_preview:
                preview.stop()


class BirdSoundPTZTracker(SoundPTZTracker):
    """仅在 BirdNET 近期确认鸟声时，按 ODAS 声源坐标驱动云台。"""

    def __init__(
        self,
        backend: PTZBackend,
        bird_gate: BirdSoundGate,
        sound_config: Optional[SoundConfig] = None,
        track_config: Optional[PTZTrackConfig] = None,
        preview: Optional[CameraPreviewThread] = None,
        headless: bool = False,
    ):
        super().__init__(backend, sound_config, track_config, preview, headless)
        self.bird_gate = bird_gate

    def _run_absolute(self, sound: SoundSourceClient, preview: Optional[CameraPreviewThread]) -> bool:
        moving = False
        last_idle_log = 0.0
        last_target: Optional[tuple[float, float]] = None

        while True:
            if preview is not None:
                valid, live_xyz = sound.parse_latest()
                preview.update_sound(live_xyz if valid else self.bird_gate.get_sound_xyz())

            if not self.bird_gate.is_active():
                now = time.monotonic()
                if now - last_idle_log >= 12.0:
                    print("[云台] 等待 BirdNET 确认鸟声…（非鸟声/最大声源不会驱动云台）")
                    last_idle_log = now
                last_target = None
                if preview is not None and not preview.is_running():
                    return moving
                time.sleep(0.05)
                continue

            sound_xyz = self.bird_gate.get_sound_xyz()
            if sound_xyz is None:
                if preview is not None and not preview.is_running():
                    return moving
                time.sleep(0.02)
                continue

            sx, sy, sz, energy = sound_xyz
            _, species, conf, ch, confirmed = self.bird_gate.status()
            pan_angle, tilt_angle = sound_xyz_to_angles(sx, sy, sz)
            if self.track_config.invert_pan:
                pan_angle = -pan_angle

            target_key = (round(pan_angle, 1), round(tilt_angle, 1))
            if target_key == last_target:
                if preview is not None and not preview.is_running():
                    return moving
                time.sleep(0.1)
                continue

            label = species if confirmed else f"{species}*"
            print(
                f"鸟声跟踪: {label} ({conf:.2f}) SSS通道{ch}, "
                f"x={sx:.3f}, y={sy:.3f}, z={sz:.3f}, E={energy:.3f}"
            )
            print(f"云台转向: pan={pan_angle:.2f}, tilt={tilt_angle:.2f}")

            if self._move_absolute(pan_angle, tilt_angle):
                moving = True
                last_target = target_key

            if preview is not None and not preview.is_running():
                return moving

            time.sleep(self.track_config.trigger_interval)

    def _run_velocity(self, sound: SoundSourceClient, preview: Optional[CameraPreviewThread]) -> bool:
        last_control = 0.0
        moving = False
        last_idle_log = 0.0

        while True:
            now = time.monotonic()
            if preview is not None:
                valid, live_xyz = sound.parse_latest()
                preview.update_sound(live_xyz if valid else self.bird_gate.get_sound_xyz())

            if not self.bird_gate.is_active():
                if now - last_idle_log >= 12.0:
                    print("[云台] 等待 BirdNET 确认鸟声…")
                    last_idle_log = now
                if moving:
                    self.backend.stop_ptz()
                    moving = False
                if preview is not None and not preview.is_running():
                    return moving
                time.sleep(0.05)
                continue

            sound_xyz = self.bird_gate.get_sound_xyz()
            if sound_xyz and now - last_control >= self.track_config.control_interval:
                sx, sy, sz, energy = sound_xyz
                pan, tilt, zoom = self.controller.compute(sx, sy, energy)
                _, species, conf, ch, _confirmed = self.bird_gate.status()
                if pan or tilt or zoom:
                    self.backend.move_ptz(pan_speed=pan, tilt_speed=tilt, zoom_speed=zoom)
                    moving = True
                    print(
                        f"鸟声跟踪 {species} ch{ch}: pan={pan:+.2f} tilt={tilt:+.2f} "
                        f"| x={sx:+.2f} y={sy:+.2f} E={energy:.2f}"
                    )
                elif moving:
                    self.backend.stop_ptz()
                    moving = False
                last_control = now

            if preview is not None and not preview.is_running():
                return moving
            time.sleep(0.02)

    def run(self) -> None:
        sound = SoundSourceClient(self.sound_config)
        sound.start()

        preview = self.preview
        owns_preview = False
        if self.track_config.show_preview and preview is None:
            preview = CameraPreviewThread()
            owns_preview = preview.start()

        mode = self.track_config.tracking_mode
        if not self.headless:
            print("鸟声跟踪已启动：BirdNET 确认鸟叫后云台转向声源方向")
            if preview is not None:
                print("按 q 退出预览窗口")
            else:
                print("无预览时 Ctrl+C 退出")
        elif preview is None:
            print("鸟声云台后台运行中（BirdNET 鸟声通道 → 锁定声源方向）")

        moving = False
        try:
            if mode == "absolute":
                moving = self._run_absolute(sound, preview)
            else:
                moving = self._run_velocity(sound, preview)
        except KeyboardInterrupt:
            print("\n已停止跟踪")
            raise
        finally:
            if moving and mode == "velocity":
                self.backend.stop_ptz()
            sound.stop()
            if preview is not None and owns_preview:
                preview.stop()
