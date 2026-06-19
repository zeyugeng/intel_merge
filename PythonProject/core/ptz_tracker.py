"""Sound-source driven PTZ tracking."""

from __future__ import annotations

import time
from typing import Optional, Protocol, Tuple

import cv2

from .config import PTZTrackConfig, SoundConfig
from .sound_client import SoundSourceClient


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

    def __init__(self, backend: PTZBackend, window_name: str = "PTZ Sound Tracking"):
        self.backend = backend
        self.window_name = window_name
        self.cap = None

    def open(self) -> bool:
        if not self.backend.stream_uri:
            self.backend.get_stream_uri()
        self.cap = cv2.VideoCapture(self.backend.stream_uri, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"无法打开 RTSP，仅声源跟踪: {self.backend.stream_uri}")
            self.cap = None
            return False
        return True

    def show(self, sound_xyz: Optional[Tuple[float, float, float, float]]) -> bool:
        if self.cap is None:
            return True

        ret, frame = self.cap.read()
        if not ret:
            print("RTSP 中断")
            return False

        if sound_xyz:
            sx, sy, sz, energy = sound_xyz
            text = f"x={sx:+.2f} y={sy:+.2f} z={sz:+.2f} E={energy:.2f}"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        cv2.imshow(self.window_name, frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
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
    ):
        self.backend = backend
        self.sound_config = sound_config or SoundConfig()
        self.track_config = track_config or PTZTrackConfig()
        self.controller = SoundToVelocityController(self.sound_config, self.track_config)

    def run(self) -> None:
        sound = SoundSourceClient(self.sound_config)
        sound.start()

        preview = None
        if self.track_config.show_preview:
            preview = RTSPPreview(self.backend)
            preview.open()

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

                if preview is not None:
                    if not preview.show(latest_sound):
                        break
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n已停止跟踪")
            raise
        finally:
            if moving:
                self.backend.stop_ptz()
            sound.stop()
            if preview is not None:
                preview.close()
