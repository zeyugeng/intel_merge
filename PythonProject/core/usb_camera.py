"""USB 摄像头封装（复用 intelcup/Camera.py 实现）。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Optional, Tuple

INTELCUP_CAMERA_PATH = Path(__file__).resolve().parents[2] / "intelcup" / "Camera.py"


def _load_camera_class():
    spec = importlib.util.spec_from_file_location("intelcup_camera", INTELCUP_CAMERA_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Camera


CameraClass = _load_camera_class()


class USBCamera:
    """与 intelcup/main.py 相同的本地 USB 摄像头接口。"""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
    ):
        if CameraClass is None:
            raise RuntimeError(f"无法加载摄像头模块: {INTELCUP_CAMERA_PATH}")
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._camera = CameraClass(camera_id=camera_id)
        self._camera.WIDTH = width
        self._camera.HEIGHT = height
        self._camera.FPS = fps

    def open(self) -> bool:
        return self._camera.get_camera()

    def read(self) -> Optional[Tuple[object, ...]]:
        frame = self._camera.get_frame()
        if frame is None:
            return None
        return frame

    @property
    def cap(self):
        return self._camera.cap

    def release(self, destroy_windows: bool = False) -> None:
        if self._camera.cap is not None:
            self._camera.cap.release()
            self._camera.cap = None
        if destroy_windows:
            import cv2

            cv2.destroyAllWindows()


def default_usb_camera() -> USBCamera:
    camera_id = int(os.getenv("PTZ_PREVIEW_CAMERA_ID", "0"))
    width = int(os.getenv("PTZ_PREVIEW_WIDTH", "1920"))
    height = int(os.getenv("PTZ_PREVIEW_HEIGHT", "1080"))
    return USBCamera(camera_id=camera_id, width=width, height=height)
