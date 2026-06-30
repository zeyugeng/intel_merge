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

    @classmethod
    def open_first_available(
        cls,
        preferred_id: Optional[int] = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        max_probe: int = 8,
    ) -> Optional["USBCamera"]:
        """Try preferred /dev/videoN, then scan other indices."""
        candidates: list[int] = []
        if preferred_id is not None:
            candidates.append(preferred_id)
        for cid in range(max_probe):
            if cid not in candidates:
                candidates.append(cid)

        for cid in candidates:
            dev = f"/dev/video{cid}"
            if not os.path.exists(dev):
                continue
            cam = cls(camera_id=cid, width=width, height=height, fps=fps)
            if cam.open():
                print(f"已打开 USB 摄像头: {dev} ({width}x{height})")
                return cam
            print(f"无法打开 {dev}，尝试下一个…")
        return None

    def read(self) -> Optional[Tuple[object, ...]]:
        frame = self._camera.get_frame()
        if frame is None:
            return None
        return frame

    def set_zoom(self, level: int) -> bool:
        """V4L2 硬件变焦；不支持时返回 False。"""
        if hasattr(self._camera, "set_zoom"):
            return bool(self._camera.set_zoom(int(level)))
        return False

    def get_zoom(self) -> int:
        if hasattr(self._camera, "zoom"):
            return int(self._camera.zoom)
        if self._camera.cap is not None:
            import cv2

            return int(self._camera.cap.get(cv2.CAP_PROP_ZOOM))
        return 0

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
    width = int(os.getenv("PTZ_PREVIEW_WIDTH", "1920"))
    height = int(os.getenv("PTZ_PREVIEW_HEIGHT", "1080"))
    env_id = os.getenv("PTZ_PREVIEW_CAMERA_ID")
    preferred = int(env_id) if env_id is not None else None
    cam = USBCamera.open_first_available(preferred_id=preferred, width=width, height=height)
    if cam is not None:
        return cam
    raise RuntimeError(
        "未找到可用 USB 摄像头。请执行 ls /dev/video* 并设置 PTZ_PREVIEW_CAMERA_ID=1"
    )
