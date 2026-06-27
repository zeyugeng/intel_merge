"""USB 摄像头 + ODAS 声源 + YOLO 声视融合（无云台转动）。"""

from __future__ import annotations

from typing import Optional

import cv2

from .config import SoundConfig, VisualConfig
from .sound_client import SoundSourceClient
from .usb_camera import default_usb_camera, USBCamera
from .visual_detector import VisualDetector


class USBAudioVisualFusion:
    """intelcup 摄像头 + ODAS 声源高亮匹配目标（对应 run_fusion，镜头不跟）。"""

    def __init__(
        self,
        sound_config: Optional[SoundConfig] = None,
        visual_config: Optional[VisualConfig] = None,
        camera: Optional[USBCamera] = None,
        window_name: str = "Bird Monitor (USB Fusion)",
    ):
        self.camera = camera or default_usb_camera()
        self.sound = SoundSourceClient(sound_config)
        self.visual_config = visual_config or VisualConfig()
        self.visual: Optional[VisualDetector] = None
        self.sound_config = sound_config or SoundConfig()
        self.window_name = window_name
        self.highlight_idx = -1

    def _match_target(self, sound_x: float, sound_e: float, detections: list) -> int:
        if not detections or sound_e <= self.sound_config.energy_threshold:
            return -1

        best_idx = -1
        best_diff = float("inf")
        for idx, det in enumerate(detections):
            diff = abs(sound_x - det["center_3d"][0])
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        return best_idx

    def run(self) -> None:
        if not self.camera.open():
            print("摄像头打开失败")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

        warmup = self.camera.read()
        if warmup is not None:
            cv2.imshow(self.window_name, warmup)
            cv2.waitKey(1)

        print("正在加载 YOLO 模型...")
        self.visual = VisualDetector(self.visual_config)
        self.sound.start()
        print("USB 声视融合已启动，按 q 退出")
        print(f"若看不到窗口，请 Alt+Tab 切换到「{self.window_name}」")

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue

                frame_copy = frame.copy()
                h, w = frame.shape[:2]
                detections = self.visual.detect(frame, w, h)

                has_sound, sound_xyz = self.sound.parse_latest()
                self.highlight_idx = -1
                overlay = ""
                if has_sound and sound_xyz is not None:
                    sound_x, sound_y, sound_z, sound_e = sound_xyz
                    self.highlight_idx = self._match_target(sound_x, sound_e, detections)
                    overlay = (
                        f"sound x={sound_x:+.2f} y={sound_y:+.2f} z={sound_z:+.2f} E={sound_e:.2f}"
                    )

                self.visual.draw(frame_copy, detections, highlight_idx=self.highlight_idx)
                if overlay:
                    cv2.putText(
                        frame_copy,
                        overlay,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow(self.window_name, frame_copy)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.sound.stop()
            self.camera.release()
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
