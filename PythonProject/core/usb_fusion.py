"""USB 摄像头 + ODAS 声源 + YOLO 声视融合（主流程 fusion 模式）。"""

from __future__ import annotations

import time
from typing import Optional

import cv2

from .config import SoundConfig, VisualConfig, VisualPTZTrackConfig
from .sound_client import SoundSourceClient
from .usb_camera import default_usb_camera, USBCamera
from .visual_detector import VisualDetector
from .visual_ptz_tracker import VisualPTZStepper

# 与 sound / visual 预览一致（ptz_tracker.PREVIEW_W/H）
PREVIEW_W = 1280
PREVIEW_H = 720


def _resize_for_display(frame, preview_w: int = PREVIEW_W, preview_h: int = PREVIEW_H):
    h, w = frame.shape[:2]
    scale = min(preview_w / w, preview_h / h)
    return cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


class USBAudioVisualFusion:
    """USB 摄像头 + YOLO 检测 + ODAS 声源高亮；云台仅在检测到鸟时转向鸟的位置。"""

    def __init__(
        self,
        sound_config: Optional[SoundConfig] = None,
        visual_config: Optional[VisualConfig] = None,
        camera: Optional[USBCamera] = None,
        window_name: str = "Bird Monitor (USB Fusion)",
        ptz_backend=None,
        visual_track_config: Optional[VisualPTZTrackConfig] = None,
    ):
        self.camera = camera
        self.sound = SoundSourceClient(sound_config)
        self.visual_config = visual_config or VisualConfig()
        self.visual: Optional[VisualDetector] = None
        self.sound_config = sound_config or SoundConfig()
        self.window_name = window_name
        self.highlight_idx = -1
        self.ptz_stepper: Optional[VisualPTZStepper] = None
        self._frame_idx = 0
        self._last_no_bird_log = 0.0
        if ptz_backend is not None:
            self.ptz_stepper = VisualPTZStepper(
                ptz_backend,
                visual_track_config or VisualPTZTrackConfig(),
            )

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
        if self.camera is None:
            try:
                self.camera = default_usb_camera()
            except RuntimeError as exc:
                print(f"摄像头打开失败: {exc}")
                raise

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, PREVIEW_W, PREVIEW_H)
        try:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

        warmup = self.camera.read()
        if warmup is not None:
            real_h, real_w = warmup.shape[:2]
            print(f"fusion 预览: 摄像头 {real_w}x{real_h} -> 窗口 {PREVIEW_W}x{PREVIEW_H}")
            cv2.imshow(self.window_name, _resize_for_display(warmup))
            cv2.waitKey(1)

        print("正在加载 YOLO 模型...")
        self.visual = VisualDetector(self.visual_config)
        self.sound.start()
        if self.ptz_stepper is not None:
            print("USB 声视融合已启动：YOLO 画框时云台转向鸟的位置，按 q 退出")
        else:
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
                self._frame_idx += 1

                if self.ptz_stepper is not None and not detections:
                    now = time.monotonic()
                    if self._frame_idx % 60 == 1 and now - self._last_no_bird_log >= 8.0:
                        print(
                            f"[YOLO] 画面未检测到鸟 (帧 {self._frame_idx})，云台等待视觉目标"
                            " — 请将鸟对准镜头或降低 --conf"
                        )
                        self._last_no_bird_log = now
                elif detections and self._frame_idx <= 3:
                    print(f"[YOLO] 检测到 {len(detections)} 个目标，云台可跟踪")

                has_sound, sound_xyz = self.sound.parse_latest()
                self.highlight_idx = -1
                overlay = ""
                if has_sound and sound_xyz is not None:
                    sound_x, sound_y, sound_z, sound_e = sound_xyz
                    self.highlight_idx = self._match_target(sound_x, sound_e, detections)
                    overlay = (
                        f"sound x={sound_x:+.2f} y={sound_y:+.2f} z={sound_z:+.2f} E={sound_e:.2f}"
                    )

                if self.ptz_stepper is not None and detections:
                    track_target = max(detections, key=lambda d: d["conf"])
                    self.ptz_stepper.step_toward_detection(track_target, w, h)

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

                cv2.imshow(self.window_name, _resize_for_display(frame_copy))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.sound.stop()
            self.camera.release()
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
