"""声源转向后的视觉找鸟：YOLO 检测 + 未命中则变焦继续搜。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .bird_sound_gate import BirdSoundGate
from .config import VisualSearchConfig
from .usb_camera import USBCamera
from .visual_detector import VisualDetector
from .visual_ptz_tracker import VisualPTZStepper


@dataclass
class ZoomTransform:
    """将检测框从当前视场映射回原始全画幅坐标。"""

    x0: int = 0
    y0: int = 0
    crop_w: int = 0
    crop_h: int = 0
    full_w: int = 0
    full_h: int = 0
    hardware_zoom: int = 0
    digital_factor: float = 1.0

    @property
    def is_identity(self) -> bool:
        return (
            self.digital_factor <= 1.001
            and self.x0 == 0
            and self.y0 == 0
            and self.crop_w == self.full_w
            and self.crop_h == self.full_h
        )


def apply_digital_zoom(frame: np.ndarray, factor: float) -> Tuple[np.ndarray, ZoomTransform]:
    h, w = frame.shape[:2]
    if factor <= 1.001:
        return frame, ZoomTransform(0, 0, w, h, w, h, digital_factor=1.0)

    crop_w = max(32, int(w / factor))
    crop_h = max(32, int(h / factor))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    zoomed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    transform = ZoomTransform(x0, y0, crop_w, crop_h, w, h, digital_factor=factor)
    return zoomed, transform


def map_detections_to_full_frame(
    detections: List[dict],
    transform: ZoomTransform,
) -> List[dict]:
    if transform.is_identity or not detections:
        return detections

    w, h = transform.full_w, transform.full_h
    sx = transform.crop_w / w
    sy = transform.crop_h / h
    mapped: List[dict] = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        ox1 = int(transform.x0 + x1 * sx)
        oy1 = int(transform.y0 + y1 * sy)
        ox2 = int(transform.x0 + x2 * sx)
        oy2 = int(transform.y0 + y2 * sy)
        center_x = (ox1 + ox2) // 2
        center_y = (oy1 + oy2) // 2
        mapped.append(
            {
                **det,
                "box": [ox1, oy1, ox2, oy2],
                "center_2d": (center_x, center_y),
                "center_3d": (
                    VisualDetector.normalize_x(center_x, w),
                    (center_y / h) * 2 - 1.0,
                ),
            }
        )
    return mapped


class VisualBirdSearcher:
    """
    麦克风阵列锁定鸟声方向后，在当前画面用 YOLO 找鸟；
    若未检出则逐步变焦（硬件优先，否则中心裁剪数字变焦）继续识别。
    """

    def __init__(
        self,
        visual: VisualDetector,
        camera: USBCamera,
        bird_gate: Optional[BirdSoundGate] = None,
        ptz_stepper: Optional[VisualPTZStepper] = None,
        config: Optional[VisualSearchConfig] = None,
    ):
        self.visual = visual
        self.camera = camera
        self.bird_gate = bird_gate
        self.ptz_stepper = ptz_stepper
        self.config = config or VisualSearchConfig()
        self._hardware_zoom = 0
        self._digital_factor = 1.0
        self._no_detect_streak = 0
        self._last_zoom_time = 0.0
        self._settle_left = 0
        self._last_status = "等待鸟声方向"
        self._hardware_zoom_ok: Optional[bool] = None

    @property
    def status_text(self) -> str:
        return self._last_status

    def _gate_active(self) -> bool:
        if self.bird_gate is None:
            return not self.config.require_sound_gate
        if not self.config.require_sound_gate:
            return True
        return self.bird_gate.is_active()

    def _try_hardware_zoom(self, level: int) -> bool:
        ok = self.camera.set_zoom(level)
        if self._hardware_zoom_ok is None:
            self._hardware_zoom_ok = ok
            if ok:
                print("[视觉找鸟] 使用摄像头硬件变焦")
            else:
                print("[视觉找鸟] 硬件变焦不可用，使用中心裁剪数字变焦")
        return ok

    def reset_zoom(self) -> None:
        self._hardware_zoom = 0
        self._digital_factor = 1.0
        self._no_detect_streak = 0
        self._settle_left = 0
        self._try_hardware_zoom(0)

    def _prepare_view(self, frame: np.ndarray) -> Tuple[np.ndarray, ZoomTransform]:
        h, w = frame.shape[:2]
        if self._hardware_zoom > 0 and self._hardware_zoom_ok is not False:
            return frame, ZoomTransform(0, 0, w, h, w, h, hardware_zoom=self._hardware_zoom)

        view, transform = apply_digital_zoom(frame, self._digital_factor)
        transform.hardware_zoom = self._hardware_zoom
        return view, transform

    def _escalate_zoom(self) -> bool:
        cfg = self.config
        now = time.monotonic()
        if now - self._last_zoom_time < cfg.zoom_interval_sec:
            return False

        if self._hardware_zoom_ok is not False:
            next_hw = self._hardware_zoom + cfg.hardware_zoom_step
            if next_hw <= cfg.max_hardware_zoom:
                if self._try_hardware_zoom(next_hw):
                    self._hardware_zoom = next_hw
                    self._last_zoom_time = now
                    self._settle_left = cfg.settle_frames_after_zoom
                    self._no_detect_streak = 0
                    print(f"[视觉找鸟] 硬件变焦 → {next_hw}")
                    return True

        next_digital = self._digital_factor * cfg.digital_zoom_step
        if next_digital <= cfg.max_digital_zoom + 1e-6:
            self._digital_factor = min(next_digital, cfg.max_digital_zoom)
            self._last_zoom_time = now
            self._settle_left = cfg.settle_frames_after_zoom
            self._no_detect_streak = 0
            print(f"[视觉找鸟] 数字变焦 → {self._digital_factor:.2f}x")
            return True

        print("[视觉找鸟] 已达最大变焦，仍未检出鸟")
        return False

    def process_frame(self, frame: np.ndarray) -> Tuple[List[dict], str]:
        """读一帧、检测、按需变焦；返回全画幅坐标下的检测列表与状态文案。"""
        cfg = self.config
        if not cfg.enabled:
            h, w = frame.shape[:2]
            detections = self.visual.detect(frame, w, h)
            self._last_status = "视觉搜索已关闭"
            return detections, self._last_status

        gate_on = self._gate_active()
        if not gate_on:
            if cfg.reset_zoom_when_idle and (
                self._hardware_zoom > 0 or self._digital_factor > 1.001
            ):
                self.reset_zoom()
            self._last_status = "等待鸟声方向（麦克风阵列）"
            h, w = frame.shape[:2]
            return [], self._last_status

        if self._settle_left > 0:
            self._settle_left -= 1
            self._last_status = f"变焦稳定中… ({self._settle_left})"
            return [], self._last_status

        view, transform = self._prepare_view(frame)
        h, w = view.shape[:2]
        detections = self.visual.detect(view, w, h)
        detections = map_detections_to_full_frame(detections, transform)
        full_h, full_w = frame.shape[:2]

        if detections:
            self._no_detect_streak = 0
            best = max(detections, key=lambda d: d["conf"])
            if self.ptz_stepper is not None:
                self.ptz_stepper.step_toward_detection(best, full_w, full_h)
            species = ""
            if self.bird_gate is not None:
                _, species, conf, ch, confirmed = self.bird_gate.status()
                tag = "已确认" if confirmed else "预转向"
                self._last_status = (
                    f"视觉锁定 {best['label']} conf={best['conf']:.2f} "
                    f"| 声源{tag} ch{ch} {species}"
                )
            else:
                self._last_status = f"视觉锁定 {best['label']} conf={best['conf']:.2f}"
            return detections, self._last_status

        self._no_detect_streak += 1
        zoom_hint = ""
        if self._hardware_zoom > 0:
            zoom_hint = f" hw_zoom={self._hardware_zoom}"
        elif self._digital_factor > 1.001:
            zoom_hint = f" dig={self._digital_factor:.2f}x"

        if self._no_detect_streak >= cfg.no_detect_frames_before_zoom:
            if self._escalate_zoom():
                self._last_status = f"未检出鸟，放大继续搜…{zoom_hint}"
            else:
                self._last_status = f"未检出鸟（已达最大变焦）{zoom_hint}"
        else:
            remain = cfg.no_detect_frames_before_zoom - self._no_detect_streak
            self._last_status = f"声源已对准，YOLO 搜索中… ({remain} 帧后变焦){zoom_hint}"

        return [], self._last_status
