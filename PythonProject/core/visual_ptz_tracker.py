"""intelcup/main.py status_2：USB 摄像头 + YOLO + 增量角度云台跟踪。"""

from __future__ import annotations

import time
from typing import Optional, Protocol, Tuple

import cv2

from .config import VisualConfig, VisualPTZTrackConfig
from .usb_camera import default_usb_camera, USBCamera
from .visual_detector import VisualDetector


class AnglePTZBackend(Protocol):
    def move_angle(self, pan_angle: float, tilt_angle: float, move_time_ms: int | None = None) -> None: ...

    def get_current_angle(self) -> Tuple[float, float]: ...


class VisualPTZTracker:
    """YOLO 检测目标，使画面中心对准目标（main.py status_2）。"""

    def __init__(
        self,
        backend: AnglePTZBackend,
        visual_config: Optional[VisualConfig] = None,
        track_config: Optional[VisualPTZTrackConfig] = None,
        camera: Optional[USBCamera] = None,
    ):
        self.backend = backend
        self.visual_config = visual_config or VisualConfig(conf=0.3)
        self.track_config = track_config or VisualPTZTrackConfig()
        self.camera = camera or default_usb_camera()
        self.visual: Optional[VisualDetector] = None

    def _target_labels(self) -> str:
        if self.visual is None:
            ids = self.visual_config.target_classes or ()
            return ",".join(str(i) for i in ids) if ids else "all"
        names = self.visual.model.names
        ids = self.visual.config.target_classes or ()
        if not ids:
            return "all"
        return ",".join(str(names.get(i, i)) for i in ids)

    def _prepare_window(self) -> None:
        window = self.track_config.window_name
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, self.track_config.preview_width, 720)
        try:
            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    def _show_frame(self, frame, status: str = "") -> bool:
        h, w = frame.shape[:2]
        preview_w = self.track_config.preview_width
        scale = preview_w / w
        preview = cv2.resize(frame, (preview_w, int(h * scale)))
        if status:
            cv2.putText(
                preview,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        cv2.imshow(self.track_config.window_name, preview)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def run(self) -> None:
        if not self.camera.open():
            print("摄像头打开失败")
            return

        cap = self.camera.cap
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._prepare_window()

        print("视觉云台跟踪已启动（intelcup/main.py status_2），按 q 退出")
        print(f"摄像头: {real_w}x{real_h} | conf>={self.visual_config.conf}")
        print(f"若看不到窗口，请 Alt+Tab 切换到「{self.track_config.window_name}」")

        warmup = self.camera.read()
        if warmup is not None:
            if not self._show_frame(warmup, "正在打开摄像头..."):
                self.camera.release()
                return

        print("正在加载 YOLO 模型（首次可能需数秒）...")
        self.visual = VisualDetector(self.visual_config)
        labels = self._target_labels()
        print(f"检测类别: {labels}")

        if warmup is not None:
            if not self._show_frame(warmup, f"YOLO 就绪 | 检测 {labels}"):
                self.camera.release()
                return

        last_ptz_time = 0.0
        cfg = self.track_config
        frame_idx = 0
        null_frames = 0

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    null_frames += 1
                    if null_frames == 50:
                        print("警告: 连续无法读取摄像头画面，检查 USB 摄像头是否被占用")
                    time.sleep(0.02)
                    continue
                null_frames = 0
                frame_idx += 1

                h, w = frame.shape[:2]
                detections = self.visual.detect(frame, w, h)

                if not detections:
                    status = f"未检测到 {labels} (帧 {frame_idx})"
                    if frame_idx % 30 == 1:
                        print(status)
                    if not self._show_frame(frame, status):
                        break
                    continue

                best = max(detections, key=lambda d: d["conf"])
                x1, y1, x2, y2 = best["box"]
                conf = best["conf"]
                bird_cx = (x1 + x2) / 2
                bird_cy = (y1 + y2) / 2
                frame_cx = w / 2
                frame_cy = h / 2

                error_x = (bird_cx - frame_cx) / frame_cx
                error_y = (bird_cy - frame_cy) / frame_cy

                pan_angle, tilt_angle = self.backend.get_current_angle()
                delta_pan = 0.0
                delta_tilt = 0.0

                pan_dir = -1.0 if cfg.invert_pan else 1.0
                tilt_dir = -1.0 if cfg.invert_tilt else 1.0

                if abs(error_x) > cfg.dead_zone:
                    delta_pan = pan_dir * error_x * cfg.pan_k
                if abs(error_y) > cfg.dead_zone:
                    delta_tilt = tilt_dir * error_y * cfg.tilt_k

                delta_pan = max(-cfg.max_step, min(cfg.max_step, delta_pan))
                delta_tilt = max(-cfg.max_step, min(cfg.max_step, delta_tilt))

                target_pan = max(-90.0, min(90.0, pan_angle + delta_pan))
                target_tilt = max(-90.0, min(90.0, tilt_angle + delta_tilt))

                now = time.monotonic()
                need_move = abs(error_x) > cfg.dead_zone or abs(error_y) > cfg.dead_zone
                if need_move and now - last_ptz_time >= cfg.ptz_interval:
                    print(
                        f"目标位置: cx={bird_cx:.1f}, cy={bird_cy:.1f}, conf={conf:.2f}, "
                        f"error_x={error_x:.3f}, error_y={error_y:.3f}"
                    )
                    print(f"云台跟踪: pan={target_pan:.2f}, tilt={target_tilt:.2f}")
                    self.backend.move_angle(target_pan, target_tilt, cfg.move_time_ms)
                    last_ptz_time = now

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{best['label']} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(frame, (int(bird_cx), int(bird_cy)), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(frame_cx), int(frame_cy)), 5, (255, 0, 0), -1)

                status = f"跟踪 {best['label']} conf={conf:.2f}"
                if not self._show_frame(frame, status):
                    break
        finally:
            self.camera.release()
            try:
                cv2.destroyWindow(self.track_config.window_name)
            except cv2.error:
                pass
