from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .config import VisualConfig
from .paths import YOLO_MODEL_PATH


class VisualDetector:
    """YOLO26 视觉检测（默认 COCO bird 类别）。"""

    def __init__(self, config: Optional[VisualConfig] = None):
        self.config = config or VisualConfig()
        model_path = self.config.model_path or str(YOLO_MODEL_PATH)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"未找到模型权重: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()

    @staticmethod
    def normalize_x(pixel_x: float, frame_width: int) -> float:
        pixel_x = max(0.0, min(float(pixel_x), frame_width))
        return (pixel_x / frame_width) * 2 - 1.0

    def detect(self, frame, frame_width: int, frame_height: int) -> List[Dict]:
        results = self.model(
            frame,
            conf=self.config.conf,
            verbose=False,
            device=self.config.device,
        )
        detections: List[Dict] = []
        if results[0].boxes is None:
            return detections

        names = results[0].names
        for box in results[0].boxes:
            cls_id = int(box.cls)
            if self.config.target_classes and cls_id not in self.config.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().reshape(-1))
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "center_2d": (center_x, center_y),
                    "center_3d": (
                        self.normalize_x(center_x, frame_width),
                        (center_y / frame_height) * 2 - 1.0,
                    ),
                    "conf": float(box.conf),
                    "label": names.get(cls_id, str(cls_id)),
                }
            )
        return detections

    def draw(
        self,
        frame,
        detections: List[Dict],
        highlight_idx: int = -1,
        default_label_prefix: str = "Bird",
    ) -> None:
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            if idx == highlight_idx:
                color = (255, 0, 255)
                thickness = 4
                label = f"Target {default_label_prefix} {idx + 1}"
            else:
                color = (0, 255, 0)
                thickness = 2
                label = f"{default_label_prefix} {idx + 1}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
