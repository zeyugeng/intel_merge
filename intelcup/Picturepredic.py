from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class PicturePredict:
    def __init__(self, model_path: str = "yolo26n.pt", confidence: float = 0.25):
        self.model_path = model_path
        self.confidence = confidence
        self.model = None

    def load_model(self) -> None:
        if self.model is not None:
            return

        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def predict_bird(self, frame) -> Optional[Detection]:
        if self.model is None:
            self.load_model()

        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}
        if boxes is None:
            return None

        best = None
        best_score = -1.0
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence:
                continue

            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, cls_id)).lower()
            if "bird" not in label and len(names) > 1:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            detection = Detection(x1, y1, x2, y2, confidence, label)
            score = detection.area * detection.confidence
            if score > best_score:
                best = detection
                best_score = score

        return best
