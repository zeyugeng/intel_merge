"""仅视觉检测：RTSP + YOLO26 鸟类检测。"""

import queue
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.camera import RTSPCamera
from core.config import CameraConfig, VisualConfig
from core.visual_detector import VisualDetector


def main():
    camera = RTSPCamera(CameraConfig(frame_width=640, frame_height=480))
    visual = VisualDetector(VisualConfig(conf=0.4))

    if not camera.connect():
        return
    camera.start_capture()

    print("视觉检测已启动，按 q 退出")
    try:
        while True:
            try:
                frame = camera.read_frame(timeout=0.1)
            except queue.Empty:
                continue

            frame_copy = frame.copy()
            cfg = camera.config
            detections = visual.detect(frame, cfg.frame_width, cfg.frame_height)
            visual.draw(frame_copy, detections)
            cv2.imshow("Bird Detection (Visual Only)", frame_copy)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop_capture()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
