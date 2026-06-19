import queue
from typing import List, Optional

import cv2

from .camera import RTSPCamera
from .config import CameraConfig, SoundConfig, VisualConfig
from .sound_client import SoundSourceClient
from .visual_detector import VisualDetector


class AudioVisualFusion:
    """声源定位 + 视觉检测融合，匹配声源方向与画面中的鸟类目标。"""

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        sound_config: Optional[SoundConfig] = None,
        visual_config: Optional[VisualConfig] = None,
    ):
        self.camera = RTSPCamera(camera_config)
        self.sound = SoundSourceClient(sound_config)
        self.visual = VisualDetector(visual_config)
        self.sound_config = sound_config or SoundConfig()
        self.highlight_idx = -1

    def _match_target(self, sound_x: float, sound_e: float, detections: List[dict]) -> int:
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

    def run(self, window_name: str = "Bird Monitor (Audio-Visual Fusion)") -> None:
        if not self.camera.connect():
            return

        self.sound.start()
        self.camera.start_capture()

        print("声视融合监测已启动，按 q 退出")
        try:
            while True:
                try:
                    frame = self.camera.read_frame(timeout=0.1)
                except queue.Empty:
                    continue

                frame_copy = frame.copy()
                cfg = self.camera.config
                detections = self.visual.detect(frame, cfg.frame_width, cfg.frame_height)

                has_sound, sound_xyz = self.sound.parse_latest()
                self.highlight_idx = -1
                if has_sound and sound_xyz is not None:
                    sound_x, _, _, sound_e = sound_xyz
                    self.highlight_idx = self._match_target(sound_x, sound_e, detections)

                self.visual.draw(frame_copy, detections, highlight_idx=self.highlight_idx)
                cv2.imshow(window_name, frame_copy)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.camera.stop_capture()
            self.sound.stop()
            cv2.destroyAllWindows()
