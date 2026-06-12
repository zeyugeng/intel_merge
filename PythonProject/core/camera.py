import os
import queue
import subprocess
import threading
import time
from typing import Optional

import cv2
from onvif import ONVIFCamera

from .config import CameraConfig


class RTSPCamera:
    """通过 ONVIF 获取 RTSP 地址，并以低延迟方式抓帧。"""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.stream_uri: Optional[str] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.is_running = False
        self._capture_thread: Optional[threading.Thread] = None

    def _ping(self) -> bool:
        param = "-n" if os.name == "nt" else "-c"
        try:
            result = subprocess.run(
                ["ping", param, "2", self.config.ip],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def connect(self) -> bool:
        if not self._ping():
            print(f"摄像头 {self.config.ip} 无法 ping 通")
            return False

        try:
            cam = ONVIFCamera(
                self.config.ip,
                self.config.port,
                self.config.user,
                self.config.password,
            )
            media_service = cam.create_media_service()
            profiles = media_service.GetProfiles()
            if profiles:
                stream_params = {
                    "ProfileToken": profiles[0].token,
                    "StreamSetup": {
                        "Stream": "RTP-Unicast",
                        "Transport": {"Protocol": "RTSP"},
                    },
                }
                self.stream_uri = media_service.GetStreamUri(stream_params).Uri
                print(f"ONVIF 连接成功: {self.stream_uri}")
                return True
        except Exception as exc:
            print(f"ONVIF 获取 RTSP 失败: {exc}")

        if self.config.rtsp_url:
            self.stream_uri = self.config.rtsp_url
            print(f"使用手动 RTSP: {self.stream_uri}")
            return True

        print("无可用的 RTSP 地址")
        return False

    def start_capture(self) -> None:
        if not self.stream_uri:
            raise RuntimeError("请先调用 connect() 获取 RTSP 地址")
        self.is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop_capture(self) -> None:
        self.is_running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2)

    def read_frame(self, timeout: float = 0.1):
        return self.frame_queue.get(timeout=timeout)

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.stream_uri, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 15)

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("RTSP 读帧失败，重试连接...")
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(self.stream_uri, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                continue

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

        cap.release()
