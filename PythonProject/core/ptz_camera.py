import datetime
from typing import Optional

import cv2

from onvif import ONVIFCamera

from .config import PTZConfig, PTZTrackConfig, SoundConfig
from .paths import CAPTURES_DIR, OUTPUT_DIR
from .ptz_tracker import CameraPreviewThread, MainThreadCameraPreview, SoundPTZTracker
from .sound_client import SoundSourceClient


class PTZCameraController:
    """云台摄像头 ONVIF 控制与 RTSP 预览。"""

    def __init__(self, config: Optional[PTZConfig] = None):
        self.config = config or PTZConfig()
        self.camera = None
        self.ptz_service = None
        self.media_service = None
        self.profile_token = None
        self.stream_uri: Optional[str] = None
        self.connected = False

    def connect(self) -> bool:
        try:
            self.camera = ONVIFCamera(
                self.config.ip,
                self.config.port,
                self.config.user,
                self.config.password,
            )
            self.media_service = self.camera.create_media_service()
            profiles = self.media_service.GetProfiles()
            self.profile_token = profiles[0].token
            self.ptz_service = self.camera.create_ptz_service()
            self.connected = True
            print(f"成功连接云台 {self.config.ip}")
            return True
        except Exception as exc:
            self.connected = False
            print(f"连接云台失败: {exc}")
            return False

    def move_ptz(self, pan_speed: float = 0.0, tilt_speed: float = 0.0, zoom_speed: float = 0.0) -> None:
        if not self.connected or self.ptz_service is None:
            return
        try:
            request = self.ptz_service.create_type("ContinuousMove")
            request.ProfileToken = self.profile_token
            request.Velocity = {
                "PanTilt": {"x": pan_speed, "y": tilt_speed},
                "Zoom": {"x": zoom_speed},
            }
            self.ptz_service.ContinuousMove(request)
        except Exception as exc:
            print(f"云台转动失败: {exc}")

    def stop_ptz(self, stop_zoom: bool = True) -> None:
        if not self.connected or self.ptz_service is None:
            return
        try:
            request = self.ptz_service.create_type("Stop")
            request.ProfileToken = self.profile_token
            request.PanTilt = True
            request.Zoom = stop_zoom
            self.ptz_service.Stop(request)
        except Exception as exc:
            print(f"停止云台失败: {exc}")

    def get_stream_uri(self) -> Optional[str]:
        if not self.connected or self.media_service is None:
            return None
        stream_params = {
            "ProfileToken": self.profile_token,
            "StreamSetup": {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            },
        }
        self.stream_uri = self.media_service.GetStreamUri(stream_params).Uri
        print(f"RTSP 流地址: {self.stream_uri}")
        return self.stream_uri

    def preview_with_sound(
        self,
        sound_config: Optional[SoundConfig] = None,
        save_video: bool = False,
        output_path: Optional[str] = None,
    ) -> None:
        if not self.stream_uri:
            self.get_stream_uri()

        sound = SoundSourceClient(sound_config)
        sound.start()

        cap = cv2.VideoCapture(self.stream_uri, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"无法打开 RTSP: {self.stream_uri}")
            sound.stop()
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

        writer = None
        if save_video:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            target = output_path or str(OUTPUT_DIR / "rtsp_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(target, fourcc, fps, (frame_width, frame_height))

        print("按 q 退出，按 s 保存单帧")
        try:
            while True:
                valid, sound_xyz = sound.parse_latest()
                if valid and sound_xyz:
                    print(
                        f"声源: x={sound_xyz[0]:.2f}, y={sound_xyz[1]:.2f}, "
                        f"z={sound_xyz[2]:.2f}, E={sound_xyz[3]:.2f}"
                    )

                ret, frame = cap.read()
                if not ret:
                    print("RTSP 中断")
                    break

                cv2.imshow("RTSP Stream", frame)
                if writer:
                    writer.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
                    filename = CAPTURES_DIR / f"ptz_frame_{datetime.datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"已保存: {filename}")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            sound.stop()

    def track_with_sound(
        self,
        sound_config: Optional[SoundConfig] = None,
        track_config: Optional[PTZTrackConfig] = None,
        preview: Optional[CameraPreviewThread | MainThreadCameraPreview] = None,
    ) -> None:
        """根据实时声源坐标驱动云台转动，可选 RTSP 预览。"""
        SoundPTZTracker(self, sound_config, track_config, preview=preview).run()
