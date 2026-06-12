from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraConfig:
    ip: str = "192.168.0.123"
    port: int = 80
    user: str = "admin"
    password: str = "123456"
    rtsp_url: Optional[str] = None
    frame_width: int = 1280
    frame_height: int = 720


@dataclass
class SoundConfig:
    host: str = "192.168.168.128"
    port: int = 5000
    energy_threshold: float = 0.25
    invert_x: bool = True


@dataclass
class VisualConfig:
    model_path: Optional[str] = None
    conf: float = 0.35
    # COCO 预训练模型中 14 = bird；后续可换成鸟类专用权重
    target_classes: tuple = (14,)
    device: str = "cpu"


@dataclass
class PTZConfig:
    ip: str = "192.168.0.2"
    port: int = 80
    user: str = "admin"
    password: str = "123456"
