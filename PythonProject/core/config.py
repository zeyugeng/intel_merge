from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent


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
    host: str = "127.0.0.1"
    port: int = 5000
    energy_threshold: float = 0.25
    invert_x: bool = True


@dataclass
class ODASConfig:
    host: str = "127.0.0.1"
    odas_port: int = 9001
    python_port: int = 5000
    bin_path: Path = field(default_factory=lambda: REPO_ROOT / "odas" / "build" / "bin" / "odaslive")
    config_path: Path = field(default_factory=lambda: REPO_ROOT / "odas" / "config" / "myArray_fusion.cfg")
    lib_path: Path = field(default_factory=lambda: REPO_ROOT / "odas" / "build" / "lib")
    log_path: Path = field(default_factory=lambda: PROJECT_ROOT / "output" / "odaslive.log")


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


@dataclass
class PTZTrackConfig:
    """声源坐标 → 云台速度映射参数（需在实机上校准方向与增益）。"""

    kp_pan: float = 0.5
    kp_tilt: float = 0.4
    deadzone: float = 0.08
    max_speed: float = 0.6
    invert_y: bool = True
    enable_zoom: bool = False
    kp_zoom: float = 0.15
    zoom_energy: float = 0.35
    control_interval: float = 0.1
    show_preview: bool = True
