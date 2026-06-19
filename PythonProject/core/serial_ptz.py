"""Optional serial PWM pan-tilt backend.

This integrates the calibration from the root-level ``pantilt_control.py`` lab
script into the main package. It is not used by default; the primary runtime
still uses ONVIF PTZ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SerialPTZConfig:
    port: str = "/dev/ttyUSB0"
    baud: int = 115200
    pan_id: int = 1
    tilt_id: int = 2
    pan_n90_pwm: int = 975
    pan_p90_pwm: int = 2270
    tilt_n90_pwm: int = 815
    tilt_p90_pwm: int = 2150
    pan_invert: bool = False
    tilt_invert: bool = False
    default_time_ms: int = 120
    angle_step_per_speed: float = 8.0

    @property
    def pan_0_pwm(self) -> float:
        return (self.pan_n90_pwm + self.pan_p90_pwm) / 2

    @property
    def tilt_0_pwm(self) -> float:
        return (self.tilt_n90_pwm + self.tilt_p90_pwm) / 2


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_to_pwm(angle: float, n90_pwm: int, zero_pwm: float, p90_pwm: int, invert: bool = False) -> int:
    if invert:
        angle = -angle

    angle = clamp(angle, -90.0, 90.0)
    if angle >= 0:
        pwm = zero_pwm + angle / 90.0 * (p90_pwm - zero_pwm)
    else:
        pwm = zero_pwm + angle / 90.0 * (zero_pwm - n90_pwm)
    return int(round(pwm))


def make_cmd(servo_id: int, pwm: int, move_time_ms: int) -> str:
    return f"#{servo_id:03d}P{int(pwm):04d}T{int(move_time_ms)}!"


class SerialPanTiltBackend:
    """Serial-servo backend compatible with PTZ speed-style tracking."""

    def __init__(self, config: SerialPTZConfig | None = None):
        self.config = config or SerialPTZConfig()
        self.ser = None
        self.pan_pwm = int(round(self.config.pan_0_pwm))
        self.tilt_pwm = int(round(self.config.tilt_0_pwm))
        self.pan_angle = 0.0
        self.tilt_angle = 0.0
        self.stream_uri: Optional[str] = None

    def connect(self) -> bool:
        try:
            import serial

            self.ser = serial.Serial(self.config.port, self.config.baud, timeout=1)
            time.sleep(1)
            print(f"成功连接串口云台 {self.config.port}")
            return True
        except Exception as exc:
            print(f"连接串口云台失败: {exc}")
            return False

    def move_angle(self, pan_angle: float, tilt_angle: float, move_time_ms: int | None = None) -> None:
        if self.ser is None:
            raise RuntimeError("串口云台尚未连接")

        move_time = move_time_ms or self.config.default_time_ms
        self.pan_angle = clamp(pan_angle, -90.0, 90.0)
        self.tilt_angle = clamp(tilt_angle, -90.0, 90.0)
        pan_pwm = angle_to_pwm(
            self.pan_angle,
            self.config.pan_n90_pwm,
            self.config.pan_0_pwm,
            self.config.pan_p90_pwm,
            self.config.pan_invert,
        )
        tilt_pwm = angle_to_pwm(
            self.tilt_angle,
            self.config.tilt_n90_pwm,
            self.config.tilt_0_pwm,
            self.config.tilt_p90_pwm,
            self.config.tilt_invert,
        )
        cmd = make_cmd(self.config.pan_id, pan_pwm, move_time) + make_cmd(
            self.config.tilt_id, tilt_pwm, move_time
        )
        print("SEND:", cmd)
        self.ser.write(cmd.encode("ascii"))
        self.pan_pwm = pan_pwm
        self.tilt_pwm = tilt_pwm

    def move_ptz(
        self,
        pan_speed: float = 0.0,
        tilt_speed: float = 0.0,
        zoom_speed: float = 0.0,
    ) -> None:
        """Approximate continuous PTZ speed by stepping absolute servo angles."""
        del zoom_speed
        step = self.config.angle_step_per_speed
        target_pan = self.pan_angle + pan_speed * step
        target_tilt = self.tilt_angle + tilt_speed * step
        self.move_angle(target_pan, target_tilt, self.config.default_time_ms)

    def stop_ptz(self, stop_zoom: bool = True) -> None:
        del stop_zoom

    def get_stream_uri(self) -> Optional[str]:
        return None

    def center(self) -> None:
        self.move_angle(0, 0)

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None
