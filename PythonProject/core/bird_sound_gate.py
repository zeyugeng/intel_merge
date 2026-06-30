"""BirdNET 鸟声门控：先快速预转向，BirdNET 异步确认。"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

SoundXYZ = Tuple[float, float, float, float]


class BirdSoundGate:
    """Provisional aim on SSS channel; confirmed or cleared after BirdNET."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_until = 0.0
        self._species = ""
        self._confidence = 0.0
        self._sound_xyz: Optional[SoundXYZ] = None
        self._sss_channel = -1
        self._confirmed = False

    def mark_provisional(
        self,
        sound_xyz: SoundXYZ,
        sss_channel: int,
        ttl_sec: float = 4.0,
    ) -> None:
        with self._lock:
            self._active_until = time.monotonic() + max(1.0, ttl_sec)
            self._sound_xyz = sound_xyz
            self._sss_channel = sss_channel
            self._species = "鸟声检测中"
            self._confidence = 0.0
            self._confirmed = False

    def mark_bird(
        self,
        species: str,
        confidence: float,
        ttl_sec: float,
        sound_xyz: Optional[SoundXYZ],
        sss_channel: int = -1,
    ) -> None:
        with self._lock:
            self._active_until = time.monotonic() + max(1.0, ttl_sec)
            self._species = species
            self._confidence = confidence
            if sound_xyz is not None:
                self._sound_xyz = sound_xyz
            if sss_channel >= 0:
                self._sss_channel = sss_channel
            self._confirmed = True

    def clear(self) -> None:
        with self._lock:
            self._active_until = 0.0
            self._sound_xyz = None
            self._sss_channel = -1
            self._species = ""
            self._confidence = 0.0
            self._confirmed = False

    def is_active(self) -> bool:
        with self._lock:
            return time.monotonic() < self._active_until and self._sound_xyz is not None

    def is_confirmed(self) -> bool:
        with self._lock:
            return self._confirmed and time.monotonic() < self._active_until

    def get_sound_xyz(self) -> Optional[SoundXYZ]:
        with self._lock:
            if time.monotonic() >= self._active_until:
                return None
            return self._sound_xyz

    def status(self) -> Tuple[bool, str, float, int, bool]:
        with self._lock:
            active = time.monotonic() < self._active_until
            return (
                active and self._sound_xyz is not None,
                self._species,
                self._confidence,
                self._sss_channel,
                self._confirmed,
            )
