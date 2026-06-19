"""Utilities for normalizing sound-source payloads.

ODAS can emit either a raw SSL payload with ``E``, a tracked SST payload with
``activity``, or an already flattened ``{"x": ..., "y": ..., "z": ..., "E": ...}``
object. The rest of the Python code uses the flattened shape only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SoundSource:
    x: float
    y: float
    z: float
    energy: float

    def as_payload(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "E": self.energy}


def normalize_sound_source(payload: Dict[str, Any]) -> Optional[SoundSource]:
    """Return the strongest source from ODAS/raw/flat JSON payloads."""
    src = payload.get("src")
    if isinstance(src, list) and src:
        best = max(src, key=_source_energy)
        return SoundSource(
            x=float(best.get("x", 0)),
            y=float(best.get("y", 0)),
            z=float(best.get("z", 0)),
            energy=_source_energy(best),
        )

    if "x" in payload and ("E" in payload or "activity" in payload):
        return SoundSource(
            x=float(payload.get("x", 0)),
            y=float(payload.get("y", 0)),
            z=float(payload.get("z", 0)),
            energy=_source_energy(payload),
        )

    return None


def _source_energy(source: Dict[str, Any]) -> float:
    """Return ODAS confidence from SSL ``E`` or SST ``activity`` payloads."""
    return float(source.get("E", source.get("activity", 0)))
