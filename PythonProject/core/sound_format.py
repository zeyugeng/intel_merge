"""Utilities for normalizing sound-source payloads.

ODAS can emit either a raw payload with a ``src`` array or an already flattened
``{"x": ..., "y": ..., "z": ..., "E": ...}`` object. The rest of the Python
code uses the flattened shape only.
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
        best = max(src, key=lambda item: float(item.get("E", 0)))
        return SoundSource(
            x=float(best.get("x", 0)),
            y=float(best.get("y", 0)),
            z=float(best.get("z", 0)),
            energy=float(best.get("E", 0)),
        )

    if all(key in payload for key in ("x", "E")):
        return SoundSource(
            x=float(payload.get("x", 0)),
            y=float(payload.get("y", 0)),
            z=float(payload.get("z", 0)),
            energy=float(payload.get("E", 0)),
        )

    return None
