"""Read ODAS SSS growing PCM files (separated.raw / postfiltered.raw)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


def _hop_bytes(hop_size: int, n_channels: int, n_bits: int = 16) -> int:
    bytes_per_sample = n_bits // 8
    return hop_size * n_channels * bytes_per_sample


def read_growing_pcm_tail(
    path: Path,
    sample_rate: int,
    hop_size: int,
    n_channels: int,
    duration_sec: float,
    n_bits: int = 16,
) -> Optional[np.ndarray]:
    """
    Read the tail of a growing interleaved PCM file written by ODAS SSS.

    Returns float32 mono audio shaped (n_samples,) or None if not enough data.
    ODAS writes int16 interleaved frames: [ch0 hop][ch1 hop]...[chN hop] per block.
    """
    if not path.is_file():
        return None

    hop_bytes = _hop_bytes(hop_size, n_channels, n_bits)
    bytes_per_sample = n_bits // 8
    want_samples = int(duration_sec * sample_rate)
    want_bytes = max(hop_bytes, want_samples * n_channels * bytes_per_sample)

    size = path.stat().st_size
    aligned_size = (size // hop_bytes) * hop_bytes
    if aligned_size < hop_bytes:
        return None

    read_bytes = min(aligned_size, want_bytes)
    read_bytes = (read_bytes // hop_bytes) * hop_bytes
    start = aligned_size - read_bytes

    with path.open("rb") as fp:
        fp.seek(start)
        raw = fp.read(read_bytes)

    if len(raw) < hop_bytes:
        return None

    pcm = np.frombuffer(raw, dtype=np.int16)
    frames = len(pcm) // n_channels
    if frames == 0:
        return None

    pcm = pcm[: frames * n_channels]
    multichannel = pcm.reshape(frames, n_channels).astype(np.float32) / 32768.0

    channel = _pick_loudest_channel(multichannel)
    mono = multichannel[:, channel]

    target_len = min(len(mono), want_samples)
    if target_len < sample_rate * 0.5:
        return None

    return mono[-target_len:]


def _pick_loudest_channel(multichannel: np.ndarray) -> int:
    rms = np.sqrt(np.mean(multichannel ** 2, axis=0))
    return int(np.argmax(rms))


def normalize_for_birdnet(
    audio: np.ndarray,
    target_peak: float = 0.85,
    target_rms: float = 0.08,
    max_gain: float = 40.0,
) -> tuple[np.ndarray, float]:
    """Boost quiet SSS clips for BirdNET without unnecessary clipping."""
    peak = float(np.max(np.abs(audio)))
    rms = clip_rms(audio)
    if peak <= 1e-6:
        return audio, 1.0

    gain = min(max_gain, target_peak / peak)
    boosted = audio * gain
    boosted_rms = clip_rms(boosted)
    if boosted_rms < target_rms and rms > 1e-6:
        gain = min(max_gain, target_rms / rms)
        boosted = np.clip(audio * gain, -1.0, 1.0)

    return boosted, gain


def write_wav_clip(audio: np.ndarray, sample_rate: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    return path


def convert_raw_to_wav(
    raw_path: Path,
    wav_path: Path,
    sample_rate: int,
    hop_size: int,
    n_channels: int,
    channel: Optional[int] = None,
    n_bits: int = 16,
) -> Path:
    """Convert a complete (or partial) SSS raw file to mono wav for BirdNET."""
    hop_bytes = _hop_bytes(hop_size, n_channels, n_bits)
    size = raw_path.stat().st_size
    aligned_size = (size // hop_bytes) * hop_bytes
    if aligned_size < hop_bytes:
        raise ValueError(f"文件太短或尚未对齐: {raw_path} ({size} bytes)")

    with raw_path.open("rb") as fp:
        raw = fp.read(aligned_size)

    pcm = np.frombuffer(raw, dtype=np.int16)
    frames = len(pcm) // n_channels
    multichannel = pcm[: frames * n_channels].reshape(frames, n_channels).astype(np.float32) / 32768.0

    if channel is None:
        channel = _pick_loudest_channel(multichannel)
    mono = multichannel[:, channel]
    return write_wav_clip(mono, sample_rate, wav_path)


def clip_rms(audio: np.ndarray) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))
