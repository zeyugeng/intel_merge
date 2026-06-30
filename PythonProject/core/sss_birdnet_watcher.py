"""Poll ODAS SSS PCM: fast provisional PTZ, async BirdNET confirm."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Optional

from .bird_sound_gate import BirdSoundGate
from .birdnet_infer import bird_species_only, is_non_bird_species, predict_audio, summarize_predictions
from .config import SSSConfig, SoundConfig
from .sound_client import SoundSourceClient
from .sss_reader import clip_rms, normalize_for_birdnet, read_growing_pcm_tail, write_wav_clip


class SSSBirdnetWatcher:
    """
    Read ODAS SSS separated.raw: on active channel clip, aim PTZ immediately,
    then run BirdNET in background to confirm or cancel.
    """

    def __init__(
        self,
        sss_config: Optional[SSSConfig] = None,
        sound_config: Optional[SoundConfig] = None,
        bird_gate: Optional[BirdSoundGate] = None,
    ):
        self.sss_config = sss_config or SSSConfig()
        self.sound_config = sound_config or SoundConfig()
        self.bird_gate = bird_gate
        self.sound = SoundSourceClient(self.sound_config)
        self._last_run = 0.0
        self._clip_index = 0
        self._infer_busy = False

    def _raw_path(self) -> Path:
        if self.sss_config.use_postfiltered:
            return self.sss_config.postfiltered_path
        return self.sss_config.separated_path

    def _read_clip(self) -> Optional[tuple]:
        result = read_growing_pcm_tail(
            self._raw_path(),
            sample_rate=self.sss_config.sample_rate,
            hop_size=self.sss_config.hop_size,
            n_channels=self.sss_config.n_channels,
            duration_sec=self.sss_config.clip_seconds,
            n_bits=self.sss_config.n_bits,
        )
        if result is None:
            return None
        audio, channel = result
        if clip_rms(audio) < self.sss_config.min_clip_rms:
            return None
        return audio, channel

    def _sound_xyz_for_sss_channel(self, channel: int) -> Optional[tuple]:
        valid, xyz = self.sound.source_xyz_for_channel(channel)
        if valid and xyz:
            return xyz
        valid, xyz = self.sound.parse_latest()
        return xyz if valid else None

    def _confirm_birdnet_async(
        self,
        wav_path: Path,
        sss_channel: int,
        bird_xyz: Optional[tuple],
        source_rms: float,
        gain: float,
        infer_sec_hint: float,
    ) -> None:
        cfg = self.sss_config
        try:
            t0 = time.perf_counter()
            predictions = predict_audio(
                wav_path,
                confidence_threshold=min(0.05, cfg.birdnet_confidence),
            )
            infer_sec = time.perf_counter() - t0
            all_rows = summarize_predictions(predictions, locale=cfg.birdnet_locale)
            bird_rows = [
                row
                for row in bird_species_only(all_rows)
                if float(row["confidence"]) >= cfg.birdnet_confidence
            ]
            non_bird_rows = [
                row
                for row in all_rows
                if float(row["confidence"]) >= cfg.birdnet_confidence
                and is_non_bird_species(str(row.get("species_raw", row["species"])))
            ]

            print(
                f"[BirdNET] 通道 {sss_channel} | {wav_path.name} "
                f"({cfg.clip_seconds}s, 推理 {infer_sec:.2f}s)"
            )

            if bird_rows:
                species = str(bird_rows[0]["species"])
                conf = float(bird_rows[0]["confidence"])
                print(f"[BirdNET] 鸟类: {species} ({conf:.3f})")
                for row in bird_rows[1:3]:
                    print(f"         备选: {row['species']} ({row['confidence']:.3f})")
                if self.bird_gate is not None:
                    ttl = max(cfg.birdnet_cooldown * 2, 3.0)
                    self.bird_gate.mark_bird(species, conf, ttl, bird_xyz, sss_channel)
            elif non_bird_rows:
                print(
                    f"[BirdNET] 非鸟类声: {non_bird_rows[0]['species']} "
                    f"({non_bird_rows[0]['confidence']:.3f})，取消预转向"
                )
                if self.bird_gate is not None:
                    self.bird_gate.clear()
            else:
                print(
                    f"[BirdNET] 未达阈值 (>{cfg.birdnet_confidence}) | "
                    f"RMS={source_rms:.4f} 增益×{gain:.1f}"
                )
        finally:
            self._infer_busy = False

    def run_once(self) -> Optional[dict[str, Any]]:
        clip = self._read_clip()
        if clip is None:
            return None
        audio, sss_channel = clip

        bird_xyz = self._sound_xyz_for_sss_channel(sss_channel)
        if bird_xyz is None:
            return None
        if bird_xyz[3] < self.sss_config.trigger_energy:
            return None

        self._clip_index += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = self.sss_config.clips_dir / f"sss_{stamp}_{self._clip_index}.wav"

        cfg = self.sss_config
        source_rms = clip_rms(audio)

        if self.bird_gate is not None:
            self.bird_gate.mark_provisional(
                bird_xyz,
                sss_channel,
                ttl_sec=max(cfg.birdnet_cooldown * 2, 3.0),
            )
            sx, sy, sz, energy = bird_xyz
            print(
                f"[云台] 快速预转向 通道{sss_channel} "
                f"x={sx:+.2f} y={sy:+.2f} z={sz:+.2f} E={energy:.2f} "
                f"(BirdNET 后台确认中)"
            )

        audio, gain = normalize_for_birdnet(
            audio,
            target_rms=cfg.normalize_target_rms,
            max_gain=cfg.normalize_max_gain,
        )
        write_wav_clip(audio, cfg.sample_rate, wav_path)

        if not self._infer_busy:
            self._infer_busy = True
            Thread(
                target=self._confirm_birdnet_async,
                args=(wav_path, sss_channel, bird_xyz, source_rms, gain, 0.0),
                daemon=True,
            ).start()

        return {
            "wav_path": wav_path,
            "sound_xyz": bird_xyz,
            "sss_channel": sss_channel,
            "provisional": True,
            "source_rms": source_rms,
            "gain": gain,
        }

    def run_loop(self) -> None:
        cfg = self.sss_config
        raw_path = self._raw_path()
        print("SSS → BirdNET 监视已启动（快速预转向 + 异步确认）")
        print(f"  读取: {raw_path} ({'postfiltered' if cfg.use_postfiltered else 'separated'})")
        print(
            f"  片段 {cfg.clip_seconds}s | 轮询 {cfg.poll_interval}s | "
            f"冷却 {cfg.birdnet_cooldown}s | 按 SSS 通道方向预转"
        )

        if not raw_path.is_file():
            print(f"  等待文件出现: {raw_path}")

        self.sound.start()
        try:
            while True:
                now = time.monotonic()
                if now - self._last_run >= cfg.birdnet_cooldown:
                    if self.run_once():
                        self._last_run = now
                time.sleep(cfg.poll_interval)
        except KeyboardInterrupt:
            print("\nSSS → BirdNET 监视已停止")
        finally:
            self.sound.stop()
