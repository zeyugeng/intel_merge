"""Poll ODAS SSS postfiltered/separated PCM and run BirdNET on active clips."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .birdnet_infer import bird_species_only, is_non_bird_species, predict_audio, summarize_predictions
from .config import SSSConfig, SoundConfig
from .sound_client import SoundSourceClient
from .sss_reader import clip_rms, normalize_for_birdnet, read_growing_pcm_tail, write_wav_clip


class SSSBirdnetWatcher:
    """
    While ODAS is running, read the growing SSS ``.raw`` file and send short
    mono clips to BirdNET when the tracked sound energy is high.

    BirdNET expects mono wav (3 s clips work well). Default input is ODAS SSS
    **separated** output (separated.raw); use postfiltered.raw if configured.
    """

    def __init__(
        self,
        sss_config: Optional[SSSConfig] = None,
        sound_config: Optional[SoundConfig] = None,
    ):
        self.sss_config = sss_config or SSSConfig()
        self.sound_config = sound_config or SoundConfig()
        self.sound = SoundSourceClient(self.sound_config)
        self._last_run = 0.0
        self._clip_index = 0

    def _raw_path(self) -> Path:
        if self.sss_config.use_postfiltered:
            return self.sss_config.postfiltered_path
        return self.sss_config.separated_path

    def _read_clip(self) -> Optional[tuple]:
        audio = read_growing_pcm_tail(
            self._raw_path(),
            sample_rate=self.sss_config.sample_rate,
            hop_size=self.sss_config.hop_size,
            n_channels=self.sss_config.n_channels,
            duration_sec=self.sss_config.clip_seconds,
            n_bits=self.sss_config.n_bits,
        )
        if audio is None:
            return None
        if clip_rms(audio) < self.sss_config.min_clip_rms:
            return None
        return audio

    def run_once(self, sound_xyz: Optional[tuple] = None) -> Optional[dict[str, Any]]:
        audio = self._read_clip()
        if audio is None:
            return None

        self._clip_index += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = self.sss_config.clips_dir / f"sss_{stamp}_{self._clip_index}.wav"

        cfg = self.sss_config
        source_rms = clip_rms(audio)
        audio, gain = normalize_for_birdnet(
            audio,
            target_rms=cfg.normalize_target_rms,
            max_gain=cfg.normalize_max_gain,
        )
        write_wav_clip(audio, cfg.sample_rate, wav_path)

        predictions = predict_audio(
            wav_path,
            confidence_threshold=min(0.05, cfg.birdnet_confidence),
        )
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

        result = {
            "wav_path": wav_path,
            "sound_xyz": sound_xyz,
            "summary": bird_rows,
            "non_bird": non_bird_rows,
            "raw_predictions": predictions,
            "source_rms": source_rms,
            "gain": gain,
        }

        if sound_xyz:
            sx, sy, sz, energy = sound_xyz
            print(
                f"[BirdNET] 声源 x={sx:+.2f} y={sy:+.2f} z={sz:+.2f} E={energy:.2f} "
                f"| 片段 {wav_path.name}"
            )
        else:
            print(f"[BirdNET] 片段 {wav_path.name}")

        if bird_rows:
            print(f"[BirdNET] 鸟类: {bird_rows[0]['species']} ({bird_rows[0]['confidence']:.3f})")
            for row in bird_rows[1:3]:
                print(f"         备选: {row['species']} ({row['confidence']:.3f})")
        elif non_bird_rows:
            print(
                f"[BirdNET] 非鸟类声: {non_bird_rows[0]['species']} "
                f"({non_bird_rows[0]['confidence']:.3f})，已忽略"
            )
        else:
            print(
                f"[BirdNET] 未达置信度阈值 (>{cfg.birdnet_confidence}) | "
                f"片段 RMS={source_rms:.4f} 增益×{gain:.1f} | {wav_path.name}"
            )
            print("  提示: 播放鸟叫/靠近声源再试；可试听 output/birdnet_clips/ 下 wav")

        return result

    def run_loop(self) -> None:
        cfg = self.sss_config
        raw_path = self._raw_path()
        print("SSS → BirdNET 监视已启动")
        print(f"  读取: {raw_path} ({'postfiltered' if cfg.use_postfiltered else 'separated'})")
        print(f"  采样率 {cfg.sample_rate} Hz | {cfg.n_channels} 通道 | 片段 {cfg.clip_seconds}s")
        print(f"  能量阈值 {cfg.trigger_energy} | BirdNET 置信度 ≥{cfg.birdnet_confidence} | 冷却 {cfg.birdnet_cooldown}s")
        print("  需 ODAS 正在写入该 raw 文件（run_sound_ptz_all 或 odaslive）")

        if not raw_path.is_file():
            print(f"  等待文件出现: {raw_path}")

        self.sound.start()
        try:
            while True:
                valid, sound_xyz = self.sound.parse_latest()
                energy = sound_xyz[3] if valid and sound_xyz else 0.0
                now = time.monotonic()

                if (
                    valid
                    and sound_xyz
                    and energy >= self.sss_config.trigger_energy
                    and now - self._last_run >= cfg.birdnet_cooldown
                ):
                    if self.run_once(sound_xyz):
                        self._last_run = now

                time.sleep(cfg.poll_interval)
        except KeyboardInterrupt:
            print("\nSSS → BirdNET 监视已停止")
        finally:
            self.sound.stop()
