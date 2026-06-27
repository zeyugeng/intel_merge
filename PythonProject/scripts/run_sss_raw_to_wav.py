#!/usr/bin/env python3
"""将 ODAS SSS 的 separated/postfiltered.raw 转为 mono wav（供 BirdNET 或人工试听）。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SSSConfig
from core.sss_reader import convert_raw_to_wav


def main() -> None:
    parser = argparse.ArgumentParser(description="SSS raw → mono wav")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="raw 路径，默认 separated.raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 wav，默认 output/birdnet_clips/sss_export.wav",
    )
    parser.add_argument("--channel", type=int, default=None, help="指定通道索引，默认选能量最大通道")
    parser.add_argument("--raw", choices=("postfiltered", "separated"), default="separated")
    args = parser.parse_args()

    cfg = SSSConfig(use_postfiltered=args.raw == "postfiltered")
    raw_path = args.input or (
        cfg.postfiltered_path if cfg.use_postfiltered else cfg.separated_path
    )
    out_path = args.output or (cfg.clips_dir / "sss_export.wav")

    print(f"转换: {raw_path} → {out_path}")
    convert_raw_to_wav(
        raw_path,
        out_path,
        sample_rate=cfg.sample_rate,
        hop_size=cfg.hop_size,
        n_channels=cfg.n_channels,
        channel=args.channel,
        n_bits=cfg.n_bits,
    )
    print("完成")


if __name__ == "__main__":
    main()
