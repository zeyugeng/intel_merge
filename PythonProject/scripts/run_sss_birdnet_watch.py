#!/usr/bin/env python3
"""监视 ODAS SSS 分离音（postfiltered.raw）并在有声时调用 BirdNET。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import ODASConfig, SSSConfig, SoundConfig
from core.sound_pipeline import SoundPipeline, cleanup_stale_services, ports_ready
from core.sss_birdnet_watcher import SSSBirdnetWatcher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 ODAS SSS 增长的 raw 文件，截取片段送入 BirdNET",
    )
    parser.add_argument(
        "--with-odas",
        action="store_true",
        help="本脚本内启动 ODAS + 桥接（否则需已运行 run_sound_ptz_all）",
    )
    parser.add_argument("--odas-cfg", type=Path, default=ODASConfig().config_path)
    parser.add_argument(
        "--raw",
        choices=("postfiltered", "separated"),
        default="separated",
        help="读取 separated.raw（默认）或 postfiltered.raw",
    )
    parser.add_argument("--clip-sec", type=float, default=3.0, help="每次识别片段长度秒")
    parser.add_argument("--cooldown", type=float, default=6.0, help="两次 BirdNET 最小间隔秒")
    parser.add_argument(
        "--activity",
        type=float,
        default=0.01,
        help="声源 activity 触发阈值（与云台 --activity 一致）",
    )
    parser.add_argument("--energy", type=float, default=None, help="fusion/velocity 能量阈值（可选）")
    parser.add_argument(
        "--birdnet-conf",
        type=float,
        default=0.15,
        help="BirdNET 置信度阈值，默认 0.15",
    )
    parser.add_argument(
        "--birdnet-locale",
        default="zh",
        help="鸟类名称语言，默认 zh",
    )
    parser.add_argument("--poll", type=float, default=0.25, help="轮询间隔秒")
    parser.add_argument("--show-odas-log", action="store_true")
    parser.add_argument("--mic-check", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    odas_config = ODASConfig(config_path=args.odas_cfg)
    sound_config = SoundConfig(
        host=odas_config.host,
        port=odas_config.python_port,
        energy_threshold=args.energy if args.energy is not None else 0.25,
    )
    sss_config = SSSConfig(
        clip_seconds=args.clip_sec,
        birdnet_cooldown=args.cooldown,
        poll_interval=args.poll,
        use_postfiltered=args.raw == "postfiltered",
        trigger_energy=args.energy if args.energy is not None else args.activity,
        birdnet_confidence=args.birdnet_conf,
        birdnet_locale=args.birdnet_locale or "zh",
    )

    pipeline: SoundPipeline | None = None
    if args.with_odas:
        ports = (odas_config.python_port, odas_config.odas_port)
        if not args.no_clean:
            cleanup_stale_services(ports, host=odas_config.host)
        if not ports_ready(odas_config.host, *ports):
            raise SystemExit("端口被占用，请先停掉 main.py / 其他 ODAS 进程")
        pipeline = SoundPipeline(odas_config, quiet_odas=not args.show_odas_log)
        try:
            pipeline.start(clean_stale=False, mic_check=args.mic_check)
            if not pipeline.wait_odas_ready(timeout=8.0):
                raise SystemExit("ODAS 未就绪")
            print("ODAS 已启动，开始监视 SSS 输出...")
            SSSBirdnetWatcher(sss_config, sound_config).run_loop()
        finally:
            pipeline.stop()
    else:
        SSSBirdnetWatcher(sss_config, sound_config).run_loop()


if __name__ == "__main__":
    main()
