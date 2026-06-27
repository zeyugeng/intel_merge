#!/usr/bin/env python3
"""一键主流程：声源跟踪 / 视觉跟云台 / 声视融合（对齐 intelcup/main.py）。"""

import argparse
import sys
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import (
    ODASConfig,
    PTZConfig,
    PTZTrackConfig,
    SoundConfig,
    VisualConfig,
    VisualPTZTrackConfig,
)
from core.ptz_camera import PTZCameraController
from core.ptz_tracker import MainThreadCameraPreview, SoundPTZTracker
from core.serial_ptz import SerialPTZConfig, SerialPanTiltBackend
from core.sound_pipeline import SoundPipeline, cleanup_stale_services, ports_ready
from core.usb_fusion import USBAudioVisualFusion
from core.visual_ptz_tracker import VisualPTZTracker


def _class_ids_from_names(model_names: dict, names: list[str]) -> tuple:
    wanted = {n.lower() for n in names}
    ids = [int(cid) for cid, name in model_names.items() if str(name).lower() in wanted]
    return tuple(ids)


def build_configs(args):
    odas_config = ODASConfig(config_path=args.odas_cfg)

    sound_config = SoundConfig(
        host=odas_config.host,
        port=odas_config.python_port,
    )
    if args.energy is not None:
        sound_config.energy_threshold = args.energy
    if args.invert_x is not None:
        sound_config.invert_x = args.invert_x

    track_config = PTZTrackConfig(show_preview=not args.no_preview)
    if args.tracking_mode is not None:
        track_config.tracking_mode = args.tracking_mode
    if args.activity is not None:
        track_config.activity_threshold = args.activity
    if args.trigger_interval is not None:
        track_config.trigger_interval = args.trigger_interval
    if args.move_time_ms is not None:
        track_config.move_time_ms = args.move_time_ms
    if args.kp_pan is not None:
        track_config.kp_pan = args.kp_pan
    if args.kp_tilt is not None:
        track_config.kp_tilt = args.kp_tilt
    if args.invert_y is not None:
        track_config.invert_y = args.invert_y
    if args.invert_pan is not None:
        track_config.invert_pan = args.invert_pan
    if args.deadzone is not None:
        track_config.deadzone = args.deadzone
    if args.max_speed is not None:
        track_config.max_speed = args.max_speed
    if args.control_interval is not None:
        track_config.control_interval = args.control_interval

    visual_config = VisualConfig(conf=args.conf)
    if args.target_class:
        from ultralytics import YOLO

        model_path = visual_config.model_path or str(ROOT / "models" / "yolo26n.pt")
        names = YOLO(model_path).names
        class_ids = _class_ids_from_names(names, args.target_class)
        if not class_ids:
            raise SystemExit(f"模型中未找到类别: {args.target_class}，可用: {names}")
        visual_config.target_classes = class_ids

    visual_track_config = VisualPTZTrackConfig()
    if args.visual_pan_k is not None:
        visual_track_config.pan_k = args.visual_pan_k
    if args.visual_tilt_k is not None:
        visual_track_config.tilt_k = args.visual_tilt_k
    if args.visual_deadzone is not None:
        visual_track_config.dead_zone = args.visual_deadzone
    if args.visual_max_step is not None:
        visual_track_config.max_step = args.visual_max_step
    if args.visual_ptz_interval is not None:
        visual_track_config.ptz_interval = args.visual_ptz_interval
    if args.move_time_ms is not None:
        visual_track_config.move_time_ms = args.move_time_ms
    if args.invert_pan is not None:
        visual_track_config.invert_pan = args.invert_pan
    if args.invert_tilt is not None:
        visual_track_config.invert_tilt = args.invert_tilt

    return odas_config, sound_config, track_config, visual_config, visual_track_config


def create_ptz_backend(args, track_config: PTZTrackConfig, mode: str):
    if args.ptz_backend == "serial":
        if args.move_time_ms is not None:
            serial_move_ms = args.move_time_ms
        elif mode == "sound" and track_config.tracking_mode == "velocity":
            serial_move_ms = 120
        elif mode == "sound":
            serial_move_ms = track_config.move_time_ms
        else:
            serial_move_ms = 200
        return SerialPanTiltBackend(
            SerialPTZConfig(
                port=args.serial_port,
                baud=args.serial_baud,
                default_time_ms=serial_move_ms,
                angle_step_per_speed=args.angle_step,
            )
        )
    return PTZCameraController(PTZConfig())


def run_birdnet_if_requested(audio_path: Path | None) -> None:
    if audio_path is None:
        return
    from core.birdnet_infer import format_predictions, predict_audio

    print(f"=== BirdNET 分析: {audio_path} ===")
    try:
        predictions = predict_audio(audio_path)
        print(format_predictions(predictions))
    except Exception as exc:
        print(f"BirdNET 分析失败: {exc}")


def run_sound_flow(args, odas_config, sound_config, track_config) -> None:
    ports = (odas_config.python_port, odas_config.odas_port)
    host = odas_config.host

    if not args.no_clean:
        print("清理旧进程与占用端口...")
        cleanup_stale_services(ports, host=host)

    if not ports_ready(host, odas_config.python_port, odas_config.odas_port):
        print(
            f"错误: 端口 {odas_config.odas_port}/{odas_config.python_port} 被占用，"
            "声源跟踪无法启动。"
        )
        print("常见原因: 另一个终端正在跑 intelcup/main.py")
        print("解决: 在那个终端 Ctrl+C，或执行 pkill -f 'intel_merge/intelcup'")
        return

    pipeline = SoundPipeline(odas_config, quiet_odas=not args.show_odas_log)
    preview = MainThreadCameraPreview() if track_config.show_preview else None

    try:
        pipeline.start(clean_stale=False, mic_check=args.mic_check)
        print("等待 ODAS 稳定输出...")
        if not pipeline.wait_odas_ready(timeout=5.0):
            print("ODAS 未就绪，请检查麦克风与 ../odas/build")
            return

        ptz = create_ptz_backend(args, track_config, "sound")
        if not ptz.connect():
            print("云台未连接，声源坐标继续输出；云台将不会转动。")

        tracker = SoundPTZTracker(
            ptz,
            sound_config=sound_config,
            track_config=track_config,
            preview=preview,
        )

        if preview is not None:
            worker = Thread(target=tracker.run, daemon=True)
            worker.start()
            preview.run_forever()
            preview.stop()
            worker.join(timeout=3.0)
        else:
            tracker.run()
    except RuntimeError as exc:
        print(f"启动失败: {exc}")
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        pipeline.stop()
        if preview is not None:
            preview.stop()


def run_visual_flow(args, visual_config, visual_track_config) -> None:
    if args.ptz_backend != "serial":
        print("visual 模式仅支持串口云台，请使用 --ptz-backend serial")
        return
    ptz = create_ptz_backend(args, PTZTrackConfig(), "visual")
    if not ptz.connect():
        print("云台未连接，仅显示检测画面。")
    try:
        VisualPTZTracker(
            ptz,
            visual_config=visual_config,
            track_config=visual_track_config,
        ).run()
    except KeyboardInterrupt:
        print("\n已退出")


def run_fusion_flow(args, odas_config, sound_config, visual_config) -> None:
    ports = (odas_config.python_port, odas_config.odas_port)
    host = odas_config.host
    if not args.no_clean:
        cleanup_stale_services(ports, host=host)
    if not ports_ready(host, odas_config.python_port, odas_config.odas_port):
        print("错误: 端口被占用，请先停掉 intelcup/main.py")
        return

    pipeline = SoundPipeline(odas_config, quiet_odas=not args.show_odas_log)
    try:
        pipeline.start(clean_stale=False, mic_check=args.mic_check)
        print("等待 ODAS 稳定输出...")
        if not pipeline.wait_odas_ready(timeout=3.0):
            return

        fusion = USBAudioVisualFusion(
            sound_config=sound_config,
            visual_config=visual_config,
        )
        fusion.run()
    except RuntimeError as exc:
        print(f"启动失败: {exc}")
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="主流程：sound=status_1 | visual=status_2 | fusion=声视高亮",
    )
    parser.add_argument(
        "--mode",
        choices=("sound", "visual", "fusion"),
        default="sound",
        help="sound=声源跟云台(status_1); visual=YOLO跟云台(status_2); fusion=声视融合",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="sound 模式专用：不打开摄像头预览（默认会打开预览窗口）",
    )
    parser.add_argument(
        "--tracking-mode",
        choices=("absolute", "velocity"),
        default=None,
        help="sound 模式：absolute=main.py atan2；velocity=连续速度",
    )
    parser.add_argument("--activity", type=float, default=None, help="sound absolute 阈值，默认 0.001")
    parser.add_argument("--trigger-interval", type=float, default=None, help="sound absolute 触发间隔秒")
    parser.add_argument("--energy", type=float, default=None, help="sound velocity / fusion 能量阈值")
    parser.add_argument("--conf", type=float, default=0.3, help="visual/fusion YOLO 置信度")
    parser.add_argument(
        "--target-class",
        nargs="+",
        default=None,
        help="检测类别名，如 bird 或 person（默认 bird）",
    )
    parser.add_argument("--visual-pan-k", type=float, default=None, help="visual 水平增益")
    parser.add_argument("--visual-tilt-k", type=float, default=None, help="visual 俯仰增益")
    parser.add_argument("--visual-deadzone", type=float, default=None, help="visual 死区")
    parser.add_argument("--visual-max-step", type=float, default=None, help="visual 单步最大角度")
    parser.add_argument("--visual-ptz-interval", type=float, default=None, help="visual 云台指令间隔秒")
    parser.add_argument("--kp-pan", type=float, default=None, help="sound velocity 水平增益")
    parser.add_argument("--kp-tilt", type=float, default=None, help="sound velocity 俯仰增益")
    parser.add_argument(
        "--invert-x",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否反转声源 x",
    )
    parser.add_argument(
        "--invert-y",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否反转声源 y",
    )
    parser.add_argument(
        "--invert-pan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="水平转向反转（默认已反转以匹配串口云台；仍反了用 --no-invert-pan）",
    )
    parser.add_argument(
        "--invert-tilt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="反转俯仰转向（visual 模式）",
    )
    parser.add_argument("--deadzone", type=float, default=None, help="sound velocity 死区")
    parser.add_argument("--max-speed", type=float, default=None, help="sound velocity 最大速度")
    parser.add_argument("--control-interval", type=float, default=None, help="sound velocity 刷新间隔")
    parser.add_argument("--odas-cfg", type=Path, default=ODASConfig().config_path, help="ODAS 配置")
    parser.add_argument("--show-odas-log", action="store_true", help="ODAS 日志打到终端")
    parser.add_argument("--ptz-backend", choices=("serial", "onvif"), default="serial")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--serial-baud", type=int, default=115200)
    parser.add_argument("--angle-step", type=float, default=8.0)
    parser.add_argument("--move-time-ms", type=int, default=None, help="云台转动时间 ms")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--mic-check", action="store_true")
    parser.add_argument(
        "--birdnet-audio",
        type=Path,
        default=None,
        help="启动时用 BirdNET 分析指定 wav（对应 main.py SoundPredict）",
    )
    args = parser.parse_args()

    if args.target_class is None and args.mode == "visual":
        # intelcup/main.py status_2 默认检测 person
        args.target_class = ["person"]
    elif args.target_class is None and args.mode == "fusion":
        args.target_class = ["bird"]

    odas_config, sound_config, track_config, visual_config, visual_track_config = build_configs(args)

    run_birdnet_if_requested(args.birdnet_audio)

    mode_labels = {
        "sound": "声源跟踪 (status_1)",
        "visual": "视觉跟云台 (status_2)",
        "fusion": "USB 声视融合",
    }
    print(f"=== 主流程: {mode_labels[args.mode]} ===")

    if args.mode == "sound":
        if args.no_preview:
            print("提示: 已使用 --no-preview，不会弹出摄像头窗口")
        else:
            print("提示: 将打开摄像头预览窗口「Camera 实时预览」")
        run_sound_flow(args, odas_config, sound_config, track_config)
    elif args.mode == "visual":
        run_visual_flow(args, visual_config, visual_track_config)
    else:
        run_fusion_flow(args, odas_config, sound_config, visual_config)


if __name__ == "__main__":
    main()
