"""OpenVINO helpers aligned with Intel OpenVINO quickstart (Core, devices, YOLO export)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def require_openvino() -> None:
    try:
        import openvino  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "未安装 OpenVINO。请执行: pip install -r requirements-openvino.txt"
        ) from exc


def get_core():
    require_openvino()
    from openvino import Core

    return Core()


def list_devices() -> list[str]:
    return list(get_core().available_devices)


def normalize_ov_device(name: str) -> str:
    device = (name or "CPU").strip().upper()
    available = list_devices()
    if device in available:
        return device
    if device == "AUTO" and available:
        return "AUTO"
    fallback = "CPU" if "CPU" in available else available[0]
    logger.warning("OpenVINO 设备 %s 不可用，使用 %s（可用: %s）", device, fallback, available)
    return fallback


def ultralytics_device_for_ov(ov_device: str) -> str:
    """Ultralytics OpenVINO backend expects device strings like intel:CPU."""
    return f"intel:{normalize_ov_device(ov_device)}"


def openvino_model_dir_for_pt(pt_path: Path) -> Path:
    return pt_path.parent / f"{pt_path.stem}_openvino_model"


def find_openvino_xml(model_dir: Path) -> Path | None:
    if not model_dir.is_dir():
        return None
    xml_files = sorted(model_dir.glob("*.xml"))
    if not xml_files:
        return None
    return xml_files[0]


def export_yolo_to_openvino(
    pt_path: Path,
    *,
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    force: bool = False,
) -> Path:
    """Export Ultralytics YOLO .pt to *_openvino_model/ (IR .xml + .bin)."""
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"未找到 PyTorch 权重: {pt_path}")

    out_dir = openvino_model_dir_for_pt(pt_path)
    if not force and find_openvino_xml(out_dir) is not None:
        return out_dir

    require_openvino()
    from ultralytics import YOLO

    logger.info("正在导出 OpenVINO 模型: %s -> %s", pt_path, out_dir)
    model = YOLO(str(pt_path))
    model.export(
        format="openvino",
        imgsz=imgsz,
        half=half,
        int8=int8,
    )
    if find_openvino_xml(out_dir) is None:
        raise RuntimeError(f"导出完成但未找到 OpenVINO IR: {out_dir}")
    return out_dir


def resolve_yolo_openvino_dir(
    pt_path: Path,
    *,
    auto_export: bool = True,
    imgsz: int = 640,
) -> Path:
    """Return OpenVINO model directory for a YOLO .pt path."""
    pt_path = Path(pt_path)
    out_dir = openvino_model_dir_for_pt(pt_path)
    if find_openvino_xml(out_dir) is not None:
        return out_dir
    if auto_export and pt_path.exists():
        return export_yolo_to_openvino(pt_path, imgsz=imgsz)
    raise FileNotFoundError(
        f"未找到 OpenVINO 模型目录 {out_dir}，请先运行 scripts/export_yolo_openvino.py"
    )


def compile_model(model_path: Path, device: str = "CPU"):
    """Compile ONNX / IR / TFLite for direct OpenVINO inference (PDF quickstart style)."""
    core = get_core()
    model_path = Path(model_path)
    ov_device = normalize_ov_device(device)
    ov_model = core.read_model(model=str(model_path))
    return core.compile_model(ov_model, device_name=ov_device)


def print_device_summary() -> None:
    core = get_core()
    print(f"OpenVINO {__import__('openvino').__version__}")
    for dev in core.available_devices:
        try:
            name = core.get_property(dev, "FULL_DEVICE_NAME")
        except Exception:
            name = dev
        print(f"  {dev}: {name}")
