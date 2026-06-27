"""Intel oneAPI helpers: oneMKL / OpenMP / TBB for numerics (aligned with oneAPI PDF)."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def _numpy_blas_info() -> dict[str, str]:
    try:
        import numpy as np

        cfg = np.__config__
        if hasattr(cfg, "show_config"):
            # NumPy 2.x: parse via get_info helpers when available
            blas = {}
            for key in ("blas_opt_info", "lapack_opt_info"):
                try:
                    info = cfg.get_info(key)
                    if info:
                        blas[key] = str(info.get("libraries", info))
                except Exception:
                    pass
            if blas:
                return blas
        return {"numpy": np.__version__}
    except ImportError:
        return {}


def detect_oneapi_stack() -> dict[str, Any]:
    """Report MKL / TBB / OpenMP packages available in the current environment."""
    stack: dict[str, Any] = {
        "numpy_blas": _numpy_blas_info(),
        "mkl": None,
        "mkl_service": None,
        "intel_openmp": None,
        "tbb": None,
    }

    try:
        import mkl

        stack["mkl"] = getattr(mkl, "__version__", "installed")
    except ImportError:
        pass

    try:
        import mkl_service

        stack["mkl_service"] = "installed"
    except ImportError:
        pass

    try:
        import intel_openmp  # noqa: F401

        stack["intel_openmp"] = "installed"
    except ImportError:
        pass

    try:
        import tbb

        stack["tbb"] = getattr(tbb, "__version__", "installed")
    except ImportError:
        pass

    stack["oneapi_ready"] = stack["mkl"] is not None
    return stack


def require_oneapi() -> None:
    if not detect_oneapi_stack()["oneapi_ready"]:
        raise ImportError(
            "未检测到 oneMKL。请执行: pip install -r requirements-oneapi.txt "
            "-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
        )


def apply_oneapi_env(num_threads: int | None = None, verbose: bool = True) -> dict[str, Any]:
    """
    Configure MKL / OpenMP threading for Intel CPUs (oneAPI Math Kernel Library).

    Safe to call multiple times; only applies MKL thread count on first call.
    """
    global _APPLIED

    if num_threads is None:
        num_threads = min(8, os.cpu_count() or 4)

    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    os.environ.setdefault("KMP_BLOCKTIME", "0")

    mkl_threads = num_threads
    try:
        import mkl

        if not _APPLIED:
            mkl.set_num_threads(num_threads)
            mkl.set_dynamic(True)
        mkl_threads = mkl.get_max_threads()
    except ImportError:
        pass

    _APPLIED = True
    stack = detect_oneapi_stack()
    if verbose:
        print_oneapi_summary(stack)
        print(f"oneAPI 线程: OMP/MKL={num_threads}, MKL 实际={mkl_threads}")
    return stack


def print_oneapi_summary(stack: dict[str, Any] | None = None) -> None:
    stack = stack or detect_oneapi_stack()
    print("=== Intel oneAPI 数值库 ===")
    if stack.get("mkl"):
        print(f"  oneMKL: {stack['mkl']}")
    else:
        print("  oneMKL: 未安装 (pip install -r requirements-oneapi.txt)")
    if stack.get("tbb"):
        print(f"  oneTBB: {stack['tbb']}")
    if stack.get("intel_openmp"):
        print("  Intel OpenMP: 已安装")
    blas = stack.get("numpy_blas") or {}
    if blas:
        print(f"  NumPy BLAS: {blas}")
    print(f"  oneAPI 就绪: {stack.get('oneapi_ready')}")


def vtune_profile_hint(script: str = "scripts/run_sound_ptz_all.py") -> str:
    """Return a VTune Profiler command line hint (from oneAPI tooling PDF)."""
    return (
        f"VTune 性能分析示例:\n"
        f"  vtune -collect hotspots -app python -- {script} --vision-backend openvino --oneapi\n"
        f"  vtune -collect threading -app python -- {script}\n"
        f"文档: https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html"
    )
