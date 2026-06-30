"""CPU / memory / Intel RAPL power sampling for pipeline benchmarks."""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


RAPL_BASE = Path("/sys/class/powercap/intel-rapl/intel-rapl:0")
_RAPL_USE_SUDO = False


def enable_rapl_sudo() -> bool:
    """Read RAPL via `sudo cat` (use as normal user; do not sudo the whole benchmark)."""
    global _RAPL_USE_SUDO
    _RAPL_USE_SUDO = True
    return rapl_available()


def _read_sysfs_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (OSError, PermissionError):
        if not _RAPL_USE_SUDO:
            return None
        try:
            result = subprocess.run(
                ["sudo", "cat", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    except ValueError:
        return None
    return None


@dataclass
class PerformanceSample:
    ts: float
    cpu_percent: float
    mem_rss_mb: float
    mem_percent: float
    rapl_watts: Optional[float] = None


@dataclass
class PerformanceReport:
    label: str
    duration_s: float
    iterations: int
    latency_ms_mean: Optional[float] = None
    latency_ms_p95: Optional[float] = None
    throughput: Optional[str] = None
    cpu_mean: float = 0.0
    cpu_peak: float = 0.0
    mem_rss_peak_mb: float = 0.0
    power_watts_mean: Optional[float] = None
    power_watts_peak: Optional[float] = None
    notes: str = ""
    samples: list[PerformanceSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "duration_s": round(self.duration_s, 3),
            "iterations": self.iterations,
            "latency_ms_mean": self.latency_ms_mean,
            "latency_ms_p95": self.latency_ms_p95,
            "throughput": self.throughput,
            "cpu_mean_pct": round(self.cpu_mean, 1),
            "cpu_peak_pct": round(self.cpu_peak, 1),
            "mem_rss_peak_mb": round(self.mem_rss_peak_mb, 1),
            "power_watts_mean": round(self.power_watts_mean, 2) if self.power_watts_mean else None,
            "power_watts_peak": round(self.power_watts_peak, 2) if self.power_watts_peak else None,
            "notes": self.notes,
        }


def read_rapl_energy_joules() -> Optional[dict[str, float]]:
    """Read Intel RAPL domain energy (J). Direct read or `sudo cat` if enabled."""
    if not RAPL_BASE.is_dir():
        return None
    domains: dict[str, float] = {}
    for domain in sorted(RAPL_BASE.glob("intel-rapl:0:*")):
        name = _read_sysfs_text(domain / "name")
        energy = _read_sysfs_text(domain / "energy_uj")
        if name is None or energy is None:
            return None
        domains[name] = int(energy) / 1_000_000.0
    return domains or None


def rapl_available() -> bool:
    return read_rapl_energy_joules() is not None


class PerformanceMonitor:
    """Background sampler for process + system CPU/RAM and optional RAPL power."""

    def __init__(self, interval: float = 0.5, pid: Optional[int] = None):
        if psutil is None:
            raise ImportError("需要 psutil: pip install psutil")
        self.interval = interval
        self._proc = psutil.Process(pid or psutil.Process().pid)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: list[PerformanceSample] = []
        self._prev_rapl: Optional[dict[str, float]] = None
        self._prev_rapl_ts: Optional[float] = None
        self._rapl_ok = rapl_available()

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[PerformanceSample]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 3)
        return self.samples

    def _sample_rapl_watts(self, now: float) -> Optional[float]:
        if not self._rapl_ok:
            return None
        current = read_rapl_energy_joules()
        if current is None:
            self._rapl_ok = False
            return None
        total_j = sum(current.values())
        watts: Optional[float] = None
        if self._prev_rapl is not None and self._prev_rapl_ts is not None:
            prev_j = sum(self._prev_rapl.values())
            dt = now - self._prev_rapl_ts
            if dt > 0:
                watts = max(0.0, (total_j - prev_j) / dt)
        self._prev_rapl = current
        self._prev_rapl_ts = now
        return watts

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            try:
                cpu = self._proc.cpu_percent(interval=None)
                mem = self._proc.memory_info()
                mem_pct = self._proc.memory_percent()
            except psutil.Error:
                break
            watts = self._sample_rapl_watts(now)
            self.samples.append(
                PerformanceSample(
                    ts=now,
                    cpu_percent=cpu,
                    mem_rss_mb=mem.rss / (1024 * 1024),
                    mem_percent=mem_pct,
                    rapl_watts=watts,
                )
            )
            self._stop.wait(self.interval)

    @staticmethod
    def summarize(
        label: str,
        samples: list[PerformanceSample],
        duration_s: float,
        iterations: int = 0,
        latencies: Optional[list[float]] = None,
        notes: str = "",
    ) -> PerformanceReport:
        cpus = [s.cpu_percent for s in samples if s.cpu_percent >= 0]
        mems = [s.mem_rss_mb for s in samples]
        watts = [s.rapl_watts for s in samples if s.rapl_watts is not None]

        report = PerformanceReport(
            label=label,
            duration_s=duration_s,
            iterations=iterations,
            cpu_mean=statistics.mean(cpus) if cpus else 0.0,
            cpu_peak=max(cpus) if cpus else 0.0,
            mem_rss_peak_mb=max(mems) if mems else 0.0,
            notes=notes,
            samples=samples,
        )
        if latencies:
            report.latency_ms_mean = statistics.mean(latencies) * 1000
            report.latency_ms_p95 = (
                statistics.quantiles(latencies, n=20)[18] * 1000
                if len(latencies) >= 20
                else max(latencies) * 1000
            )
            mean_s = statistics.mean(latencies)
            if mean_s > 0:
                report.throughput = f"{1.0 / mean_s:.1f} ops/s"
        if watts:
            report.power_watts_mean = statistics.mean(watts)
            report.power_watts_peak = max(watts)
        return report


def run_monitored(
    label: str,
    workload: Callable[[], None],
    *,
    repeats: int = 1,
    warmup: int = 0,
    sample_interval: float = 0.5,
) -> PerformanceReport:
    """Run callable under monitoring; returns aggregated report."""
    monitor = PerformanceMonitor(interval=sample_interval)
    latencies: list[float] = []

    for _ in range(warmup):
        workload()

    monitor.start()
    start = time.perf_counter()
    for _ in range(repeats):
        t0 = time.perf_counter()
        workload()
        latencies.append(time.perf_counter() - t0)
    duration = time.perf_counter() - start
    samples = monitor.stop()

    return PerformanceMonitor.summarize(
        label,
        samples,
        duration_s=duration,
        iterations=repeats,
        latencies=latencies,
    )


def print_report(report: PerformanceReport) -> None:
    print(f"\n=== {report.label} ===")
    print(f"  时长: {report.duration_s:.2f}s | 迭代: {report.iterations}")
    if report.latency_ms_mean is not None:
        print(
            f"  延迟: mean={report.latency_ms_mean:.1f} ms"
            f" p95≈{report.latency_ms_p95:.1f} ms | {report.throughput}"
        )
    print(f"  CPU: mean={report.cpu_mean:.1f}% peak={report.cpu_peak:.1f}%")
    print(f"  内存 RSS 峰值: {report.mem_rss_peak_mb:.1f} MB")
    if report.power_watts_mean is not None:
        print(
            f"  功耗(RAPL): mean={report.power_watts_mean:.1f} W"
            f" peak={report.power_watts_peak:.1f} W"
        )
    else:
        print("  功耗(RAPL): 不可用（普通用户请加 --rapl-sudo；勿 sudo 整个 benchmark）")
    if report.notes:
        print(f"  备注: {report.notes}")
