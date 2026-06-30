#!/usr/bin/env python3
"""实时监控系统 CPU/内存/功耗，用于 fusion 主流程旁路 profiling。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil

from core.performance_monitor import rapl_available, read_rapl_energy_joules
from core.paths import OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="实时 CPU/内存/RAPL 监控")
    parser.add_argument("--duration", type=float, default=60.0, help="监控秒数")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔秒")
    parser.add_argument("--save", action="store_true", help="保存 CSV")
    parser.add_argument(
        "--rapl-sudo",
        action="store_true",
        help="通过 sudo cat 读 RAPL（普通用户运行）",
    )
    args = parser.parse_args()

    if os.geteuid() == 0:
        print("错误: 不要用 sudo 运行；请: python scripts/monitor_system.py --rapl-sudo")
        raise SystemExit(1)

    if args.rapl_sudo:
        from core.performance_monitor import enable_rapl_sudo

        enable_rapl_sudo()

    rapl_ok = rapl_available()
    print(f"=== 系统监控 {args.duration}s (间隔 {args.interval}s) ===")
    print(f"CPU 核: {psutil.cpu_count()} | RAPL: {'可用' if rapl_ok else '需 sudo'}")
    print("请在另一终端运行 fusion 主流程，或对着系统施加负载")
    print(f"{'时间':>8}  {'CPU%':>6}  {'MEM%':>6}  {'RSS_MB':>8}  {'Power_W':>8}")
    print("-" * 44)

    rows: list[str] = []
    prev_rapl = read_rapl_energy_joules() if rapl_ok else None
    prev_ts = time.perf_counter()
    end = prev_ts + args.duration

    while time.perf_counter() < end:
        time.sleep(args.interval)
        now = time.perf_counter()
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        rss_mb = mem.used / (1024 * 1024)

        watts = ""
        power: float | None = None
        if rapl_ok:
            cur = read_rapl_energy_joules()
            if cur and prev_rapl:
                dt = now - prev_ts
                if dt > 0:
                    power = max(0.0, (sum(cur.values()) - sum(prev_rapl.values())) / dt)
                    watts = f"{power:.1f}"
                prev_rapl = cur
        prev_ts = now

        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"{stamp}  {cpu:6.1f}  {mem.percent:6.1f}  {rss_mb:8.0f}  {watts:>8}"
        print(line)
        rows.append(f"{stamp},{cpu:.2f},{mem.percent:.2f},{rss_mb:.0f},{power or ''}")

    if args.save:
        out_dir = OUTPUT_DIR / "performance"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path.write_text("time,cpu_pct,mem_pct,rss_mb,power_w\n" + "\n".join(rows), encoding="utf-8")
        print(f"\n已保存: {path}")


if __name__ == "__main__":
    main()
