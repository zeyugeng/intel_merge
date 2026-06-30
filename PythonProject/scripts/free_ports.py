#!/usr/bin/env python3
"""Kill processes listening on ODAS bridge ports (5000 / 9001) for current user."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PORTS = (5000, 9001)
PATTERNS = (
    "run_sound_ptz_all.py",
    "run_sss_birdnet_watch.py",
    "run_sound_client.py",
    "odas_bridge.py",
    "odaslive",
    "intelcup/main.py",
    "MicrophoneArray",
)


def port_bindable(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _port_hex(port: int) -> str:
    return f"{port:04X}"


def _listener_inodes(port: int) -> set[str]:
    port_hex = _port_hex(port)
    inodes: set[str] = set()
    for proc_net in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(proc_net).read_text(encoding="utf-8", errors="ignore").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr, state, inode = parts[1], parts[3], parts[9]
            if state != "0A":
                continue
            host_port = local_addr.rsplit(":", 1)[-1]
            if host_port.upper() == port_hex:
                inodes.add(inode)
    return inodes


def _pids_on_port_proc(port: int) -> list[int]:
    inodes = _listener_inodes(port)
    if not inodes:
        return []
    pids: set[int] = set()
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        fd_dir = entry / "fd"
        try:
            for fd in fd_dir.iterdir():
                target = os.readlink(fd)
                if not target.startswith("socket:["):
                    continue
                if target[8:-1] in inodes:
                    pids.add(pid)
        except (OSError, PermissionError):
            continue
    return sorted(pids)


def _read_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="ignore"
        ).strip()
    except OSError:
        return ""


def _kill_pid(pid: int, sig: signal.Signals) -> None:
    if pid in (os.getpid(), os.getppid()):
        return
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        print(f"  无权限结束 pid {pid}，请执行: sudo kill -9 {pid}")


def _kill_tree(pid: int, sig: signal.Signals) -> None:
    result = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                _kill_tree(int(line.strip()), sig)
            except ValueError:
                pass
    _kill_pid(pid, sig)


def pkill_patterns() -> None:
    me = os.getpid()
    for pattern in PATTERNS:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid in (me, os.getppid()):
                continue
            cmd = _read_cmdline(pid)
            print(f"  结束: pid {pid} ({pattern}) {cmd[:80]}")
            _kill_tree(pid, signal.SIGKILL)


def kill_port_listeners(port: int) -> None:
    for pid in _pids_on_port_proc(port):
        cmd = _read_cmdline(pid)
        print(f"  结束端口 {port} 监听: pid {pid} {cmd[:80]}")
        _kill_tree(pid, signal.SIGKILL)


def main() -> int:
    print("=== 释放端口 5000 / 9001 ===")
    pkill_patterns()
    for port in PORTS:
        kill_port_listeners(port)
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    time.sleep(0.8)

    ok = True
    for port in PORTS:
        free = port_bindable("127.0.0.1", port)
        status = "已释放" if free else "仍被占用"
        print(f"  端口 {port}: {status}")
        if not free:
            ok = False
            for pid in _pids_on_port_proc(port):
                print(f"    pid {pid}: {_read_cmdline(pid)}")
            print(f"    请执行: sudo kill -9 <PID>  或  sudo fuser -k {port}/tcp")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
