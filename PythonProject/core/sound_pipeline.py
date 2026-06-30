"""Runtime management for the ODAS sound pipeline."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

from .config import ODASConfig
from .odas_bridge import OdasBridge


def _kill_processes_matching(
    pattern: str,
    exclude_pid: Optional[int] = None,
    sig: signal.Signals = signal.SIGTERM,
) -> None:
    """Kill other processes matching pattern, never the current launcher."""
    exclude_pid = exclude_pid or os.getpid()
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return

    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid in (exclude_pid, os.getppid()):
            continue
        _kill_process_tree(pid, sig)


def _kill_process_tree(pid: int, sig: signal.Signals) -> None:
    if pid in (os.getpid(), os.getppid()):
        return
    child = subprocess.run(
        ["pgrep", "-P", str(pid)],
        capture_output=True,
        text=True,
    )
    if child.returncode == 0:
        for line in child.stdout.splitlines():
            try:
                child_pid = int(line.strip())
            except ValueError:
                continue
            _kill_process_tree(child_pid, sig)
    _kill_pid(pid, sig)


def port_is_available(host: str, port: int) -> bool:
    """Return True when host:port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _read_proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="ignore"
        )
    except OSError:
        return ""


def _pids_on_port_proc(port: int) -> list[int]:
    """Find listening PIDs via /proc/net/tcp when lsof/ss are empty."""
    port_hex = f"{port:04X}"
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
            if not local_addr.endswith(f":{port_hex}") or state != "0A":
                continue
            inodes.add(inode)

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
                inode = target[8:-1]
                if inode in inodes:
                    pids.add(pid)
        except (OSError, PermissionError):
            continue
    return sorted(pids)


def _pids_on_port(port: int) -> list[int]:
    """Return PIDs holding a TCP port (listeners and connected sockets)."""
    pids: set[int] = set()
    for args in (
        ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        ["lsof", "-nP", "-t", f"-i:{port}"],
    ):
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            try:
                pids.add(int(line.strip()))
            except ValueError:
                continue

    result = subprocess.run(
        ["ss", "-lptn", f"sport = :{port}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "pid=" not in line:
                continue
            for token in line.split(","):
                token = token.strip()
                if token.startswith("pid="):
                    try:
                        pids.add(int(token.split("=", 1)[1]))
                    except ValueError:
                        pass

    result = subprocess.run(
        ["fuser", f"{port}/tcp"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for token in result.stdout.replace(f"{port}/tcp:", "").split():
            try:
                pids.add(int(token))
            except ValueError:
                continue

    for pid in _pids_on_port_proc(port):
        pids.add(pid)
    return sorted(pids)


def _kill_pid(pid: int, sig: signal.Signals) -> None:
    if pid in (os.getpid(), os.getppid()):
        return
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def _kill_port_listeners(port: int, sig: signal.Signals = signal.SIGTERM) -> None:
    """Kill processes using a TCP port (intelcup/main.py, stale bridge, etc.)."""
    if sig == signal.SIGTERM:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    else:
        subprocess.run(["fuser", "-k", "-KILL", f"{port}/tcp"], capture_output=True)

    for pid in _pids_on_port(port):
        _kill_process_tree(pid, sig)


def _kill_other_launchers() -> None:
    """Stop other PythonProject launcher instances (they hold 5000/9001 in-process)."""
    for pattern in (
        "run_sound_ptz_all.py",
        "run_sss_birdnet_watch.py",
        "run_sound_client.py",
    ):
        _kill_processes_matching(pattern)


def _kill_intelcup_main() -> None:
    """Stop intelcup/main.py even when launched as ``python main.py`` from intelcup/."""
    result = subprocess.run(["pgrep", "-f", "main.py"], capture_output=True, text=True)
    if result.returncode != 0:
        return

    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid in (os.getpid(), os.getppid()):
            continue
        cmdline = ""
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="ignore"
            )
        except OSError:
            pass
        cwd = ""
        try:
            cwd = str(Path(f"/proc/{pid}/cwd").resolve())
        except OSError:
            pass
        if "intelcup" in cwd or "intelcup" in cmdline or "MicrophoneArray" in cmdline:
            _kill_process_tree(pid, signal.SIGTERM)


def _kill_port_holder_processes(ports: tuple[int, ...]) -> None:
    """Kill anything listening on our ports, including stale run_sound_ptz_all."""
    seen: set[int] = set()
    for port in ports:
        for pid in _pids_on_port(port):
            if pid in seen or pid in (os.getpid(), os.getppid()):
                continue
            seen.add(pid)
            _kill_process_tree(pid, signal.SIGKILL)


def _describe_port_holders(port: int) -> str:
    result = subprocess.run(
        ["lsof", "-nP", "-i", f":{port}"],
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr or "").strip()


def ports_ready(host: str, python_port: int, odas_port: int) -> bool:
    return port_is_available(host, python_port) and port_is_available(host, odas_port)


def _cleanup_ports_once(ports: tuple[int, ...], force: bool = False) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    for port in ports:
        _kill_port_listeners(port, sig)


def _report_busy_ports(ports: tuple[int, ...], host: str) -> None:
    busy = [p for p in ports if not port_is_available(host, p)]
    if not busy:
        return
    print(f"端口仍被占用: {busy}")
    for port in busy:
        detail = _describe_port_holders(port)
        if detail:
            print(detail)
        for pid in _pids_on_port(port):
            cmd = _read_proc_cmdline(pid).strip()
            if cmd:
                print(f"  pid {pid}: {cmd}")
    print("可手动执行: fuser -k 5000/tcp 9001/tcp; pkill -9 -f run_sound_ptz_all")


def cleanup_stale_services(ports: tuple[int, ...], host: str = "127.0.0.1") -> None:
    """Best-effort cleanup of old bridge/ODAS/debug processes and port holders."""
    ensure_ports_free(ports, host=host)


def ensure_ports_free(
    ports: tuple[int, ...],
    host: str = "127.0.0.1",
    rounds: int = 6,
) -> bool:
    """Kill stale launchers and retry until ports can bind, or give up."""
    patterns = (
        "odas_bridge.py",
        "odaslive",
        "intelcup/main.py",
        "intel_merge/intelcup",
        "intel_merge/odas",
        "run_sound_ptz_all.py",
        "run_sss_birdnet_watch.py",
    )
    for round_idx in range(rounds):
        sig = signal.SIGKILL if round_idx >= 1 else signal.SIGTERM
        for pattern in patterns:
            _kill_processes_matching(pattern, sig=sig)
        _kill_other_launchers()
        _kill_intelcup_main()
        if round_idx >= 1:
            _kill_port_holder_processes(ports)
        _cleanup_ports_once(ports, force=round_idx >= 2)
        time.sleep(0.8 + round_idx * 0.4)

        if all(port_is_available(host, port) for port in ports):
            return True

    _kill_port_holder_processes(ports)
    subprocess.run(["fuser", "-k", "-KILL", "5000/tcp"], capture_output=True)
    subprocess.run(["fuser", "-k", "-KILL", "9001/tcp"], capture_output=True)
    time.sleep(1.0)

    if all(port_is_available(host, port) for port in ports):
        return True

    _report_busy_ports(ports, host)
    print("或执行: bash scripts/stop_services.sh")
    return False


class SoundPipeline:
    """Owns the ODAS subprocess and Python bridge lifecycle."""

    def __init__(self, config: Optional[ODASConfig] = None, quiet_odas: bool = True):
        self.config = config or ODASConfig()
        self.quiet_odas = quiet_odas
        self.bridge: Optional[OdasBridge] = None
        self.odas_proc: Optional[subprocess.Popen] = None
        self._odas_log_fp = None
        self._stopped = False

    def start(self, clean_stale: bool = True, mic_check: bool = False) -> None:
        if clean_stale:
            print("清理旧进程与占用端口...")
            cleanup_stale_services(
                (self.config.python_port, self.config.odas_port),
                host=self.config.host,
            )

        if mic_check:
            self._preflight_microphone()

        self._start_bridge()
        self._start_odas_with_retry()

    def wait_odas_ready(self, timeout: float = 3.0) -> bool:
        """Wait briefly and confirm ODAS is still running."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.odas_proc and self.odas_proc.poll() is not None:
                if self._odas_log_fp:
                    self._odas_log_fp.flush()
                self._print_odas_failure()
                return False
            time.sleep(0.5)
        return self.odas_proc is not None and self.odas_proc.poll() is None

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        if self.bridge:
            self.bridge.stop()
            self.bridge = None

        if self.odas_proc and self.odas_proc.poll() is None:
            self.odas_proc.terminate()
            try:
                self.odas_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.odas_proc.kill()
            print("ODAS 已停止")
        self.odas_proc = None

        if self._odas_log_fp:
            self._odas_log_fp.close()
            self._odas_log_fp = None

    def _start_bridge(self) -> None:
        if not port_is_available(self.config.host, self.config.odas_port) or not port_is_available(
            self.config.host, self.config.python_port
        ):
            raise RuntimeError(
                f"端口 {self.config.odas_port}/{self.config.python_port} 仍被占用。"
                " 请先 Ctrl+C 退出 intelcup/main.py，或执行: "
                "pkill -f 'intelcup/main.py'; pkill -f odas_bridge; pkill -f odaslive; "
                "fuser -k 5000/tcp 9001/tcp"
            )

        self.bridge = OdasBridge(
            self.config.host,
            self.config.odas_port,
            self.config.python_port,
        )
        self.bridge.start()

        if not self.bridge._ready.wait(timeout=2.0):
            raise RuntimeError("桥接服务启动超时")
        print("桥接端口就绪")

    def _preflight_microphone(self) -> None:
        print("麦克风预检 (14 通道 @ 32kHz, 2 秒)...")
        result = subprocess.run(
            [
                "arecord",
                "-D",
                "hw:1,0",
                "-f",
                "S16_LE",
                "-r",
                "32000",
                "-c",
                "14",
                "--period-time=1000",
                "--buffer-time=2000",
                "-d",
                "2",
                "/tmp/odas_mic_test.wav",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("麦克风录音 OK")
            return

        stderr = (result.stderr or "").strip()
        print("麦克风录音失败，ODAS 无法稳定运行:")
        if stderr:
            print(stderr)
        print("建议执行: cd ../odas && ./scripts/diagnose_mic.sh")
        raise RuntimeError("麦克风预检失败")

    def _start_odas_with_retry(self, max_attempts: int = 3) -> None:
        for attempt in range(1, max_attempts + 1):
            self._launch_odas_process()
            if self._wait_odas_alive(timeout=4.0):
                return

            self._stop_odas_only()
            if attempt < max_attempts:
                print(
                    f"ODAS 异常退出，1.5 秒后重试 "
                    f"({attempt}/{max_attempts})..."
                )
                time.sleep(1.5)

        if self._odas_log_fp:
            self._odas_log_fp.flush()
        self._print_odas_failure()
        raise RuntimeError("ODAS 多次启动失败，请检查麦克风 USB 连接")

    def _wait_odas_alive(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.odas_proc and self.odas_proc.poll() is not None:
                return False
            time.sleep(0.25)
        return self.odas_proc is not None and self.odas_proc.poll() is None

    def _stop_odas_only(self) -> None:
        if self.odas_proc and self.odas_proc.poll() is None:
            self.odas_proc.terminate()
            try:
                self.odas_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.odas_proc.kill()
                self.odas_proc.wait(timeout=1)
        self.odas_proc = None

        if self._odas_log_fp:
            self._odas_log_fp.close()
            self._odas_log_fp = None

    def _launch_odas_process(self) -> None:
        if not self.config.bin_path.is_file():
            raise FileNotFoundError(f"未找到 ODAS: {self.config.bin_path}")
        if not self.config.config_path.is_file():
            raise FileNotFoundError(f"未找到配置: {self.config.config_path}")

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            f"{self.config.lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
        )

        self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._odas_log_fp = open(
            self.config.log_path, "w", encoding="utf-8", buffering=1
        )

        print(
            f"启动 ODAS: {self.config.bin_path.name} -c "
            f"{self.config.config_path.name}"
        )
        if self.quiet_odas:
            print(f"ODAS 日志: {self.config.log_path}")

        odas_cwd = self.config.config_path.parent.parent
        self._cleanup_sss_raw_files(odas_cwd)
        self.odas_proc = subprocess.Popen(
            [str(self.config.bin_path), "-c", str(self.config.config_path)],
            cwd=str(odas_cwd),
            env=env,
            stdout=self._odas_log_fp if self.quiet_odas else None,
            stderr=subprocess.STDOUT if self.quiet_odas else None,
        )

    def _print_odas_failure(self) -> None:
        log_path = Path(self.config.log_path)
        exit_code = self.odas_proc.returncode if self.odas_proc else "unknown"
        print(f"ODAS 进程已退出 (code={exit_code})，请查看日志: {log_path}")
        print("若日志为空，常见原因是麦克风 USB 带宽不足或 bridge 未就绪。")
        print("可先单独测试: cd ../odas && ./scripts/run_odas_test.sh")
        if not log_path.exists():
            return
        tail = log_path.read_text(encoding="utf-8", errors="ignore").strip()
        if tail:
            print("--- odaslive 最后输出 ---")
            print(tail[-2000:])

    def _cleanup_sss_raw_files(self, odas_cwd: Path) -> None:
        """Remove stale SSS outputs so tail reads align to the new ODAS session."""
        for name in ("separated.raw", "postfiltered.raw"):
            path = odas_cwd / name
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
