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


def _kill_processes_matching(pattern: str, exclude_pid: Optional[int] = None) -> None:
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
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def port_is_available(host: str, port: int) -> bool:
    """Return True when host:port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def cleanup_stale_services(ports: tuple[int, ...]) -> None:
    """Best-effort cleanup of old bridge/ODAS debug processes."""
    patterns = (
        "odas_bridge.py",
        "odaslive",
        "run_sound_client.py",
    )
    for pattern in patterns:
        _kill_processes_matching(pattern)

    for port in ports:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)

    time.sleep(1.5)


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
            cleanup_stale_services((self.config.python_port, self.config.odas_port))

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
                " 请先执行: pkill -f odas_bridge; pkill -f odaslive; "
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
