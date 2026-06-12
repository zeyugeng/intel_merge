import json
import socket
import threading
from typing import Any, Dict, Optional, Tuple

from .config import SoundConfig


class SoundSourceClient:
    """接收 ODAS 声源服务器推送的 JSON 数据（每行一条）。"""

    def __init__(self, config: Optional[SoundConfig] = None):
        self.config = config or SoundConfig()
        self._socket: Optional[socket.socket] = None
        self._buffer = ""
        self._latest: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._socket:
            self._socket.close()
            self._socket = None

    def _receive_loop(self) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.config.host, self.config.port))
            print(f"已连接声源服务器 {self.config.host}:{self.config.port}")

            while self._running:
                try:
                    chunk = self._socket.recv(1024).decode("utf-8")
                    if not chunk:
                        break
                    self._buffer += chunk
                    while "\n" in self._buffer:
                        line, self._buffer = self._buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        payload = json.loads(line)
                        with self._lock:
                            self._latest = payload
                except socket.timeout:
                    continue
                except json.JSONDecodeError as exc:
                    print(f"声源 JSON 解析失败: {exc}")
        except OSError as exc:
            print(f"无法连接声源服务器: {exc}")
        finally:
            if self._socket:
                self._socket.close()

    def parse_latest(self) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
        with self._lock:
            if self._latest is None:
                return False, None
            data = self._latest

        try:
            sound_x = float(data.get("x", 0))
            if self.config.invert_x:
                sound_x = -sound_x
            sound_x = max(-1.0, min(1.0, sound_x))
            sound_y = float(data.get("y", 0))
            sound_e = float(data.get("E", 0))
            return True, (sound_x, sound_y, sound_e)
        except (TypeError, ValueError) as exc:
            print(f"声源数据解析失败: {exc}")
            return False, None
