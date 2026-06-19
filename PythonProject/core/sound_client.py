import json
import socket
import threading
import time
from typing import Any, Dict, Optional, Tuple

from .config import SoundConfig
from .sound_format import normalize_sound_source


class SoundSourceClient:
    """Receive flattened sound-source JSON from the ODAS bridge."""

    def __init__(self, config: Optional[SoundConfig] = None, reconnect_delay: float = 1.0):
        self.config = config or SoundConfig()
        self.reconnect_delay = reconnect_delay
        self._socket: Optional[socket.socket] = None
        self._buffer = ""
        self._latest: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected_once = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self.config.host, self.config.port))
        return sock

    def _receive_loop(self) -> None:
        while self._running:
            try:
                self._socket = self._connect()
                self._buffer = ""
                if not self._connected_once:
                    print(f"已连接声源服务器 {self.config.host}:{self.config.port}")
                    self._connected_once = True

                while self._running:
                    try:
                        chunk = self._socket.recv(1024).decode("utf-8")
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    self._buffer += chunk
                    self._consume_buffer()
            except OSError as exc:
                if self._running and not self._connected_once:
                    print(f"无法连接声源服务器，等待重试: {exc}")
            finally:
                if self._socket:
                    try:
                        self._socket.close()
                    except OSError:
                        pass
                    self._socket = None

            if self._running:
                time.sleep(self.reconnect_delay)

    def _consume_buffer(self) -> None:
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"声源 JSON 解析失败: {exc}")
                continue
            with self._lock:
                self._latest = payload

    def parse_latest(self) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
        with self._lock:
            if self._latest is None:
                return False, None
            data = self._latest

        try:
            source = normalize_sound_source(data)
            if source is None:
                return False, None
            sound_x = source.x
            if self.config.invert_x:
                sound_x = -sound_x
            sound_x = max(-1.0, min(1.0, sound_x))
            return True, (sound_x, source.y, source.z, source.energy)
        except (TypeError, ValueError) as exc:
            print(f"声源数据解析失败: {exc}")
            return False, None
