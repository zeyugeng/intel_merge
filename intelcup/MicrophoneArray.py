import json
import socket
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time


@dataclass(frozen=True)
class SoundSource:
    id: int
    tag: str
    x: float
    y: float
    z: float
    activity: float


@dataclass(frozen=True)
class SoundSourceFrame:
    time_stamp: int
    sources: List[SoundSource]
    raw: Dict[str, Any]


class MicrophoneArray:
    """Receive ODAS SST socket output and keep the latest complete frame."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, recv_size: int = 4096):
        self.host = host
        self.port = port
        self.recv_size = recv_size

        self._server: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._addr = None
        self._connected = False
        self._running = False
        self._buffer = ""
        self._latest: Optional[SoundSourceFrame] = None
        self._lock = threading.Lock()
        self._receiver_thread: Optional[threading.Thread] = None

    def connect(self, wait: bool = True) -> bool:
        """Start the socket server and receive ODAS data.

        ODAS connects to this program, so this method listens on host:port.
        If wait=True, it blocks until ODAS connects.
        """
        if self.is_connected():
            return True

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self._running = True

        if wait:
            self._accept_once()
        else:
            threading.Thread(target=self._accept_once, daemon=True).start()

        return self.is_connected()

    def is_connected(self) -> bool:
        """Return True after ODAS has connected and the receiver is running."""
        with self._lock:
            return self._connected

    def get_latest(self) -> Optional[SoundSourceFrame]:
        """Return only the newest complete SST frame received from ODAS."""
        with self._lock:
            return self._latest

    def print_raw_loop(self) -> None:
        """Accept one ODAS connection and print raw received text for debugging."""
        if self._server is None:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((self.host, self.port))
            self._server.listen(1)

        print(f"Waiting for ODAS connection on {self.host}:{self.port} ...")
        conn, addr = self._server.accept()
        print(f"ODAS connected: {addr}")

        with conn:
            while True:
                data = conn.recv(self.recv_size)
                if not data:
                    break
                print(data.decode("utf-8", errors="ignore"), end="")

    def close(self) -> None:
        self._running = False
        with self._lock:
            self._connected = False

        for sock in (self._conn, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def _accept_once(self) -> None:
        if self._server is None:
            return

        print(f"Waiting for ODAS connection on {self.host}:{self.port} ...")
        conn, addr = self._server.accept()

        with self._lock:
            self._conn = conn
            self._addr = addr
            self._connected = True

        print(f"ODAS connected: {addr}")
        self._receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receiver_thread.start()

    def _receive_loop(self) -> None:
        while self._running and self._conn is not None:
            try:
                data = self._conn.recv(self.recv_size)
            except OSError:
                break

            if not data:
                break

            text = data.decode("utf-8", errors="ignore")
            self._buffer += text
            self._parse_buffer()

        with self._lock:
            self._connected = False

    def _parse_buffer(self) -> None:
        decoder = json.JSONDecoder()

        while self._buffer:
            stripped = self._buffer.lstrip()
            skipped = len(self._buffer) - len(stripped)
            if skipped:
                self._buffer = stripped

            if not self._buffer:
                return

            start = self._buffer.find("{")
            if start < 0:
                self._buffer = ""
                return
            if start > 0:
                self._buffer = self._buffer[start:]

            try:
                obj, end = decoder.raw_decode(self._buffer)
            except json.JSONDecodeError:
                return

            self._buffer = self._buffer[end:]
            frame = self._to_frame(obj)
            if frame is not None:
                with self._lock:
                    self._latest = frame

    @staticmethod
    def _to_frame(obj: Any) -> Optional[SoundSourceFrame]:
        if not isinstance(obj, dict):
            return None

        raw_sources = obj.get("src", [])
        if not isinstance(raw_sources, list):
            raw_sources = []

        sources = []
        for item in raw_sources[:4]:
            if not isinstance(item, dict):
                continue
            sources.append(
                SoundSource(
                    id=int(item.get("id", 0)),
                    tag=str(item.get("tag", "")),
                    x=float(item.get("x", 0.0)),
                    y=float(item.get("y", 0.0)),
                    z=float(item.get("z", 0.0)),
                    activity=float(item.get("activity", 0.0)),
                )
            )

        return SoundSourceFrame(
            time_stamp=int(obj.get("timeStamp", 0)),
            sources=sources,
            raw=obj,
        )

    @staticmethod
    def is_silent_frame(frame: Optional[SoundSourceFrame], threshold: float = 0.001) -> bool:
        # 无帧数据，直接判定静音
        if frame is None:
            print("no sound_frame")
            return True
        # 遍历所有声源，任意一个activity超过阈值则不是静音
        for src in frame.sources:
            if src.activity > threshold:
                return False
        # 全部声源音量都低于阈值，静音
        return True


if __name__ == "__main__":
    mic = MicrophoneArray(host="0.0.0.0", port=5000)

    try:
        mic.connect(wait=False)

        while True:
            frame = mic.get_latest()

            if frame is not None:
                print("timeStamp:", frame.time_stamp)

                for src in frame.sources:
                    print(
                        f"id={src.id}, "
                        f"x={src.x:.3f}, y={src.y:.3f}, z={src.z:.3f}, "
                        f"activity={src.activity:.3f}"
                    )

                print("-" * 40)

            time.sleep(0.05)

    finally:
        mic.close()
