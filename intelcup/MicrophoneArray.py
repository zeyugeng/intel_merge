import json
import socket
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


DEFAULT_ODAS_CMD = [
    "/home/intel2026/桌面/intel_merge/odas/build/bin/odaslive",
    "-c",
    "/home/intel2026/桌面/intel_merge/odas/config/myArray.cfg",
]


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


class odasconnecter:
    def __init__(self, odas_cmd: Optional[List[str]] = None):
        self.odas_cmd = odas_cmd or DEFAULT_ODAS_CMD
        self.process: Optional[subprocess.Popen] = None

    def open_odas(self) -> None:
        self.release_odas()
        try:
            self.process = subprocess.Popen(
                self.odas_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("ODAS started:", " ".join(self.odas_cmd))
            time.sleep(0.5)
        except FileNotFoundError:
            print("ODAS 启动失败：找不到", self.odas_cmd[0])
            self.process = None

    @staticmethod
    def release_odas() -> None:
        try:
            subprocess.run(
                ["pkill", "-f", "odaslive"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
        except FileNotFoundError:
            pass

    def close_odas(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=2)
            print("ODAS terminated")
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            print("ODAS killed")
        except OSError:
            pass
        finally:
            self.process = None


class sstprocess:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000, recv_size: int = 4096):
        self.host = host
        self.port = port
        self.recv_size = recv_size
        self._server: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._running = False
        self._buffer = ""
        self._latest: Optional[SoundSourceFrame] = None
        self._lock = threading.Lock()

    def connect(self, wait: bool = True) -> bool:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self._running = True
        target = self._accept_once
        target() if wait else threading.Thread(target=target, daemon=True).start()
        return self.is_connected()

    def is_connected(self) -> bool:
        with self._lock:
            return self._conn is not None

    def get_latest(self) -> Optional[SoundSourceFrame]:
        with self._lock:
            return self._latest

    def print_raw_loop(self) -> None:
        if self._server is None:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((self.host, self.port))
            self._server.listen(1)

        print(f"Waiting for ODAS SST on {self.host}:{self.port} ...")
        conn, addr = self._server.accept()
        print("SST connected:", addr)

        with conn:
            while True:
                data = conn.recv(self.recv_size)
                if not data:
                    break
                print(data.decode("utf-8", errors="ignore"), end="")

    def close(self) -> None:
        self._running = False
        for sock in (self._conn, self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        with self._lock:
            self._conn = None

    def _accept_once(self) -> None:
        if self._server is None:
            return
        print(f"Waiting for ODAS SST on {self.host}:{self.port} ...")
        conn, addr = self._server.accept()
        print("SST connected:", addr)
        with self._lock:
            self._conn = conn
        threading.Thread(target=self._receive_loop, daemon=True).start()

    def _receive_loop(self) -> None:
        while self._running and self._conn is not None:
            try:
                data = self._conn.recv(self.recv_size)
            except OSError:
                break
            if not data:
                break
            self._buffer += data.decode("utf-8", errors="ignore")
            self._parse_buffer()
        with self._lock:
            self._conn = None

    def _parse_buffer(self) -> None:
        decoder = json.JSONDecoder()
        while self._buffer:
            self._buffer = self._buffer.lstrip()
            start = self._buffer.find("{")
            if start < 0:
                self._buffer = ""
                return
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
        sources = []
        raw_sources = obj.get("src", [])
        if not isinstance(raw_sources, list):
            raw_sources = []
        for item in raw_sources[:4]:
            if isinstance(item, dict):
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
        return SoundSourceFrame(int(obj.get("timeStamp", 0)), sources, obj)

    @staticmethod
    def is_silent_frame(frame: Optional[SoundSourceFrame], threshold: float = 0.001) -> bool:
        return frame is None or all(src.activity <= threshold for src in frame.sources)


class sssprocess:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 10010,
        sr: int = 32000,
        ch: int = 4,
        keep_sec: int = 10,
    ):
        self.host = host
        self.port = port
        self.sr = sr
        self.ch = ch
        self.max_len = sr * keep_sec
        self.q = deque()
        self.n = 0
        self.lock = threading.Lock()
        self.server: Optional[socket.socket] = None
        self.running = False

    def start(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        self.running = True
        print(f"[SSS] listening on {self.host}:{self.port}")
        threading.Thread(target=self._receive_loop, daemon=True).start()

    def add(self, x: np.ndarray) -> None:
        with self.lock:
            self.q.append(x)
            self.n += len(x)
            while self.n > self.max_len:
                over = self.n - self.max_len
                first = self.q[0]
                if len(first) <= over:
                    self.q.popleft()
                    self.n -= len(first)
                else:
                    self.q[0] = first[over:]
                    self.n -= over

    def get_last(self, sec: int = 3) -> np.ndarray:
        with self.lock:
            blocks = list(self.q)
        if not blocks:
            return np.empty((0, self.ch), dtype=np.float32)
        return np.concatenate(blocks, axis=0)[-self.sr * sec :]

    def save_last_3s_wav(self, path: str = "birdnet_input.wav") -> Optional[str]:
        x = self.get_last(3)
        if len(x) < self.sr // 2:
            print("[SSS] no enough audio")
            return None

        rms = np.sqrt(np.mean(x * x, axis=0))
        best_ch = int(np.argmax(rms))
        pcm = (np.clip(x[:, best_ch], -1, 1) * 32767).astype(np.int16)

        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sr)
            f.writeframes(pcm.tobytes())

        #print(f"[SSS] saved {path}, ch={best_ch}, rms={np.round(rms, 5)}")
        return path

    def close(self) -> None:
        self.running = False
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass

    def _receive_loop(self) -> None:
        if self.server is None:
            return
        bytes_per_frame = self.ch * 2
        while self.running:
            try:
                conn, addr = self.server.accept()
            except OSError:
                break
            print("[SSS] connected:", addr)
            pending = bytearray()
            with conn:
                while self.running:
                    data = conn.recv(8192)
                    if not data:
                        print("[SSS] disconnected")
                        break
                    pending.extend(data)
                    n = len(pending) // bytes_per_frame * bytes_per_frame
                    if n:
                        raw = bytes(pending[:n])
                        del pending[:n]
                        x = np.frombuffer(raw, dtype="<i2")
                        self.add(x.reshape(-1, self.ch).astype(np.float32) / 32768.0)


if __name__ == "__main__":
    odas = odasconnecter()
    sst = sstprocess()
    odas.open_odas()
    try:
        sst.connect(wait=False)
        while True:
            frame = sst.get_latest()
            if frame is not None:
                print("timeStamp:", frame.time_stamp)
                for src in frame.sources:
                    print(
                        f"id={src.id}, x={src.x:.3f}, y={src.y:.3f}, "
                        f"z={src.z:.3f}, activity={src.activity:.3f}"
                    )
                print("-" * 40)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("程序退出")
    finally:
        sst.close()
        odas.close_odas()
