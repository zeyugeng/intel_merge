"""ODAS TCP bridge.

ODAS writes streaming JSON to ``odas_port``. Python modules consume newline
delimited, flattened sound-source JSON from ``python_port``.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any, Dict, List, Optional, Tuple

from .sound_format import normalize_sound_source


def extract_json_objects(buffer: str) -> Tuple[List[Dict[str, Any]], str]:
    """Parse complete JSON objects from a streaming text buffer."""
    objects: List[Dict[str, Any]] = []
    start = 0
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(buffer):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = buffer[start : i + 1]
                try:
                    objects.append(json.loads(chunk))
                except json.JSONDecodeError:
                    pass

    if depth == 0:
        return objects, ""
    return objects, buffer[start:]


def flatten_odas_payload(payload: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Convert raw ODAS JSON to the flat payload consumed by Python modules."""
    source = normalize_sound_source(payload)
    if source is None:
        return None
    return source.as_payload()


class OdasBridge:
    """Forward ODAS sound-source JSON to Python clients."""

    def __init__(self, host: str, odas_port: int, python_port: int):
        self.host = host
        self.odas_port = odas_port
        self.python_port = python_port
        self._clients: List[socket.socket] = []
        self._clients_lock = threading.Lock()
        self._latest_line: Optional[str] = None
        self._running = False
        self._odas_server: Optional[socket.socket] = None
        self._python_server: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
        self._ready = threading.Event()

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._ready.clear()
        self._threads = [
            threading.Thread(target=self._serve_odas, daemon=True),
            threading.Thread(target=self._serve_python, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

        self._ready.wait(timeout=2.0)
        print(
            f"桥接服务已启动: ODAS → {self.host}:{self.odas_port}, "
            f"Python → {self.host}:{self.python_port}"
        )

    def stop(self) -> None:
        self._running = False

        for server in (self._odas_server, self._python_server):
            if server is not None:
                try:
                    server.close()
                except OSError:
                    pass

        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass

        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

    def _broadcast(self, line: str) -> None:
        self._latest_line = line
        data = (line + "\n").encode("utf-8")
        dead: List[socket.socket] = []
        with self._clients_lock:
            for client in self._clients:
                try:
                    client.sendall(data)
                except OSError:
                    dead.append(client)
            for client in dead:
                self._clients.remove(client)
                client.close()

    def _handle_odas_connection(self, conn: socket.socket, addr) -> None:
        print(f"ODAS 已连接: {addr}")
        buffer = ""
        try:
            while self._running:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                objects, buffer = extract_json_objects(buffer)
                for obj in objects:
                    flat = flatten_odas_payload(obj)
                    if flat is None:
                        continue
                    line = json.dumps(flat, ensure_ascii=False)
                    self._broadcast(line)
                    print(
                        f"声源 x={flat['x']:.3f}, y={flat['y']:.3f}, "
                        f"z={flat['z']:.3f}, E={flat['E']:.3f}"
                    )
        except OSError as exc:
            if self._running:
                print(f"ODAS 连接断开: {exc}")
        finally:
            conn.close()
            print("ODAS 连接已关闭")

    def _serve_odas(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._odas_server = server
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.odas_port))
        server.listen(1)
        server.settimeout(1.0)
        self._ready.set()
        print(f"等待 ODAS 连接 {self.host}:{self.odas_port} ...")

        while self._running:
            try:
                conn, addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_odas_connection, args=(conn, addr), daemon=True
            ).start()

    def _handle_python_client(self, conn: socket.socket, addr) -> None:
        print(f"Python 客户端已连接: {addr}")
        with self._clients_lock:
            self._clients.append(conn)
        if self._latest_line is not None:
            try:
                conn.sendall((self._latest_line + "\n").encode("utf-8"))
            except OSError:
                pass
        try:
            while self._running:
                if not conn.recv(1024):
                    break
        except OSError:
            pass
        finally:
            with self._clients_lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            conn.close()
            print(f"Python 客户端断开: {addr}")

    def _serve_python(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._python_server = server
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.python_port))
        server.listen(5)
        server.settimeout(1.0)
        self._ready.set()
        print(f"等待 Python 客户端 {self.host}:{self.python_port} ...")

        while self._running:
            try:
                conn, addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_python_client, args=(conn, addr), daemon=True
            ).start()
