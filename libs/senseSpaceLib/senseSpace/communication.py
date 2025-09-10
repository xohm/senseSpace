"""
senseSpaceLib.senseSpace.communication

Shared TCP communication utilities for senseSpace clients and servers.
"""

import socket
import threading
import json
from typing import Callable, Any

class TCPServer:
    def __init__(self, host: str, port: int, on_data: Callable[[dict, socket.socket], None]):
        self.host = host
        self.port = port
        self.on_data = on_data
        self.clients = []
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen()
        while self.running:
            conn, addr = s.accept()
            self.clients.append(conn)
            threading.Thread(target=self._client_handler, args=(conn, addr), daemon=True).start()

    def _client_handler(self, conn, addr):
        try:
            while self.running:
                data = conn.recv(4096)
                if not data:
                    break
                try:
                    msg = json.loads(data.decode('utf-8'))
                    self.on_data(msg, conn)
                except Exception:
                    pass
        finally:
            if conn in self.clients:
                self.clients.remove(conn)
            conn.close()

    def broadcast(self, data: dict):
        msg = (json.dumps(data) + "\n").encode("utf-8")
        for c in list(self.clients):
            try:
                c.sendall(msg)
            except:
                self.clients.remove(c)

class TCPClient:
    def __init__(self, host: str, port: int, on_data: Callable[[dict], None]):
        self.host = host
        self.port = port
        self.on_data = on_data
        self.sock = None
        self.running = False

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.running = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while self.running:
            data = self.sock.recv(4096)
            if not data:
                break
            try:
                msg = json.loads(data.decode('utf-8'))
                self.on_data(msg)
            except Exception:
                pass

    def send(self, data: dict):
        msg = (json.dumps(data) + "\n").encode("utf-8")
        self.sock.sendall(msg)

    def close(self):
        self.running = False
        if self.sock:
            self.sock.close()
