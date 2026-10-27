"""Zero-config LAN discovery with UDP broadcast beacons.

Workers broadcast a small JSON beacon every couple of seconds; a coordinator
listens for a while and collects the workers it heard. No mDNS daemon or extra
package is needed, which keeps it working on stock Windows, macOS and Linux.
Routers do not forward broadcasts, so discovery is limited to one subnet; use
``--workers host:port,...`` across subnets.
"""

from __future__ import annotations

import json
import socket
import threading
import time

DISCOVERY_PORT = 47474
SERVICE = "moelan"


def local_ip() -> str:
    """Best guess at this machine's LAN address (no packet is actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Beacon(threading.Thread):
    def __init__(self, info: dict, port: int = DISCOVERY_PORT, interval: float = 2.0):
        super().__init__(daemon=True, name="moelan-beacon")
        self.info = info
        self.port = port
        self.interval = interval
        self._stop = threading.Event()

    def run(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            while not self._stop.is_set():
                msg = json.dumps(dict(self.info, service=SERVICE, ts=time.time())).encode()
                for addr in ("255.255.255.255", "127.0.0.1"):
                    try:
                        s.sendto(msg, (addr, self.port))
                    except OSError:
                        pass
                self._stop.wait(self.interval)
        finally:
            s.close()

    def stop(self) -> None:
        self._stop.set()


def discover(timeout: float = 3.0, port: int = DISCOVERY_PORT) -> list[dict]:
    """Listen for worker beacons for ``timeout`` seconds."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", port))
    s.settimeout(0.25)
    found: dict[str, dict] = {}
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                data, (src, _) = s.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                info = json.loads(data)
            except ValueError:
                continue
            if info.get("service") != SERVICE or "port" not in info:
                continue
            host = info.get("host") or src
            if host in ("0.0.0.0", ""):
                host = src
            info["host"] = host
            found[f"{host}:{info['port']}"] = info
    finally:
        s.close()
    return sorted(found.values(), key=lambda i: (i["host"], i["port"]))
