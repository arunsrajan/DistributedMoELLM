"""A worker (or `serve`) started on a port something already listens on must
fail, not share it: on Windows SO_REUSEADDR let a second worker come up on the
first one's port and take some of its connections."""

import socket
import socketserver
from http.server import BaseHTTPRequestHandler

import pytest

from moelan import server as http_server
from moelan import worker
from moelan.protocol import check_port_free


@pytest.fixture(params=["worker", "serve"])
def server_cls(request):
    if request.param == "worker":
        return worker._Server, socketserver.BaseRequestHandler
    return http_server._Server, BaseHTTPRequestHandler


def test_second_server_on_a_busy_port_is_refused(server_cls):
    cls, handler = server_cls
    first = cls(("127.0.0.1", 0), handler)
    try:
        port = first.server_address[1]
        with pytest.raises(OSError):
            cls(("127.0.0.1", port), handler).server_close()
    finally:
        first.server_close()


def test_a_program_asking_to_share_the_port_is_refused_too(server_cls):
    """An older moelan, or any program, binding with SO_REUSEADDR."""
    cls, handler = server_cls
    first = cls(("127.0.0.1", 0), handler)
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            s.bind(("127.0.0.1", first.server_address[1]))
            s.listen()
    finally:
        s.close()
        first.server_close()


def test_restart_on_the_same_port_after_a_connection(server_cls):
    """A stopped server's port comes back at once, even with the closed
    connection still in TIME_WAIT on the server's side."""
    cls, handler = server_cls
    first = cls(("127.0.0.1", 0), handler)
    port = first.server_address[1]
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = first.socket.accept()
    conn.close()                    # the server closes first
    client.recv(1)
    client.close()
    first.server_close()
    cls(("127.0.0.1", port), handler).server_close()


def test_early_check_names_the_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        with pytest.raises(SystemExit, match=f"port {port} on 127.0.0.1 is already in use"):
            check_port_free("127.0.0.1", port, "worker")
    finally:
        s.close()
    check_port_free("127.0.0.1", port, "worker")     # free again: no complaint
