from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .state import AppState

logger = logging.getLogger(__name__)


class _StatefulHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        state: AppState,
    ) -> None:
        self.state = state
        super().__init__(server_address, request_handler_class)


class _HealthRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            path = urlsplit(self.path).path
            if path == "/healthz":
                self._send_text(status_code=200, body="ok\n")
                return
            if path == "/readyz":
                if self._state().is_ready():
                    self._send_text(status_code=200, body="ready\n")
                else:
                    self._send_text(status_code=503, body="not ready\n")
                return
            if path == "/status":
                self._send_json(status_code=200, payload=self._state().to_status())
                return
            self._send_text(status_code=404, body="not found\n")
        except Exception:
            logger.exception("health server failed", extra={"path": self.path})
            self._send_text(status_code=500, body="internal server error\n")

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("health request", extra={"message": format % args})

    def _send_text(self, status_code: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _state(self) -> AppState:
        server = self.server
        if not isinstance(server, _StatefulHTTPServer):
            raise RuntimeError("health handler is not attached to a stateful server")
        return server.state


class HealthServer:
    def __init__(self, host: str, port: int, state: AppState) -> None:
        self._host = host
        self._port = port
        self._state = state
        self._server: _StatefulHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self._port
        return int(self._server.server_port)

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            server = _StatefulHTTPServer(
                (self._host, self._port), _HealthRequestHandler, self._state
            )
            thread = threading.Thread(
                target=server.serve_forever,
                name="hf-cache-preloader-health",
                daemon=True,
            )
            thread.start()
        except Exception:
            logger.exception("health server failed", extra={"host": self._host, "port": self._port})
            raise
        self._server = server
        self._thread = thread
        logger.info("health server started", extra={"host": self._host, "port": self.port})

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        self._server = None
        self._thread = None
        logger.info("health server stopped", extra={"host": self._host, "port": self._port})
