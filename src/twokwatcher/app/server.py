"""The local HTTP server behind the app window.

Deliberately built on the standard library rather than FastAPI or Flask. There
is exactly one user on one machine, so a threading HTTP server with JSON
endpoints and a polling UI is entirely adequate — and it keeps the dependency
surface small, which matters when this gets frozen into a single .exe.

It binds to localhost only. Nothing here is authenticated, because nothing off
this machine can reach it.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ..capture import list_devices
from ..config import Config
from ..storage import Database
from .live import LiveState
from .watcher import WatcherThread

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class AppServer:
    def __init__(self, config: Config, db_path, *, host: str = "127.0.0.1",
                 port: int = 8770) -> None:
        self.config = config
        self.db_path = db_path
        self.live = LiveState()
        self.watcher = WatcherThread(config, self.live, db_path)
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def serve_forever(self) -> None:
        self._httpd = ThreadingHTTPServer(
            (self.host, self.port), _handler_for(self))
        self._httpd.serve_forever()

    def start_background(self) -> None:
        self._httpd = ThreadingHTTPServer(
            (self.host, self.port), _handler_for(self))
        threading.Thread(target=self._httpd.serve_forever, daemon=True,
                         name="http").start()

    def shutdown(self) -> None:
        self.watcher.stop()
        if self._httpd is not None:
            self._httpd.shutdown()


def _handler_for(app: AppServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass  # The access log is noise for a single-user local app.

        # --- helpers ---------------------------------------------------

        def _json(self, payload, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            return json.loads(self.rfile.read(length) or b"{}")

        def _static(self, name: str) -> None:
            path = (STATIC_DIR / name).resolve()
            # Refuse anything that escapes the static directory.
            if not path.is_file() or STATIC_DIR.resolve() not in path.parents:
                self._json({"error": "not found"}, 404)
                return
            body = path.read_bytes()
            kind = {"html": "text/html", "css": "text/css",
                    "js": "application/javascript"}.get(
                        path.suffix.lstrip("."), "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", f"{kind}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # --- routes ----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                return self._static("index.html")
            if route.startswith("/static/"):
                return self._static(route[len("/static/"):])
            if route == "/api/status":
                return self._json(app.live.snapshot())
            if route == "/api/devices":
                return self._json({"devices": list_devices()})
            if route == "/api/regions":
                return self._json({
                    "regions": {n: {"x": r.x, "y": r.y, "w": r.w, "h": r.h}
                                for n, r in app.config.regions.items()}})
            if route == "/api/history":
                with Database(app.db_path) as db:
                    return self._json({
                        "games": [dict(r) for r in db.recent_games()],
                        "roster": [dict(r) for r in db.roster()],
                        "corrections": [dict(r) for r in db.corrections()],
                    })
            return self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            try:
                payload = self._read_json()
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)

            if route == "/api/watch/start":
                app.watcher.device_index = payload.get("device_index")
                app.watcher.video_path = payload.get("video_path")
                started = app.watcher.start()
                return self._json({"started": started,
                                   "running": app.watcher.is_running})
            if route == "/api/watch/stop":
                app.watcher.stop()
                return self._json({"running": app.watcher.is_running})

            if route == "/api/correct":
                corrected = str(payload.get("corrected", "")).strip()
                if not corrected:
                    return self._json({"error": "corrected is required"}, 400)
                crop_b64 = payload.get("crop")
                crop = None
                if crop_b64:
                    try:
                        crop = base64.b64decode(crop_b64)
                    except (ValueError, TypeError):
                        crop = None
                with Database(app.db_path) as db:
                    row_id = db.record_correction(
                        kind=str(payload.get("kind", "scoreboard")),
                        field=payload.get("field"),
                        observed=payload.get("observed"),
                        corrected=corrected,
                        frame_index=payload.get("frame_index"),
                        crop_png=crop,
                    )
                app.live.note("correction",
                              f"{payload.get('field') or 'value'}: "
                              f"{payload.get('observed')!r} → {corrected!r}")
                return self._json({"id": row_id})

            if route == "/api/roster":
                tag = str(payload.get("gamertag", "")).strip()
                if not tag:
                    return self._json({"error": "gamertag is required"}, 400)
                with Database(app.db_path) as db:
                    pid = db.upsert_player(
                        tag, display_name=payload.get("display_name"),
                        is_me=bool(payload.get("is_me")),
                        is_friend=not bool(payload.get("is_me")))
                return self._json({"id": pid})

            return self._json({"error": "not found"}, 404)

    return Handler
