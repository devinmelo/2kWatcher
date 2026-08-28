import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from twokwatcher.app.live import LiveState
from twokwatcher.app.server import AppServer
from twokwatcher.config import Config
from twokwatcher.pipeline import Event, EventBus
from twokwatcher.storage import Database


@pytest.fixture
def server(tmp_path):
    app = AppServer(Config.load(), tmp_path / "app.db", port=0)
    app.start_background()
    # port=0 lets the OS pick; read back what it actually bound.
    app.port = app._httpd.server_address[1]
    yield app
    app.shutdown()


def get(app, path):
    with urllib.request.urlopen(f"{app.url.rstrip('/')}{path}", timeout=5) as r:
        return json.loads(r.read())


def post(app, path, payload):
    req = urllib.request.Request(
        f"{app.url.rstrip('/')}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_status_reports_idle_before_watching(server):
    status = get(server, "/api/status")
    assert status["running"] is False
    assert status["state"] == "unknown"
    assert status["previews"] == {}


def test_index_is_served(server):
    with urllib.request.urlopen(server.url, timeout=5) as r:
        body = r.read().decode()
    assert "2kWatcher" in body
    assert r.headers["Content-Type"].startswith("text/html")


def test_static_traversal_is_refused(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server, "/static/../../../etc/passwd")
    assert exc.value.code == 404


def test_correction_persists_with_its_crop(server, tmp_path):
    crop = np.full((20, 60, 3), 200, dtype=np.uint8)
    encoded = base64.b64encode(cv2.imencode(".jpg", crop)[1].tobytes()).decode()

    result = post(server, "/api/correct", {
        "kind": "scoreboard", "field": "score_home",
        "observed": "B8", "corrected": "88", "crop": encoded,
    })
    assert result["id"] > 0

    with Database(server.db_path) as db:
        row = db.conn.execute("SELECT * FROM corrections").fetchone()
    assert row["observed"] == "B8"
    assert row["corrected"] == "88"
    # The crop is the point: a correction without it is not a training example.
    assert row["crop_png"] and len(row["crop_png"]) > 0


def test_correction_requires_a_value(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server, "/api/correct", {"field": "score_home", "corrected": "  "})
    assert exc.value.code == 400


def test_roster_round_trips_through_the_api(server):
    post(server, "/api/roster", {"gamertag": "devinmelo", "is_me": True})
    history = get(server, "/api/history")
    assert [p["gamertag"] for p in history["roster"]] == ["devinmelo"]
    assert history["roster"][0]["is_me"] == 1


def test_live_state_encodes_previews_for_the_ui():
    live = LiveState()
    bus = EventBus()
    live.subscribe(bus)

    bus.publish(Event(kind="preview", frame_index=7, video_ts=1.0, data={
        "crops": {"game_clock": np.full((16, 48, 3), 120, dtype=np.uint8),
                  "empty": np.zeros((0, 0, 3), dtype=np.uint8)},
        "frame_size": (1920, 1080),
    }))
    snap = live.snapshot()
    assert "game_clock" in snap["previews"]
    # A zero-size crop must be skipped, not emitted as a broken image.
    assert "empty" not in snap["previews"]
    assert base64.b64decode(snap["previews"]["game_clock"])


def test_live_state_tracks_transitions_and_scoreboard():
    live = LiveState()
    bus = EventBus()
    live.subscribe(bus)

    bus.publish(Event(kind="state_change", frame_index=1, video_ts=0.0,
                      data={"previous": "menu", "current": "live"}))
    bus.publish(Event(kind="scoreboard", frame_index=2, video_ts=0.1,
                      data={"score_home": 88, "score_away": 74}))

    snap = live.snapshot()
    assert snap["state"] == "live"
    assert snap["scoreboard"]["score_home"] == 88
    assert "menu → live" in snap["events"][0]["message"]
