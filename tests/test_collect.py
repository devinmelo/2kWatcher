import json

import numpy as np
import pytest

from twokwatcher.app.collect import FrameCollector
from twokwatcher.doctor import format_report, run_checks
from twokwatcher.pipeline import Event, EventBus


def _frame(value=90):
    return np.full((90, 160, 3), value, dtype=np.uint8)


def _preview(index=0, ts=0.0):
    return Event(kind="preview", frame_index=index, video_ts=ts,
                 data={"crops": {}, "frame": _frame(), "frame_size": (160, 90)})


def _state(current, previous="unknown"):
    return Event(kind="state_change", frame_index=0, video_ts=0.0,
                 data={"previous": previous, "current": current})


def test_frames_are_filed_by_screen(tmp_path):
    collector = FrameCollector(tmp_path, min_interval=0.0)
    bus = EventBus()
    collector.subscribe(bus)

    bus.publish(_state("live"))
    bus.publish(_preview(1))
    bus.publish(_state("post_game", "live"))
    bus.publish(_preview(2))

    assert (tmp_path / "live" / "live-000.png").exists()
    assert (tmp_path / "post_game" / "post_game-000.png").exists()
    assert collector.counts == {"live": 1, "post_game": 1}


def test_per_state_caps_are_respected(tmp_path):
    collector = FrameCollector(tmp_path, caps={"live": 3}, min_interval=0.0)
    bus = EventBus()
    collector.subscribe(bus)
    bus.publish(_state("live"))
    for i in range(20):
        bus.publish(_preview(i))
    assert collector.counts["live"] == 3


def test_interval_spreads_captures_out(tmp_path):
    """Thirty frames from one possession are worth less than three moments."""
    collector = FrameCollector(tmp_path, min_interval=999.0)
    bus = EventBus()
    collector.subscribe(bus)
    bus.publish(_state("live"))
    for i in range(10):
        bus.publish(_preview(i))
    # The transition lets one through; the interval blocks the rest.
    assert collector.counts["live"] == 1


def test_a_transition_always_gets_a_frame(tmp_path):
    """Transitions are the most informative moment, so they bypass the interval."""
    collector = FrameCollector(tmp_path, min_interval=999.0)
    bus = EventBus()
    collector.subscribe(bus)

    bus.publish(_state("live"))
    bus.publish(_preview(1))
    bus.publish(_state("post_game", "live"))
    bus.publish(_preview(2))
    bus.publish(_state("live", "post_game"))
    bus.publish(_preview(3))

    assert collector.counts == {"live": 2, "post_game": 1}


def test_disabled_collector_writes_nothing(tmp_path):
    collector = FrameCollector(tmp_path, enabled=False, min_interval=0.0)
    bus = EventBus()
    collector.subscribe(bus)
    bus.publish(_state("live"))
    bus.publish(_preview(1))
    assert collector.total == 0
    assert not (tmp_path / "live").exists()


def test_manifest_indexes_what_was_collected(tmp_path):
    collector = FrameCollector(tmp_path, min_interval=0.0)
    bus = EventBus()
    collector.subscribe(bus)
    bus.publish(_state("live"))
    bus.publish(_preview(7, ts=1.25))

    path = collector.flush_manifest()
    manifest = json.loads(path.read_text())
    assert manifest["counts"] == {"live": 1}
    entry = manifest["frames"][0]
    assert entry["state"] == "live"
    assert entry["frame_index"] == 7
    assert entry["size"] == [160, 90]


def test_manifest_is_skipped_when_nothing_collected(tmp_path):
    assert FrameCollector(tmp_path).flush_manifest() is None


def test_doctor_reports_and_flags_blocking_failures(tmp_path):
    checks = run_checks(config_path=tmp_path / "nope.yaml",
                        db_path=tmp_path / "d.db")
    names = {c.name for c in checks}
    assert {"Python", "Glyph atlas", "Essential regions"} <= names

    report, blocking = format_report(checks)
    assert "PASS" in report
    # A missing atlas is expected, so it must warn rather than block.
    atlas = next(c for c in checks if c.name == "Glyph atlas")
    assert atlas.warning and not atlas.ok
    assert isinstance(blocking, bool)


def test_every_failing_check_says_what_to_do(tmp_path):
    for check in run_checks(config_path=tmp_path / "nope.yaml",
                            db_path=tmp_path / "d.db"):
        if not check.ok:
            assert check.fix, f"{check.name} fails without telling you the fix"
