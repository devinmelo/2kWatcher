"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, Config, Region
from .pipeline import Event, EventBus, Runner
from .storage import Database


def _source_from_args(args, config: Config):
    from .capture import open_source

    if args.video:
        return open_source("file", video_path=args.video)
    index = args.device if args.device is not None else config.capture.get(
        "device_index", 0
    )
    return open_source("virtualcam", device_index=index)


def cmd_devices(args) -> int:
    """Probe video device indices to find the OBS virtual camera."""
    from .capture import list_devices

    devices = list_devices()
    if not devices:
        print("No working video devices found.")
        print("Start OBS and enable 'Start Virtual Camera', then try again.")
        return 1
    for d in devices:
        print(f"  [{d['index']}]  {d['width']}x{d['height']}  "
              f"{d['fps']:.0f}fps" if d["fps"] else
              f"  [{d['index']}]  {d['width']}x{d['height']}")
    return 0


def cmd_calibrate(args) -> int:
    """Drag a box on a frame to define a HUD region."""
    import cv2

    config = Config.load(args.config)
    with _source_from_args(args, config) as source:
        frame = None
        # Skip a few frames; the first from a virtual camera is often garbage.
        for _ in range(args.skip + 1):
            frame = source.read()
            if frame is None:
                print("Could not read a frame from the source.", file=sys.stderr)
                return 1

    w, h = frame.size
    print(f"Frame is {w}x{h}. Drag a box around '{args.region}', "
          f"then press ENTER (or C to cancel).")
    box = cv2.selectROI(f"calibrate: {args.region}", frame.image,
                        showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, bw, bh = box
    if bw == 0 or bh == 0:
        print("Cancelled, nothing written.")
        return 1

    config.regions[args.region] = Region(
        name=args.region, x=x / w, y=y / h, w=bw / w, h=bh / h
    )
    path = config.save(args.config or DEFAULT_CONFIG_PATH)
    print(f"Wrote '{args.region}' = "
          f"({x/w:.4f}, {y/h:.4f}, {bw/w:.4f}, {bh/h:.4f}) to {path}")
    return 0


def cmd_probe(args) -> int:
    """Print per-frame classifier signals, for tuning thresholds."""
    from .state.detectors import ScreenClassifier

    config = Config.load(args.config)
    classifier = ScreenClassifier(config)

    print(f"{'frame':>8}  {'state':<10}  {'edges':>7}  {'luma':>6}  {'clock':>5}")
    with _source_from_args(args, config) as source:
        for frame in source:
            if frame.index % args.stride:
                continue
            state, sig = classifier.classify(frame.image)
            print(f"{frame.index:>8}  {state.value:<10}  "
                  f"{sig.scoreboard_edge_density:>7.4f}  {sig.mean_luma:>6.1f}  "
                  f"{'yes' if sig.clock_changed else '-':>5}")
            if args.limit and frame.index >= args.limit:
                break
    return 0


def cmd_record(args) -> int:
    """Record raw footage to a file, for offline parser development."""
    import cv2

    config = Config.load(args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _source_from_args(args, config) as source:
        first = source.read()
        if first is None:
            print("Could not read from the source.", file=sys.stderr)
            return 1
        w, h = first.size
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), source.fps, (w, h)
        )
        try:
            writer.write(first.image)
            written = 1
            print(f"Recording {w}x{h} to {out_path} — Ctrl-C to stop.")
            for frame in source:
                writer.write(frame.image)
                written += 1
                if args.seconds and written >= args.seconds * source.fps:
                    break
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            writer.release()
    print(f"Wrote {written} frames to {out_path}")
    return 0


def cmd_snapshot(args) -> int:
    """Grab still frames from the live feed.

    Lighter than `record` and usually all that is needed: calibration wants a
    few clean stills of the HUD, not minutes of video.
    """
    import time

    import cv2

    config = Config.load(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    written = []
    with _source_from_args(args, config) as source:
        for n in range(args.count):
            if n and args.interval:
                # Space shots out so they capture different game situations
                # rather than three copies of the same moment.
                deadline = time.monotonic() + args.interval
                while time.monotonic() < deadline:
                    if source.read() is None:
                        break
            # Discard a few frames first; the first off a virtual camera is
            # frequently blank or half-composited.
            frame = None
            for _ in range(args.skip + 1):
                frame = source.read()
            if frame is None:
                break
            path = out_dir / f"{stamp}-{n:02d}.png"
            cv2.imwrite(str(path), frame.image)
            written.append(path)
            print(f"  {path}  ({frame.size[0]}x{frame.size[1]})")

    if not written:
        print("Could not read from the source.", file=sys.stderr)
        return 1
    print(f"\nWrote {len(written)} snapshot(s) to {out_dir}")
    return 0


def cmd_run(args) -> int:
    """Watch gameplay and log what happens."""
    config = Config.load(args.config)
    bus = EventBus()

    with Database(args.db) as db:
        session_id = db.start_session(source="file" if args.video else "virtualcam")

        def on_state(event: Event) -> None:
            db.log_state(session_id, event.data["previous"], event.data["current"],
                         event.frame_index, event.video_ts)
            print(f"[{event.video_ts:8.1f}s] {event.data['previous']} "
                  f"-> {event.data['current']}")

        def on_scoreboard(event: Event) -> None:
            d = event.data
            print(f"[{event.video_ts:8.1f}s] {d['score_home']}-{d['score_away']} "
                  f"clock {d['game_clock']} shot {d['shot_clock']}")

        bus.subscribe("state_change", on_state)
        if args.verbose:
            bus.subscribe("scoreboard", on_scoreboard)

        try:
            with _source_from_args(args, config) as source:
                stats = Runner(source, config, bus=bus).run(max_frames=args.limit)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0
        finally:
            db.end_session(session_id)

    print(f"\nSaw {stats.frames_seen} frames, sampled {stats.frames_sampled}, "
          f"{stats.transitions} transitions.")
    for state, count in sorted(stats.state_frames.items(),
                               key=lambda kv: -kv[1]):
        print(f"  {state:<10} {count:>6}")
    return 0


def cmd_doctor(args) -> int:
    """Check the setup before a session, rather than during one."""
    from .doctor import format_report, run_checks

    report, blocking = format_report(run_checks(args.config, args.db))
    print(report)
    return 1 if blocking else 0


def cmd_app(args) -> int:
    """Open the desktop app."""
    from .app.desktop import launch

    return launch(Config.load(args.config), args.db, port=args.port,
                  window=not args.browser,
                  config_path=args.config or DEFAULT_CONFIG_PATH,
                  collect_dir=args.collect_dir, collect=not args.no_collect)


def cmd_roster(args) -> int:
    """Show or edit the player registry."""
    with Database(args.db) as db:
        if args.add:
            player_id = db.upsert_player(
                args.add, display_name=args.name,
                is_me=args.me, is_friend=not args.me,
            )
            print(f"Registered {args.add} as player {player_id}")
        rows = db.roster()
        if not rows:
            print("Roster is empty. Add yourself:  2kw roster --add <gamertag> --me")
            return 0
        for r in rows:
            tag = "me" if r["is_me"] else ("friend" if r["is_friend"] else "")
            print(f"  {r['gamertag']:<24} {r['display_name'] or '':<20} {tag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Options every subcommand accepts. Defined on a parent parser rather than
    # the top level so they work *after* the subcommand, which is how anyone
    # actually types them: `2kw run --db foo.db`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help="path to regions.yaml")
    common.add_argument("--db", type=Path, default=Path("data/2kwatcher.db"))
    common.add_argument("-v", "--verbose", action="store_true")

    # Source selection, for the subcommands that read frames.
    src = argparse.ArgumentParser(add_help=False)
    src.add_argument("--device", type=int, help="video device index")
    src.add_argument("--video", type=Path, help="read a recording instead of live")

    parser = argparse.ArgumentParser(
        prog="2kw", description="Watch NBA 2K gameplay and turn it into stats."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", parents=[common],
                   help="list video devices").set_defaults(func=cmd_devices)

    p = sub.add_parser("calibrate", parents=[common, src],
                       help="define a HUD region by dragging a box")
    p.add_argument("--region", required=True, help="region name, e.g. scoreboard")
    p.add_argument("--skip", type=int, default=10,
                   help="frames to discard before grabbing one")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("probe", parents=[common, src],
                       help="dump classifier signals for threshold tuning")
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("record", parents=[common, src], help="record footage to a file")
    p.add_argument("--output", type=Path, default="captures/session.mp4")
    p.add_argument("--seconds", type=int, default=0, help="0 means until Ctrl-C")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("snapshot", parents=[common, src],
                       help="grab still frames from the live feed")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--interval", type=float, default=3.0,
                   help="seconds between shots")
    p.add_argument("--skip", type=int, default=10,
                   help="frames to discard before each shot")
    p.add_argument("--output", type=Path, default=Path("data/snapshots"))
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("run", parents=[common, src], help="watch and log gameplay")
    p.add_argument("--limit", type=int, default=0, help="stop after N samples")
    p.set_defaults(func=cmd_run)

    sub.add_parser("doctor", parents=[common],
                   help="check the setup before a session").set_defaults(
        func=cmd_doctor)

    p = sub.add_parser("app", parents=[common],
                       help="open the desktop app (default way to use this)")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--browser", action="store_true",
                   help="open in your browser instead of a native window")
    p.add_argument("--collect-dir", type=Path, default=Path("data/collect"))
    p.add_argument("--no-collect", action="store_true",
                   help="do not save frames for calibration")
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("roster", parents=[common], help="show or edit the player registry")
    p.add_argument("--add", metavar="GAMERTAG")
    p.add_argument("--name", help="display name for --add")
    p.add_argument("--me", action="store_true", help="mark --add as yourself")
    p.set_defaults(func=cmd_roster)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
