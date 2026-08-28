# 2kWatcher

Watches NBA 2K gameplay over a capture card and turns it into stats.

Xbox → Elgato (split) → monitor for play, PC for analysis. Because the console
does the rendering and the monitor runs off HDMI passthrough, the PC is a
dedicated analysis box: nothing this program does can affect how the game plays
or feels.

Everything runs locally. No cloud, no API calls, no recurring cost.

## Why it's built this way

**It's video only.** The Xbox exposes no memory and no API, so every fact this
program knows comes from pixels. That's the constraint that shapes everything.

**The state machine comes first.** Knowing what screen you're on gates all the
expensive work. Partly that's compute, but mostly it's data quality — frames
from replays, cutscenes and timeouts produce plausible-looking garbage, and
excluding them at the source beats filtering them out of the database later. The
strongest signal for "live play" is the game clock decrementing, which costs one
small crop per sample.

**OBS owns the capture card.** The Elgato driver generally allows one consumer,
and OBS is going to want it anyway — for the replay buffer that will power
highlight clipping and for NVENC recording that costs no CUDA. So OBS holds the
card and this reads its virtual camera.

**Template matching, not OCR.** The scoreboard is one fixed font at one fixed
size drawing eleven glyphs. General OCR is slower and less reliable than
matching against a glyph atlas built from your own footage.

**Court coordinates, not pixels.** When the tracker lands, positions get stored
in feet from centre court. Pixel positions are meaningless once the camera pans.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

In OBS: add your Elgato as a source, then **Start Virtual Camera**.

```bash
2kw devices                                  # find the virtual camera's index
cp config/regions.example.yaml config/regions.yaml
```

## Calibrating

**The shipped region coordinates are placeholders.** 2K moves its HUD every
year, so they must be fitted to your own capture before anything downstream
works. Drag a box on a live frame for each region:

```bash
2kw calibrate --region scoreboard
2kw calibrate --region game_clock
2kw calibrate --region shot_feedback
```

Then tune the classifier thresholds against real footage:

```bash
2kw record --output captures/rec-game.mp4 --seconds 300
2kw probe --video captures/rec-game.mp4 --stride 30
```

`probe` prints the raw measurements per frame. Find where menu frames and
in-game frames actually separate on `edges`, and set `edge_threshold` in
`state/detectors.py` accordingly. The shipped value of `0.045` is a guess.

## Running

```bash
2kw roster --add <your-gamertag> --me
2kw roster --add <friend-gamertag>

2kw run                                      # live
2kw run --video captures/rec-game.mp4        # against a recording
```

Develop parsers against recordings, not live games. You cannot iterate on a
parser and play at the same time, and recordings are repeatable.

## Layout

```
capture/    frame sources — virtual camera, video file, swappable
state/      screen classification and the debounced state machine
hud/        scoreboard and HUD parsing
pipeline/   the frame loop and the event bus stages attach to
storage/    SQLite schema and access
```

The frame loop publishes events and knows nothing about the database, the
dashboard, or the tracker — those attach as subscribers. Adding the tracker
should mean registering another stage, not restructuring the loop.

## What works now

- Frame capture from the OBS virtual camera or a recording
- Screen-state classification and debounced transitions, logged to SQLite
- Region calibration and threshold-tuning tools
- Player registry keyed on gamertag
- Scoreboard reader — structurally complete, needs a glyph atlas to produce values

## What's next

1. **Glyph atlas** (`2kw atlas`) so the scoreboard actually reads.
2. **Shot feedback logging** — every release with its timing verdict and
   outcome. The highest-value signal in the project, and the one 2K gives you
   no way to analyze.
3. **Post-game box score parsing.** In Rec, one screenshot yields all ten
   gamertags with full stat lines. Dense, structured, no real-time CV needed —
   and it doubles as ground truth for grading everything else.
4. **Squad analytics.** Once box scores are keyed to the roster, lineup +/- for
   your friend group falls out of a GROUP BY.
5. **Highlight clipping** via obs-websocket triggering OBS's replay buffer.
6. **Player tracking.** Court homography first, then detection, then tracking in
   court space — in that order, because homography makes tracking easier rather
   than the reverse.
