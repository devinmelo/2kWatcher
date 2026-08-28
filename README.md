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

## The app

`2kw app` opens a native window (pywebview) hosting a local server on
127.0.0.1. That window is the normal way to use this: open it, hit **Start
watching**, play.

The main panel shows **the actual pixels each parser is looking at, beside what
it read**. That is the point of it — a wrong number on its own tells you
nothing about whether the crop, the threshold or the atlas is at fault, but the
crop next to the value makes it obvious. Click any value to correct it.

Corrections are stored **with the crop that caused them**, which is what turns
a fix into a labelled training example. Correcting the app while using it is
how the glyph atlas and, later, the models improve.

Because it is a local server behind a native window rather than a bundled
frontend, the same UI is reachable from a phone or tablet, and later from OBS
as a browser source, with no separate build. Without pywebview installed it
falls back to opening a browser tab rather than refusing to start.

```bash
pip install -e ".[app]"
2kw app
2kw app --browser      # skip the native window
```

## Shot feedback

Every shot puts a banner at the top of the screen: release timing, how well you
were contested, and the shot distance. It is the most valuable signal in the
game and 2K gives you no way to analyse it — it shows for about a second and is
gone. Logged over a session it answers the questions players actually have:
whether your releases drift late as the night goes on, which jumpshot greens for
you, whether you shoot worse contested from the wing than the corner.

Three things shape the parser:

- **The banner is centre-justified with a variable panel count** (two panels
  without a distance reading, three with), so panel positions cannot be
  hard-coded. A generous band is read whole and the values are located by
  content, not position.
- **The values come from a closed vocabulary**, which makes OCR accuracy much
  less critical than it looks: "UGHT CONTEST" and "GXCELLENT" both resolve by
  fuzzy match. Anything matching nothing is kept verbatim and flagged, so an
  unseen verdict surfaces as unknown instead of becoming the nearest known one.
- **One frame is not trusted; the event is.** The banner stays up for dozens of
  frames, so a shot is read by consensus across them. On low-resolution footage
  most individual frames abstain — but the ones that read are correct, and a
  value is only accepted when it wins by a margin.

Presence detection is gated on edge density in the band, calibrated against a
real recording to produce no false positives. The banner animates in, so the
first frame or two of an event fall under the threshold; that costs nothing,
since an event only has to be noticed once.

## Post-game box score

`2kw boxscore <image>` parses the MyCareer recap GAME STATS screen: ten
players, gamertags, grades and full stat lines from one frame.

This is the densest structured data 2K exposes, and it is worth more per unit
of effort than any amount of real-time vision. Two properties of the screen do
most of the work:

- **A green triangle marks your row and a red one your matchup.** Identity is
  read off the screen rather than configured, so it stays correct when you
  change build, team or mode. AI-filled slots are detected by the absence of a
  platform icon — a structural signal, unlike the "AI Player" label, which OCR
  renders "Al Player" about as often.
- **The TOTAL row is a checksum.** Player rows must sum to it for PTS, REB,
  AST, STL, BLK and the made/attempted fractions, which validates a parse for
  free. FOULS and TO are deliberately excluded: 2K reports team fouls and team
  turnovers there, and those genuinely differ from the sum of the rows.

Unlike the live scoreboard, this is read once per game, so OCR is affordable —
and necessary, since gamertags are arbitrary strings that templates cannot
cover. OCR still confuses a few glyph pairs, so tags are snapped onto the
player registry by fuzzy match: a gamertag only has to be read correctly, and
confirmed, once.

Cells that cannot be read come back as `None`, never as a guess, and are
listed so the app can ask. A parse is `trustworthy` only when nothing is unread
and every checksum passes.

Needs OCR:

```bash
pip install -e ".[ocr]"
# plus the binary: apt install tesseract-ocr  /  choco install tesseract
```

## Frame collection

The app saves a bounded, spread-out sample of frames to `data/collect/`, sorted
into a folder per screen, and writes a `manifest.json` alongside.

This is on by default because early sessions are worth far more as a labelled
frame set than as a database. Every parser not yet written — the glyph atlas,
the scoreboard reader, the box score parser — is blocked on nobody having seen
this particular HUD, and those can all be built offline from collected frames.
Play a normal session and you finish the night with the calibration set.

Caps are per-screen (post-game gets the largest budget, being the densest
structured data in the game), and a minimum interval keeps a session from
filling up with thirty near-identical frames of one possession. State
transitions bypass that interval, since the moment a screen changes is the most
informative one to catch.

```bash
2kw app --no-collect              # turn it off
2kw app --collect-dir path/to/x   # put it somewhere else
```

## Live vs. recorded

Normal use is live. `2kw run` reads the virtual camera in real time while you
play and logs as it goes — no footage involved.

Recordings exist for *building* the parsers, not for using them. Tuning a
threshold means running the same frames repeatedly with different values, and
you cannot replay a live moment or iterate on a parser during a game you are
also trying to play. It is a setup cost, not the operating model.

## Preflight

```bash
2kw doctor
```

Checks Python, dependencies, capture devices, config, calibration, atlas, and
disk. Every failure says what to do about it. Run it before a session rather
than discovering a broken setup with a game already going.

## Calibrating

**The shipped region coordinates are placeholders.** 2K moves its HUD every
year, so they must be fitted to your own capture before anything downstream
works.

Easiest is in the app: open **Calibrate regions**, hit *Grab frame* with 2K on
screen, pick a region name, drag a box, Save. Already-defined regions are drawn
on the frame so you can see what is set and what is not.

The CLI equivalents still exist if you prefer them:

```bash
2kw snapshot --count 3
2kw calibrate --region scoreboard
```

Then tune the classifier thresholds. This is the one step that wants video
rather than stills, because it needs the same frames replayed repeatedly:

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
vision/     court geometry and the image <-> court-feet transform
pipeline/   the frame loop and the event bus stages attach to
storage/    SQLite schema and access
app/        local server, web UI, and the native window hosting it
```

The frame loop publishes events and knows nothing about the database, the
dashboard, or the tracker — those attach as subscribers. Adding the tracker
should mean registering another stage, not restructuring the loop.

## What works now

- Frame capture from the OBS virtual camera or a recording
- Screen-state classification and debounced transitions, logged to SQLite
- Region calibration, snapshots, and threshold-tuning tools
- Player registry keyed on gamertag
- Scoreboard reader — structurally complete, needs a glyph atlas to produce values
- Desktop app: live crop previews, activity log, start/stop, inline corrections
- In-app region calibration by dragging boxes on a live frame
- Automatic frame collection, filed by screen, with a manifest
- `2kw doctor` preflight checks
- Post-game box score parsing, verified cell-for-cell against a real capture
- Shot feedback (timing, contest, distance), read by consensus across an event
- Court model and homography, with a reprojection overlay to verify a fit

## What's next

1. **Glyph atlas** (`2kw atlas`) so the live scoreboard actually reads. The
   box score already parses; this is the in-game HUD.
2. **Wire shot feedback into the live pipeline**, so events are detected and
   logged as you play rather than parsed from saved frames.
3. **Wire the box score into the live pipeline** so reaching POST_GAME parses
   and stores a game automatically.
4. **Squad analytics.** Once box scores are keyed to the roster, lineup +/- for
   your friend group falls out of a GROUP BY.
5. **Highlight clipping** via obs-websocket triggering OBS's replay buffer.
6. **Player tracking.** Homography is done; next is landmark detection to feed
   it, then a YOLO player detector, then ByteTrack running in court space.
