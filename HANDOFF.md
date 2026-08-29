# 2kWatcher — handoff

Context for a session running locally on the Windows machine that owns the
capture card. You have something the remote session did not: **the user's real
frames on disk**. That is the bottleneck this note exists to unblock.

## What the project is

Watches NBA 2K over a capture card and turns gameplay into stats.

Xbox → Elgato (split) → monitor for play, PC for analysis. The console renders
and the monitor runs off HDMI passthrough, so nothing here can affect how the
game plays. The PC is a dedicated analysis box: an idle RTX 3070, 8GB VRAM.

Video only. The Xbox exposes no memory and no API, so every fact comes from
pixels. Everything runs locally; no cloud, no API cost.

## Setup on this machine

Repo at `C:\Users\dtmel\Documents\Claude\2kWatcher`, branch
`claude/nba2k-gameplay-monitor-q13vp8`. Python 3.12 venv at `.venv`.
PowerShell needs a leading `.\` on venv commands.

```powershell
.\.venv\Scripts\2kw doctor     # preflight
.\.venv\Scripts\2kw app        # the app
.\.venv\Scripts\python -m pytest -q
```

OBS owns the Elgato (its driver allows one consumer) and exposes it as a
virtual camera on **device 0, 1920x1080**. Tesseract was still uninstalled as
of the last check — needed for the box score and shot feedback, found
automatically at the UB-Mannheim default path.

## Where things stand

Working and verified:

- Capture, screen-state machine, event bus, SQLite storage, desktop app
- **Box score parser** — every cell of a real 1080p post-game screen, checked
  against hand-read truth in `tests/test_boxscore.py`
- **Shot feedback reader** — both shot events in a real Rec clip, by consensus
- **Court model + homography** — round-trips synthetic cameras to within an inch
- 90 tests passing

Not done:

- Neither parser is wired into the live pipeline. They work standalone only.
  This is the biggest single gap: the app currently tracks state and collects
  frames, and logs no shots or games.
- No glyph atlas, so live scoreboard values read "not read".
- No player tracking beyond the homography layer.

## The two open bugs — start here

Both need the frames in `data\collect\` (about 50 of them, in per-screen
folders). **Measure from those rather than reasoning about it.** Note the
folders are partly mislabelled, because of bug 2 — sort them by eye first.

### 1. Bottom scoreboard regions are slightly off

Rec draws its scoreboard at the BOTTOM. Coordinates in
`config/regions.example.yaml` were measured from a **1180x664** screen
recording and verified there, but on the user's 1920x1080 capture the
`score_away` crop comes out as a flat grey rectangle — landing on the plate but
missing the number. `shot_feedback` transferred correctly and reads
"TIMING / SLIGHTLY EARLY", so the frame is not letterboxed or offset globally;
the bottom-bar numbers just need re-measuring.

Measure from a real `live` frame and update the config. Coordinates are
normalized fractions of frame width/height.

### 2. The state machine flaps

Transitions fire every 1-2 seconds — `live → menu → dead_ball → live` — which
is `scoreboard_present` oscillating on its threshold, not the game changing.
The 3-sample debounce in `state/machine.py` cannot rescue a signal flickering
at the source.

`state/detectors.py` currently gates on two measurements of the scoreboard
region, both published to the app's Session panel while it runs:

- edge density >= 0.050
- dark fraction (share of pixels below luma 70) within 0.35 - 0.88

Measured on real Rec gameplay at 1180x664: edges 0.063-0.078, dark fraction
0.63-0.65, tightly clustered. Those thresholds were calibrated against many
positives but only **one** negative frame, which is very likely why the signal
sits near a boundary. Recalibrate against real menu and loading frames, and
print the distributions before choosing thresholds. `2kw probe --video <file>`
dumps per-frame signals; the app shows them live.

Fixing region placement may resolve the flapping on its own — if the region is
half off the plate, both measurements land near their limits. Do bug 1 first.

## Design decisions worth keeping

These were arrived at by measurement, not preference. Changing them without
new evidence will regress things.

- **Refuse rather than guess.** Every parser returns None for a cell it cannot
  read. A wrong number entering the database is far worse than a gap, and the
  app has a correction UI for gaps. Do not add fallbacks that invent values.
- **Corrections are training data.** Fixes made in the app store the crop that
  caused them, in the `corrections` table. That is the path to a glyph atlas.
- **Live scoreboard uses template matching; the box score uses OCR.** Different
  constraints — one runs 10x/second on a fixed font, the other once per game on
  arbitrary gamertags. Do not unify them.
- **The box score TOTAL row is a checksum**, but only for PTS/REB/AST/STL/BLK
  and the made/attempted fractions. FOULS and TO are excluded: 2K reports team
  fouls and team turnovers there, confirmed against the fixture (opponent fouls
  sum to 7 while the total says 4).
- **Identity is read off the screen, not configured.** A green triangle marks
  the user's row and a red one their matchup. AI slots are detected by the
  missing platform icon, not the "AI Player" label, which OCR renders "Al
  Player" about half the time.
- **Shot feedback is read by consensus across an event**, never one frame. Most
  frames abstain; the ones that read are correct. Cutoffs are set so a marginal
  frame returns nothing rather than a confident wrong verdict.
- **Track in court feet, not pixels.** `vision/court.py` and
  `vision/homography.py`. Pixel positions are meaningless once the camera pans.
- **Regions are normalized fractions**, so a resolution change is free and only
  a HUD change needs recalibration. 2K moves its HUD every year.

## After that

1. Wire both parsers into the live pipeline. Shot feedback: the presence gate
   fires, buffer the banner frames, read by consensus when it clears, log it.
   Box score: on transition to POST_GAME, parse and write the game plus all ten
   stat lines. This is what makes the app actually do something.
2. Build the glyph atlas from collected frames so the live scoreboard reads.
3. Squad analytics — with games keyed to the roster, lineup +/- is a GROUP BY.
4. Player tracking: court landmark detection to feed the existing homography,
   then a YOLO detector, then ByteTrack in court space. Month-plus; do last.

## Things to be careful about

- Verify against real frames, not synthetic ones. Two bugs so far came from
  reasoning about a HUD nobody had looked at — the original config assumed a
  scoreboard at the top of the screen, which Rec does not have.
- `doctor` once passed "Regions calibrated" merely because the file existed.
  Be suspicious of checks that pass for doing nothing.
- Other modes (Play Now, MyCareer games) lay the HUD out differently from Rec
  and need their own calibration. Do not assume Rec coordinates transfer.
