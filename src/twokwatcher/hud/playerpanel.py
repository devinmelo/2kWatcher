"""Reading the rotating player panel on the Rec scoreboard.

The right third of the scoreboard plate is a live per-player box score. It
cycles roughly every six seconds, and over a game it shows every player on the
floor: gamertag, position, teammate grade, points, rebounds, assists, and one
more stat that varies by player.

This matters because it is the only per-player line available *during* a game.
The post-game box score is richer, but you only get it if the game reaches its
post-game screen — a quit, a disconnect, or a missed transition and the whole
game is lost. This panel is a running record that survives all three, and it
independently cross-checks the box score parser when both are available.

Three things shape this parser:

  * **The layout is discovered, not assumed.** 2K draws a single player under a
    GAME STATS heading and two side by side under TOP PERFORMERS, and their
    columns sit in completely different places. Rather than hard-coding either,
    the column-label row is segmented and the count of label clusters gives the
    arrangement: five columns for one player, eight for two. Every cell is then
    placed from the labels actually found, so the parser survives the columns
    shifting.

  * **The arrangement is not the meaning.** The same five-column frame is also
    used for an ATTRIBUTES panel, which shows a player's ratings — DUNK, BALL,
    3PT — rather than their game. It is pixel-identical in structure: same
    columns, same heading position, same everything but the words. So a panel
    is only reported as stats once its labels confirm it, and until then the
    arrangement is reported without a layout. Reading ratings as a stat line
    would be silent corruption of exactly the kind this project refuses.

  * **The fifth column is not a fixed stat.** It has been seen as 3PM, 3P%,
    FG% and OREB, chosen per player. Its label is read alongside its value and
    kept with it, because filing a 3PM as an FG% would silently corrupt the
    numbers. A value whose label cannot be read is discarded rather than
    guessed at.

  * **Reads are gated on the panel being settled.** Between players the panel
    cross-fades, and a half-faded frame is exactly the kind of input that
    produces a confident wrong answer. Frames are only read once the panel has
    stopped changing, and a rotation is read by consensus across its frames the
    same way the shot banner is.

Numbers come from the glyph atlas, as everywhere else on the live HUD. The
gamertag is arbitrary text and needs OCR; the labels and the teammate grade are
small closed vocabularies, so OCR plus a fuzzy match to the known values is
enough for them. Without an atlas the numbers read None, and without Tesseract
the text reads None. Neither is invented.
"""

from __future__ import annotations

import collections
import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from .scoreboard import DEFAULT_ATLAS_DIR, match_glyphs, preprocess_digits

log = logging.getLogger(__name__)

PANEL_REGION = "player_panel"

# The panel geometry was measured at this size; every box below is expressed in
# these reference pixels and scaled to whatever the crop actually is, so a
# resolution change costs nothing.
REFERENCE_SIZE = (458, 92)                 # width, height

# Row bands, panel-relative. The name row carries the gamertags and the heading
# that names the layout; the label and value rows carry the stat columns.
NAME_ROW = (12, 34)                        # y0, y1
LABEL_ROW = (37, 58)
VALUE_ROW = (53, 86)
# Column labels never run past here; beyond it lies the panel's background art
# and, in the two-player layout, the second player's portrait.
LABEL_SEARCH_X = (20, 410)
COLUMN_HALF_WIDTH = 31

# Cells that differ between the two layouts, panel-relative (x0, x1).
GAME_STATS_CELLS = {"name": (45, 202), "heading": (202, 328), "badge": (22, 62)}
TOP_PERFORMERS_CELLS = {
    "name": (18, 140), "heading": (150, 255), "badge": (10, 40),
    "name2": (288, 410), "badge2": (337, 370),
}
BADGE_ROW = (68, 90)

# How many label clusters each column arrangement draws.
LAYOUT_BY_CLUSTERS = {5: "game_stats", 8: "top_performers"}
# Recognised stat labels needed before a panel is accepted as a game line
# rather than the identically-shaped attributes panel. Two is enough: TG, PTS,
# REB and AST are all present on a real line, so resolving only one means the
# labels are either unreadable or not stats at all.
MIN_CONFIRMING_LABELS = 2

# Closed vocabularies. Anything that matches none of these is dropped rather
# than snapped to the nearest, so an unseen stat shows up as missing.
STAT_LABELS = ("TG", "PTS", "REB", "AST", "STL", "BLK", "TO", "FLS",
               "3PM", "3PA", "3P%", "FG%", "FT%", "OREB", "DREB", "PF")
GRADES = ("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-",
          "D+", "D", "D-", "F")
POSITIONS = ("PG", "SG", "SF", "PF", "C")
LABEL_MATCH_CUTOFF = 0.7

# Mean absolute difference between consecutive panel crops, below which the
# panel is considered to have stopped moving. A cross-fade sits far above it.
SETTLE_TOLERANCE = 2.0
# A reading must win this share of the non-empty votes in a rotation.
CONSENSUS_SHARE = 0.5


@dataclass
class PlayerStat:
    """One player's line off the panel. Anything unread stays None."""

    gamertag: str | None = None
    position: str | None = None
    grade: str | None = None
    points: int | None = None
    rebounds: int | None = None
    assists: int | None = None
    # The varying fifth column, kept as a label/value pair because the label is
    # what gives the number its meaning.
    extra_label: str | None = None
    extra_value: str | None = None

    @property
    def any_read(self) -> bool:
        return any(v is not None for v in
                   (self.gamertag, self.grade, self.points,
                    self.rebounds, self.assists))


@dataclass
class PlayerPanel:
    """One reading of the panel.

    `columns` is structural and always available: how many stat columns were
    found. `layout` is the semantic answer and stays None until the labels
    confirm the panel is showing a game line rather than, say, attributes.
    """

    columns: int = 0
    layout: str | None = None
    players: list[PlayerStat] = field(default_factory=list)

    @property
    def any_read(self) -> bool:
        return any(p.any_read for p in self.players)


class PlayerPanelReader:
    """Reads the rotating player panel from a frame."""

    def __init__(self, config: Config, atlas_dir: Path = DEFAULT_ATLAS_DIR) -> None:
        self.config = config
        self.atlas_dir = Path(atlas_dir)
        self.atlas = _load_atlas(self.atlas_dir)
        self._tesseract = None
        self._previous: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        """Whether numbers can be read at all."""
        return bool(self.atlas)

    def panel(self, image: np.ndarray) -> np.ndarray:
        return self.config.region(PANEL_REGION).crop(image)

    def settled(self, image: np.ndarray) -> bool:
        """Whether the panel has stopped changing since the last frame.

        Stateful by design: the question is not answerable from one frame, and
        the caller should be feeding every sampled frame through here anyway.
        The first frame after a reset is never settled, because there is
        nothing to compare it against.
        """
        current = cv2.cvtColor(self.panel(image), cv2.COLOR_BGR2GRAY)
        small = cv2.resize(current, (64, 16))
        previous, self._previous = self._previous, small
        if previous is None:
            return False
        return float(np.mean(cv2.absdiff(small, previous))) <= SETTLE_TOLERANCE

    def reset(self) -> None:
        """Forget the previous frame, e.g. after a state change."""
        self._previous = None

    def read(self, image: np.ndarray) -> PlayerPanel:
        """Read the panel from one full frame. Prefer `read_rotation`."""
        return self.read_panel(self.panel(image))

    def read_panel(self, panel: np.ndarray) -> PlayerPanel:
        """Read an already-cropped panel.

        Split out from `read` so the geometry can be exercised against small
        fixtures rather than whole 1080p frames.
        """
        if panel.size == 0:
            return PlayerPanel()
        scale_x = panel.shape[1] / REFERENCE_SIZE[0]
        scale_y = panel.shape[0] / REFERENCE_SIZE[1]

        columns = self._label_columns(panel, scale_x, scale_y)
        arrangement = LAYOUT_BY_CLUSTERS.get(len(columns))
        if arrangement is None:
            # Mid-fade, or a variant nobody has measured. Either way, refuse.
            return PlayerPanel(columns=len(columns))

        labels = [_closest(self._text(self._column(panel, c, LABEL_ROW,
                                                   scale_x, scale_y)), STAT_LABELS)
                  for c in columns]
        if sum(label is not None for label in labels) < MIN_CONFIRMING_LABELS:
            # Could be the attributes panel, or the labels could simply be
            # unreadable. Either way this is not a confirmed stat line.
            return PlayerPanel(columns=len(columns))

        cells = (GAME_STATS_CELLS if arrangement == "game_stats"
                 else TOP_PERFORMERS_CELLS)
        if arrangement == "game_stats":
            players = [self._read_player(panel, columns, labels, cells,
                                         scale_x, scale_y,
                                         name_key="name", badge_key="badge")]
        else:
            players = [
                self._read_player(panel, columns[:4], labels[:4], cells,
                                  scale_x, scale_y,
                                  name_key="name", badge_key="badge"),
                self._read_player(panel, columns[4:], labels[4:], cells,
                                  scale_x, scale_y,
                                  name_key="name2", badge_key="badge2"),
            ]
        return PlayerPanel(columns=len(columns), layout=arrangement,
                           players=players)

    def read_rotation(self, images: list[np.ndarray]) -> PlayerPanel:
        """Read one player's slot from several frames of it, by consensus.

        The panel holds each player for seconds, which is dozens of frames, and
        a misread glyph in one of them is not repeated in the next. Taking the
        most common reading turns many weak reads into one trustworthy one, and
        a slot whose frames genuinely disagree reports nothing.
        """
        readings = [self.read_panel(p) for p in images]
        usable = [r for r in readings if r.layout is not None]
        if not usable:
            columns = _consensus_int([r.columns for r in readings if r.columns])
            return PlayerPanel(columns=columns or 0)

        layout = _consensus([r.layout for r in usable])
        matching = [r for r in usable if r.layout == layout]
        count = max(len(r.players) for r in matching)
        merged = []
        for index in range(count):
            lines = [r.players[index] for r in matching if len(r.players) > index]
            merged.append(PlayerStat(
                gamertag=_consensus([p.gamertag for p in lines]),
                position=_consensus([p.position for p in lines]),
                grade=_consensus([p.grade for p in lines]),
                points=_consensus_int([p.points for p in lines]),
                rebounds=_consensus_int([p.rebounds for p in lines]),
                assists=_consensus_int([p.assists for p in lines]),
                extra_label=_consensus([p.extra_label for p in lines]),
                extra_value=_consensus([p.extra_value for p in lines]),
            ))
        return PlayerPanel(columns=matching[0].columns, layout=layout,
                           players=merged)

    # --- geometry ------------------------------------------------------

    def _label_columns(self, panel: np.ndarray, scale_x: float,
                       scale_y: float) -> list[float]:
        """Centres of the stat-label clusters, in panel pixels.

        This is what identifies the layout, so it stays deliberately dumb: a
        binary threshold and a scan for runs of ink. No template, no OCR.
        """
        x0, x1 = (int(v * scale_x) for v in LABEL_SEARCH_X)
        y0, y1 = (int(v * scale_y) for v in LABEL_ROW)
        band = panel[y0:y1, x0:x1]
        if band.size == 0:
            return []
        binary = _binarize(band)
        ink = binary.sum(0) > 0

        runs, start = [], None
        for index, lit in enumerate(ink):
            if lit and start is None:
                start = index
            elif not lit and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, len(ink)))

        # Letters within a label are separated by a pixel or two; labels are
        # separated by tens. Merge across the small gaps only.
        gap = max(2, int(16 * scale_x))
        merged: list[list[int]] = []
        for run in runs:
            if merged and run[0] - merged[-1][1] < gap:
                merged[-1][1] = run[1]
            else:
                merged.append([run[0], run[1]])
        width = max(2, int(10 * scale_x))
        return [x0 + (m[0] + m[1]) / 2 for m in merged if m[1] - m[0] >= width]

    def _read_player(self, panel: np.ndarray, columns: list[float],
                     labels: list[str | None], cells: dict,
                     scale_x: float, scale_y: float, *,
                     name_key: str, badge_key: str) -> PlayerStat:
        stat = PlayerStat()
        stat.gamertag = self._text(_cell(panel, cells[name_key], NAME_ROW,
                                         scale_x, scale_y))
        stat.position = _closest(self._text(_cell(panel, cells[badge_key],
                                                  BADGE_ROW, scale_x, scale_y)),
                                 POSITIONS)

        for label, centre in zip(labels, columns):
            if label is None:
                continue
            crop = self._column(panel, centre, VALUE_ROW, scale_x, scale_y)
            if label == "TG":
                stat.grade = _closest(self._text(crop), GRADES)
                continue
            digits = self._digits(crop)
            if label == "PTS":
                stat.points = digits
            elif label == "REB":
                stat.rebounds = digits
            elif label == "AST":
                stat.assists = digits
            else:
                # The varying column. Keep the number with the label that gives
                # it meaning; a value without one is worse than nothing.
                if digits is not None:
                    stat.extra_label = label
                    stat.extra_value = str(digits)
        return stat

    def _column(self, panel: np.ndarray, centre: float, row: tuple[int, int],
                scale_x: float, scale_y: float) -> np.ndarray:
        half = COLUMN_HALF_WIDTH * scale_x
        y0, y1 = (int(v * scale_y) for v in row)
        return panel[y0:y1, max(0, int(centre - half)):int(centre + half)]

    # --- reading -------------------------------------------------------

    def _digits(self, crop: np.ndarray) -> int | None:
        if not self.atlas or crop.size == 0:
            return None
        text = match_glyphs(preprocess_digits(crop), self.atlas)
        if not text:
            return None
        kept = "".join(c for c in text if c.isdigit())
        return int(kept) if kept else None

    def _text(self, crop: np.ndarray) -> str | None:
        """OCR a small cell. None whenever Tesseract is unavailable."""
        if crop.size == 0 or not self._ocr_available():
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=4, fy=4,
                              interpolation=cv2.INTER_CUBIC)
        prepared = cv2.copyMakeBorder(cv2.bitwise_not(upscaled), 20, 20, 20, 20,
                                      cv2.BORDER_CONSTANT, value=255)
        try:
            text = self._tesseract.image_to_string(prepared, config="--psm 7")
        except Exception:                                   # noqa: BLE001
            log.exception("Player panel OCR failed")
            return None
        text = " ".join(text.split())
        return text or None

    def _ocr_available(self) -> bool:
        if self._tesseract is not None:
            return True
        from .ocr import configure
        if not configure():
            return False
        import pytesseract
        self._tesseract = pytesseract
        return True


def _load_atlas(atlas_dir: Path) -> dict[str, np.ndarray]:
    if not atlas_dir.exists():
        return {}
    atlas: dict[str, np.ndarray] = {}
    for png in sorted(atlas_dir.glob("*.png")):
        glyph = ":" if png.stem == "colon" else png.stem
        image = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            atlas[glyph] = image
    return atlas


def _binarize(crop: np.ndarray) -> np.ndarray:
    """Bright HUD text on a dark plate, as white on black."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return cv2.resize(binary, (crop.shape[1], crop.shape[0]))


def _cell(panel: np.ndarray, xs: tuple[int, int], row: tuple[int, int],
          scale_x: float, scale_y: float) -> np.ndarray:
    x0, x1 = (int(v * scale_x) for v in xs)
    y0, y1 = (int(v * scale_y) for v in row)
    return panel[y0:y1, x0:x1]


# Glyphs OCR routinely swaps on this HUD's condensed font. Applied only as an
# alternative candidate, never in place of the original.
_OCR_CONFUSIONS = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})


def _closest(text: str | None, vocabulary: tuple[str, ...]) -> str | None:
    """Snap OCR output to a known value, or give up.

    The vocabularies here are tiny and their members look nothing alike, so a
    loose cutoff still separates them — but a token matching nothing is dropped
    rather than forced onto the nearest, which is how a 3PM becomes an FG%.

    Fuzzy matching alone is not enough, because near-misses here are not
    symmetric: "0REB" is one edit from OREB and one deletion from REB, and
    plain difflib prefers the shorter word — quietly turning offensive rebounds
    into total rebounds. So an exact hit on a confusion-corrected candidate is
    taken ahead of any fuzzy match.
    """
    if not text:
        return None
    candidate = text.strip().upper().replace(" ", "")
    corrected = candidate.translate(_OCR_CONFUSIONS)
    for variant in (candidate, corrected):
        if variant in vocabulary:
            return variant

    best, best_ratio = None, 0.0
    for variant in {candidate, corrected}:
        for known in vocabulary:
            ratio = difflib.SequenceMatcher(None, variant, known).ratio()
            if ratio > best_ratio:
                best, best_ratio = known, ratio
    return best if best_ratio >= LABEL_MATCH_CUTOFF else None


def _consensus(values: list) -> object | None:
    """Most common non-empty reading, if it commands a clear majority."""
    votes = [v for v in values if v is not None and v != ""]
    if not votes:
        return None
    value, count = collections.Counter(votes).most_common(1)[0]
    return value if count >= max(1, len(votes) * CONSENSUS_SHARE) else None


def _consensus_int(values: list) -> int | None:
    result = _consensus(values)
    return int(result) if result is not None else None
