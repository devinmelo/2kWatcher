"""Parsing the MyCareer post-game GAME STATS screen.

This screen is the densest structured data 2K gives you: ten players with full
stat lines, in one frame, already aggregated. Parsing it is worth more per unit
of effort than any amount of real-time computer vision, and it doubles as
ground truth for grading everything else.

Different constraints apply here than to the live scoreboard. That is read ten
times a second, so it uses template matching. This is read once per game, so
OCR is affordable and handles arbitrary gamertags, which templates cannot.

Two properties of the screen do most of the work:

  * A green triangle marks your own row and a red one marks your matchup, in a
    narrow gutter left of the names. Identity is read off the screen rather
    than configured, so it stays right even when you switch builds or teams.
  * The TOTAL row is a checksum. Player rows must sum to it for PTS, REB, AST,
    STL and BLK, and for the made/attempted fractions. FOULS and TO are
    excluded: 2K reports team fouls and team turnovers there, which genuinely
    differ from the sum of the individual rows.

Cells that cannot be read come back as None rather than a guess. A wrong number
silently entering the database is far worse than a gap the app can ask about.
"""

from __future__ import annotations

import collections
import difflib
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

# A '/' rendered in the box score font, normalized to GLYPH_HEIGHT, harvested
# from a real capture. Used only to locate the slash inside a fraction cell.
SLASH_TEMPLATE_PATH = Path(__file__).parent / "glyphs" / "slash.png"
GLYPH_HEIGHT = 24
SLASH_MATCH_THRESHOLD = 0.45

# Normalized layout, measured from a 1920x1080 capture of the 2K MyCareer
# recap. Fractions of frame width/height, so it survives a resolution change.
# 2K moves its UI every year; this is the block to re-measure when it does.
ROW_HEIGHT = 0.0287
MARKER_GUTTER = (0.3073, 0.3177)

OPPONENT_ROWS = (0.2593, 0.2981, 0.3352, 0.3722, 0.4102)
OPPONENT_TOTAL = 0.4519
TEAM_ROWS = (0.5500, 0.5889, 0.6278, 0.6657, 0.7037)
TEAM_TOTAL = 0.7454

COLUMNS: dict[str, tuple[float, float]] = {
    "name":  (0.3188, 0.4323),
    "grade": (0.4401, 0.4740),
    "pts":   (0.4818, 0.5130),
    "reb":   (0.5208, 0.5531),
    "ast":   (0.5615, 0.5927),
    "stl":   (0.6000, 0.6313),
    "blk":   (0.6380, 0.6693),
    "fouls": (0.6771, 0.7094),
    "tov":   (0.7161, 0.7438),
    "fg":    (0.7526, 0.8063),
    "tp":    (0.8115, 0.8656),
    "ft":    (0.8672, 0.9177),
}

FRACTION_COLUMNS = ("fg", "tp", "ft")
COUNT_COLUMNS = ("pts", "reb", "ast", "stl", "blk", "fouls", "tov")
# Columns whose player rows genuinely sum to the TOTAL row. FOULS and TO are
# excluded because 2K reports team totals there, not a sum of the rows.
CHECKSUM_COLUMNS = ("pts", "reb", "ast", "stl", "blk", *FRACTION_COLUMNS)

COUNT_RE = re.compile(r"^\d{1,3}$")
FRACTION_RE = re.compile(r"^\d{1,3}/\d{1,3}$")
GRADE_RE = re.compile(r"^[ABCDF][+-]?$")
# Gamertags are alphanumerics plus a few separators; anything else is OCR noise.
GAMERTAG_RE = re.compile(r"[^A-Za-z0-9 _.\-]")


@dataclass
class PlayerRow:
    """One player's line. Any unread field is None, never a guess."""

    name: str | None = None
    grade: str | None = None
    pts: int | None = None
    reb: int | None = None
    ast: int | None = None
    stl: int | None = None
    blk: int | None = None
    fouls: int | None = None
    tov: int | None = None
    fgm: int | None = None
    fga: int | None = None
    tpm: int | None = None
    tpa: int | None = None
    ftm: int | None = None
    fta: int | None = None
    team: str = "them"
    is_you: bool = False
    is_matchup: bool = False
    is_ai: bool = False

    @property
    def complete(self) -> bool:
        return self.name is not None and self.pts is not None


@dataclass
class BoxScore:
    """A parsed GAME STATS screen."""

    players: list[PlayerRow] = field(default_factory=list)
    totals: dict[str, PlayerRow] = field(default_factory=dict)
    checksum_failures: list[str] = field(default_factory=list)
    unread_cells: list[str] = field(default_factory=list)

    @property
    def you(self) -> PlayerRow | None:
        return next((p for p in self.players if p.is_you), None)

    @property
    def trustworthy(self) -> bool:
        """Whether this parse can be written to the database unreviewed."""
        return not self.checksum_failures and not self.unread_cells


class BoxScoreParser:
    """Reads the post-game GAME STATS screen."""

    def __init__(self, *, min_confidence_votes: int = 1) -> None:
        self.min_votes = min_confidence_votes
        self._tesseract = None

    def available(self) -> bool:
        """Whether OCR is usable. Names cannot be read without it."""
        from .ocr import configure

        if not configure():
            return False
        import pytesseract
        self._tesseract = pytesseract
        return True

    def parse(self, image: np.ndarray) -> BoxScore:
        if self._tesseract is None and not self.available():
            raise RuntimeError(
                "Tesseract is not available; the box score needs it for "
                "gamertags. Install the tesseract-ocr package and pytesseract."
            )

        box = BoxScore()
        for rows, total_y, team in (
            (OPPONENT_ROWS, OPPONENT_TOTAL, "them"),
            (TEAM_ROWS, TEAM_TOTAL, "us"),
        ):
            parsed = [self._read_row(image, y, team, box) for y in rows]
            box.players.extend(parsed)
            box.totals[team] = self._read_row(image, total_y, team, box,
                                              is_total=True)
            box.checksum_failures.extend(
                self._verify(parsed, box.totals[team], team))
        return box

    # --- rows and cells -------------------------------------------------

    def _read_row(self, image: np.ndarray, y: float, team: str,
                  box: BoxScore, *, is_total: bool = False) -> PlayerRow:
        row = PlayerRow(team=team)

        if not is_total:
            marker = self._marker(image, y)
            row.is_you = marker == "you"
            row.is_matchup = marker == "matchup"

        for column in COLUMNS:
            crop = self._cell(image, y, column)
            if column == "name":
                row.name, had_icon = self._read_name(crop)
                # A slot filled by the CPU carries no platform icon. That is a
                # structural signal; the row's "AI Player" label is not, since
                # OCR renders it "Al Player" about as often.
                row.is_ai = row.name is not None and not had_icon
            elif column == "grade":
                row.grade = self._read_cell(crop, GRADE_RE, "ABCDF+-")
            elif column in FRACTION_COLUMNS:
                text = self._read_cell(crop, FRACTION_RE, "0123456789/")
                if text is None:
                    text = self._read_fraction_by_halves(crop)
                made, attempted = _split_fraction(text)
                prefix = {"fg": ("fgm", "fga"), "tp": ("tpm", "tpa"),
                          "ft": ("ftm", "fta")}[column]
                setattr(row, prefix[0], made)
                setattr(row, prefix[1], attempted)
            else:
                text = self._read_cell(crop, COUNT_RE, "0123456789")
                setattr(row, column, int(text) if text else None)

            if column != "name" and _is_unread(row, column):
                label = "TOTAL" if is_total else (row.name or f"row@{y:.3f}")
                box.unread_cells.append(f"{team}/{label}/{column}")

        return row

    def _cell(self, image: np.ndarray, y: float, column: str) -> np.ndarray:
        h, w = image.shape[:2]
        x0, x1 = COLUMNS[column]
        half = ROW_HEIGHT / 2
        return image[
            max(0, int((y - half) * h)):int((y + half) * h),
            max(0, int(x0 * w)):int(x1 * w),
        ]

    def _marker(self, image: np.ndarray, y: float) -> str | None:
        """Read the YOU / MATCHUP triangle from the gutter left of the names.

        Reading identity off the screen beats configuring it: it keeps working
        when you switch builds, teams or modes.
        """
        h, w = image.shape[:2]
        half = ROW_HEIGHT / 2
        gutter = image[
            max(0, int((y - half) * h)):int((y + half) * h),
            int(MARKER_GUTTER[0] * w):int(MARKER_GUTTER[1] * w),
        ]
        if gutter.size == 0:
            return None
        hsv = cv2.cvtColor(gutter, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([40, 120, 90]),
                            np.array([85, 255, 255])).sum()
        red = cv2.inRange(hsv, np.array([0, 120, 90]),
                          np.array([10, 255, 255])).sum()
        if max(green, red) < 255 * 20:
            return None
        return "you" if green >= red else "matchup"

    def _read_name(self, crop: np.ndarray) -> tuple[str | None, bool]:
        """Read a gamertag, removing the platform icon that precedes it.

        The name column has to start left of the icon, because AI-filled slots
        have no icon and their text begins where the icon would be — cropping
        past it truncates them. So the icon is erased instead of avoided.
        """
        stripped, had_icon = _erase_platform_icon(crop)
        text = self._ocr(stripped, psm=7, whitelist=None, scale=4, pad=8)
        cleaned = GAMERTAG_RE.sub("", text).strip() if text else ""
        return (cleaned or None), had_icon

    def _read_cell(self, crop: np.ndarray, pattern: re.Pattern,
                   whitelist: str) -> str | None:
        """Read one small cell, voting across OCR configurations.

        A single glyph is the awkward case: at psm 7 Tesseract often discards
        it as noise, so cells with one component are offered the single-
        character modes first. Only readings matching the expected shape are
        counted, so a malformed result abstains rather than winning.
        """
        single = _component_count(crop) <= 1
        psms = (10, 7, 8) if single else (7, 6, 11)

        votes: list[str] = []
        for psm in psms:
            for scale, pad in ((4, 8), (6, 10)):
                text = self._ocr(crop, psm=psm, whitelist=whitelist,
                                 scale=scale, pad=pad)
                if text and pattern.match(text):
                    votes.append(text)
        if len(votes) < self.min_votes:
            return None
        return collections.Counter(votes).most_common(1)[0][0]

    def _read_fraction_by_halves(self, crop: np.ndarray) -> str | None:
        """Locate the slash, then read each side on its own.

        Reserved for cells the whole-cell read gave up on. A thin italic '1'
        fuses with the slash, and no Tesseract configuration recovers it from
        the cell as a whole — but split at the slash, each side reads cleanly.

        Strictly a fallback, because this is the less accurate method overall:
        it drops a leading '1' on the right-hand side where the whole-cell read
        does not. The two fail on disjoint cells, so consulting this one only
        where the other abstained takes the best of both.
        """
        slash = _slash_template()
        if slash is None or crop is None or crop.size == 0:
            return None

        binary = _binarize(crop)
        ys, xs = np.nonzero(binary)
        if len(xs) == 0:
            return None
        trimmed = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        scale = GLYPH_HEIGHT / trimmed.shape[0]
        width = max(4, int(round(trimmed.shape[1] * scale)))
        if width < slash.shape[1]:
            return None
        normalized = cv2.resize(trimmed, (width, GLYPH_HEIGHT),
                                interpolation=cv2.INTER_AREA)

        result = cv2.matchTemplate(normalized, slash, cv2.TM_CCOEFF_NORMED)
        if float(result.max()) < SLASH_MATCH_THRESHOLD:
            return None

        # Map the match back onto the original crop.
        slash_x = int(xs.min() + round(int(result.argmax()) / scale))
        slash_w = int(round(slash.shape[1] / scale))
        left = crop[:, max(0, int(xs.min()) - 2):slash_x + 2]
        right = crop[:, slash_x + slash_w - 2:int(xs.max()) + 3]

        made, attempted = self._read_digits(left), self._read_digits(right)
        if not (made and attempted):
            return None
        return f"{made}/{attempted}"

    def _read_digits(self, crop: np.ndarray) -> str:
        """Read a short run of digits, taking the first plausible reading."""
        for psm in (10, 8, 7):
            text = self._ocr(crop, psm=psm, whitelist="0123456789",
                             scale=6, pad=12)
            if text.isdigit():
                return text
        return ""

    def _ocr(self, crop: np.ndarray, *, psm: int, whitelist: str | None,
             scale: int, pad: int) -> str:
        if crop is None or crop.size == 0:
            return ""
        prepared = _prepare(crop, scale, pad)
        config = f"--psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        try:
            raw = self._tesseract.image_to_string(prepared, config=config)
        except Exception:                                   # noqa: BLE001
            log.exception("OCR failed on a cell")
            return ""
        return raw.strip().replace("\n", "")

    # --- validation -----------------------------------------------------

    def _verify(self, players: list[PlayerRow], total: PlayerRow,
                team: str) -> list[str]:
        """Check player rows against the TOTAL row.

        Only the columns 2K actually totals are checked; FOULS and TO report
        team figures that legitimately differ from the sum of the rows.
        """
        failures = []
        for column in CHECKSUM_COLUMNS:
            fields = ({"fg": ("fgm", "fga"), "tp": ("tpm", "tpa"),
                       "ft": ("ftm", "fta")}.get(column) or (column,))
            for name in fields:
                values = [getattr(p, name) for p in players]
                expected = getattr(total, name)
                if expected is None or any(v is None for v in values):
                    continue        # unread cells are reported separately
                if sum(values) != expected:
                    failures.append(
                        f"{team}/{name}: rows sum to {sum(values)}, "
                        f"TOTAL says {expected}")
        return failures


# How close an OCR'd gamertag must be to a known one to be treated as it.
# Tuned above the observed error rate (a dropped or doubled character in a
# ~12-character tag scores about 0.93) and below the similarity of genuinely
# different tags.
NAME_MATCH_RATIO = 0.82


def resolve_names(box: BoxScore, roster: list[str]) -> dict[str, str]:
    """Snap OCR'd gamertags onto known ones from the player registry.

    Gamertags are the one field templates cannot help with, and OCR reliably
    confuses a handful of glyph pairs — capital I for lowercase L, a doubled
    i, a dropped digit. Since the same small group of people play together
    night after night, the registry resolves those: a tag only has to be read
    correctly, and confirmed, once.

    Returns the substitutions made, so they can be shown rather than applied
    silently.
    """
    if not roster:
        return {}
    changes: dict[str, str] = {}
    for player in box.players:
        if not player.name or player.name in roster:
            continue
        matches = difflib.get_close_matches(
            player.name, roster, n=1, cutoff=NAME_MATCH_RATIO)
        if matches:
            changes[player.name] = matches[0]
            player.name = matches[0]
    return changes


def _is_unread(row: PlayerRow, column: str) -> bool:
    if column in FRACTION_COLUMNS:
        made = {"fg": "fgm", "tp": "tpm", "ft": "ftm"}[column]
        return getattr(row, made) is None
    return getattr(row, column, None) is None


def _split_fraction(text: str | None) -> tuple[int | None, int | None]:
    if not text or "/" not in text:
        return None, None
    made, _, attempted = text.partition("/")
    if not (made.isdigit() and attempted.isdigit()):
        return None, None
    made_n, attempted_n = int(made), int(attempted)
    # Makes cannot exceed attempts; if they do, the read is wrong.
    if made_n > attempted_n:
        return None, None
    return made_n, attempted_n


# The platform icon is noticeably larger than any character: ~24x24, where
# text glyphs top out around 15x16. Size discriminates; ink does not — on the
# highlighted row the icon draws as an outline (~190px of ink) rather than
# filled (~370px), which overlaps the range of ordinary letters.
ICON_MIN_HEIGHT = 20
ICON_MIN_WIDTH = 18


def _erase_platform_icon(crop: np.ndarray) -> tuple[np.ndarray, bool]:
    """Strip the leading Xbox/PSN icon. Returns the crop and whether one was there.

    Left in place it OCRs as a stray leading character; cropped around, it
    truncates the AI rows that lack it. Detecting it by size handles both.

    Whether an icon is present is also how a human player is told from an
    AI-filled slot — a structural signal, unlike the row's label, which OCR
    can misread ("AI Player" comes back as "Al Player").
    """
    if crop is None or crop.size == 0:
        return crop, False
    binary = _binarize(crop)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    icon = None
    for i in range(1, count):
        x, y, w, h, _area = stats[i]
        if h >= ICON_MIN_HEIGHT and w >= ICON_MIN_WIDTH:
            # Only the leading glyph can be the icon.
            if icon is None or x < icon[0]:
                icon = (x, y, w, h)
    if icon is None:
        return crop, False

    # Slice past it rather than painting over it: any painted patch has to
    # match the row background exactly, and on the highlighted row it does
    # not — the leftover rectangle then reads as a stray underscore.
    x, _y, w, _h = icon
    return crop[:, x + w + 2:], True


_SLASH_CACHE: np.ndarray | None = None
_SLASH_LOADED = False


def _slash_template() -> np.ndarray | None:
    global _SLASH_CACHE, _SLASH_LOADED
    if not _SLASH_LOADED:
        _SLASH_LOADED = True
        if SLASH_TEMPLATE_PATH.exists():
            _SLASH_CACHE = cv2.imread(str(SLASH_TEMPLATE_PATH),
                                      cv2.IMREAD_GRAYSCALE)
        else:
            log.warning("No slash template at %s; fraction fallback disabled",
                        SLASH_TEMPLATE_PATH)
    return _SLASH_CACHE


def _binarize(crop: np.ndarray) -> np.ndarray:
    """Threshold a cell to white glyphs on black, whichever way it was drawn.

    The selected row is drawn dark-on-yellow while every other row is
    light-on-dark, so polarity is decided from the border, which is always
    background.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if border.mean() >= 128:
        binary = cv2.bitwise_not(binary)
    return binary


def _component_count(crop: np.ndarray, min_area: int = 12) -> int:
    if crop is None or crop.size == 0:
        return 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(_binarize(crop), 8)
    return sum(1 for i in range(1, count)
               if stats[i][4] >= min_area and stats[i][3] >= 6)


def _prepare(crop: np.ndarray, scale: int, pad: int) -> np.ndarray:
    """Upscale, threshold and pad a cell into what Tesseract wants.

    Tesseract expects dark text on light with quiet margin around it; without
    the margin it clips glyphs at the edge of the image.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    scaled = cv2.resize(gray, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(scaled, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if border.mean() < 128:
        binary = cv2.bitwise_not(binary)
    return cv2.copyMakeBorder(binary, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)
