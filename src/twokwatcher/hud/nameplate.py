"""Reading the gamertag plates 2K draws over players during a game.

2K labels two players at any moment: whoever has the ball, and your own player
wherever they are. That is the whole basis for attributing a shot to a shooter,
and it means the shooter is named on screen at the moment it matters — no
tracker, no re-identification, no court geometry.

The plate is not a HUD panel in a fixed place. It follows the player, so it has
to be found rather than cropped, and there is no background box to find it by:
the gamertag is drawn straight over the court as light text with a dark
outline. What is fixed is the little platform badge immediately left of every
name, so that is the anchor — matched as a template, with the name read from
the strip to its right. Plates shrink with distance, so the badge is matched at
several scales and the strip is sized to whichever one hit.

The reading is then constrained by something the box score already gives us:
the ten gamertags in this game. That turns open-ended OCR into picking the
closest of ten known strings, which is a far easier problem and the reason a
mangled read still resolves correctly. A plate matching nothing in the roster
is reported unresolved rather than guessed at.

Two known gaps, both waiting on footage rather than ideas:

  * Only the Xbox badge is templated, because it is the only one that appears
    clearly in the frames on hand. Players on other platforms carry a different
    badge and will not be found until one can be cut from a capture.
  * Light text over pale floorboards is not reading yet. Over dark boards it
    separates on brightness and reads cleanly; over bare wood the contrast runs
    the other way and none of the preparations here recover it.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

BADGE_DIR = Path(__file__).parent / "glyphs"
BADGE_PATHS = (BADGE_DIR / "plate_xbox.png",)

# Measured on a 1920x1080 capture: the badge is about 35x31 and the gamertag
# runs from a few pixels to its right, in text about 24 tall. Everything is
# relative to the badge that anchors it, so a plate reads the same wherever on
# screen the player has wandered to.
BADGE_SIZE = (35, 31)
NAME_GAP = 3               # badge's right edge to the first letter
NAME_MAX_WIDTH = 240       # generous: gamertags run long
NAME_PAD_Y = 4

# Below this the badge is being matched against court markings rather than a
# plate. Real badges on the frames here score 0.70 and above.
BADGE_MATCH_THRESHOLD = 0.60
# Plates shrink with distance — a player at the far end carries a badge around
# 60% the size of one in the foreground — so match at a range of scales.
BADGE_SCALES = (1.0, 0.85, 0.7, 0.6, 0.5)
# Two hits closer together than this are the same badge found twice.
MIN_BADGE_SEPARATION = 24
# How close a reading must be to a roster name to be accepted as it.
NAME_MATCH_CUTOFF = 0.6


@dataclass
class Nameplate:
    """One gamertag plate found on screen."""

    name: str | None            # resolved against the roster, else None
    raw: str                    # what OCR actually returned
    x: int                      # badge centre, in frame pixels
    y: int
    confidence: float           # badge template match
    scale: float = 1.0

    @property
    def resolved(self) -> bool:
        return self.name is not None


class NameplateReader:
    """Finds and reads the gamertag plates on a gameplay frame."""

    def __init__(self) -> None:
        self._badges: list[np.ndarray] | None = None
        self._tesseract = None

    def available(self) -> bool:
        """Whether plates can be read. Needs OCR; badges ship with the package."""
        if not self._load_badges():
            return False
        if self._tesseract is not None:
            return True
        from .ocr import configure
        if not configure():
            return False
        import pytesseract
        self._tesseract = pytesseract
        return True

    def _load_badges(self) -> bool:
        if self._badges is None:
            self._badges = []
            for path in BADGE_PATHS:
                if not path.exists():
                    log.warning("No plate badge template at %s", path)
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    self._badges.append(image)
        return bool(self._badges)

    # --- finding --------------------------------------------------------

    def badges(self, image: np.ndarray) -> list[tuple[int, int, float, float]]:
        """Locate platform badges, as (x, y, score, scale) of their centres.

        Template matching rather than anything cleverer because the badge is a
        fixed piece of UI art — the same reasoning the scoreboard uses for its
        digits.
        """
        if not self._load_badges() or image is None or image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found: list[tuple[int, int, float, float]] = []
        for badge in self._badges:
            for scale in BADGE_SCALES:
                sized = badge if scale == 1.0 else cv2.resize(
                    badge, (max(1, int(badge.shape[1] * scale)),
                            max(1, int(badge.shape[0] * scale))))
                if (gray.shape[0] < sized.shape[0]
                        or gray.shape[1] < sized.shape[1]):
                    continue
                result = cv2.matchTemplate(gray, sized, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(result >= BADGE_MATCH_THRESHOLD)
                for x, y in zip(xs, ys):
                    found.append((int(x + sized.shape[1] // 2),
                                  int(y + sized.shape[0] // 2),
                                  float(result[y, x]), scale))
        return _strongest_apart(found, MIN_BADGE_SEPARATION)

    def read(self, image: np.ndarray,
             roster: list[str] | None = None) -> list[Nameplate]:
        """Every plate on the frame, resolved against the roster where possible."""
        if not self.available():
            return []
        plates = []
        for x, y, score, scale in self.badges(image):
            strip = self._name_strip(image, x, y, scale)
            name, raw = self._read_strip(strip, roster)
            plates.append(Nameplate(name=name, raw=raw, x=x, y=y,
                                    confidence=score, scale=scale))
        return plates

    def ball_handler(self, image: np.ndarray, roster: list[str],
                     me: str | None) -> Nameplate | None:
        """The plate that is not yours, which is whoever has the ball.

        Your own plate is always drawn; a second appears for the player holding
        the ball. So when two plates are up, the one that is not you is the
        handler — and when only yours is up, you have it.
        """
        plates = [p for p in self.read(image, roster) if p.resolved]
        if not plates:
            return None
        others = [p for p in plates if me is None or p.name != me]
        if others:
            others.sort(key=lambda p: -p.confidence)
            return others[0]
        return plates[0] if me is not None else None

    # --- reading --------------------------------------------------------

    def _name_strip(self, image: np.ndarray, x: int, y: int,
                    scale: float = 1.0) -> np.ndarray:
        """The text beside a badge, sized to the badge that anchors it."""
        h, w = image.shape[:2]
        half_w = int(BADGE_SIZE[0] * scale) // 2
        half_h = int(BADGE_SIZE[1] * scale) // 2
        x0 = min(w, x + half_w + max(1, int(NAME_GAP * scale)))
        x1 = min(w, x0 + max(1, int(NAME_MAX_WIDTH * scale)))
        y0 = max(0, y - half_h - max(1, int(NAME_PAD_Y * scale)))
        y1 = min(h, y + half_h + max(1, int(NAME_PAD_Y * scale)))
        return image[y0:y1, x0:x1]

    def _read_strip(self, strip: np.ndarray,
                    roster: list[str] | None) -> tuple[str | None, str]:
        """Read a name strip, trying each preparation until one lands.

        A plate is drawn over whatever the court happens to show, so no single
        threshold suits all of them: light text on dark boards separates on
        brightness, the same text over pale wood does not. Rather than tune one
        compromise, several preparations are tried and the roster decides — a
        reading is accepted once it matches a gamertag in this game, and the
        attempts stop there.
        """
        best_raw = ""
        for prepared in _preparations(strip):
            raw = self._ocr(prepared)
            if not raw:
                continue
            if not best_raw:
                best_raw = raw
            name = _resolve(raw, roster)
            if name is not None:
                return name, raw
        return None, best_raw

    def _ocr(self, prepared: np.ndarray) -> str:
        try:
            text = self._tesseract.image_to_string(prepared, config="--psm 7")
        except Exception:                                   # noqa: BLE001
            log.exception("Nameplate OCR failed")
            return ""
        return " ".join(text.split())


def _preparations(strip: np.ndarray):
    """Successive attempts at making a name strip readable."""
    if strip is None or strip.size == 0:
        return
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (0, 0), 9)
    contrast = cv2.convertScaleAbs(
        up.astype(np.int16) - blurred.astype(np.int16), alpha=3)
    for image in (
        cv2.threshold(up, 190, 255, cv2.THRESH_BINARY)[1],
        up,
        cv2.threshold(contrast, 60, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(up, 150, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(contrast, 40, 255, cv2.THRESH_BINARY)[1],
    ):
        yield cv2.copyMakeBorder(cv2.bitwise_not(image), 20, 20, 20, 20,
                                 cv2.BORDER_CONSTANT, value=255)


def _strongest_apart(found, separation: int):
    """Keep the best match in each cluster, so one badge is reported once."""
    kept = []
    for entry in sorted(found, key=lambda f: -f[2]):
        x, y = entry[0], entry[1]
        if any(abs(x - k[0]) < separation and abs(y - k[1]) < separation
               for k in kept):
            continue
        kept.append(entry)
    return kept


def _resolve(raw: str, roster: list[str] | None) -> str | None:
    """Snap a reading to a gamertag in this game, or give up.

    The roster is what makes this reliable. Ten known strings is a small enough
    space that a mangled read still lands on the right one — and a read landing
    on none of them is far better reported as unknown than as whichever was
    nearest.
    """
    if not raw or not roster:
        return None
    cleaned = raw.strip()
    for name in roster:
        if cleaned.lower() == name.lower():
            return name
    lowered = {name.lower(): name for name in roster}
    match = difflib.get_close_matches(cleaned.lower(), list(lowered),
                                      n=1, cutoff=NAME_MATCH_CUTOFF)
    return lowered[match[0]] if match else None
