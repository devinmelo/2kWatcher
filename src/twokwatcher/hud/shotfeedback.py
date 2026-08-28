"""Reading the shot feedback banner.

Every shot you take puts a banner at the top of the screen giving the release
timing, how well you were contested, and the shot distance. That is the single
most valuable signal in the game and 2K gives you no way to analyse it: it
appears for about a second and is gone.

Logged over a session it answers the questions players actually have. Whether
your releases drift late as the night goes on. Which jumpshot greens for you
and which does not. Whether you shoot worse contested from the wing than from
the corner. None of that is available anywhere else.

Two design points carry this parser:

  * The banner is centre-justified and its panel count varies — two panels for
    a shot with no distance reading, three with one — so panel positions are
    not fixed and cannot be hard-coded. A generous band is read as a whole
    instead, and the panels identify themselves by their labels.
  * The values come from a closed vocabulary. That makes OCR accuracy far less
    critical than it looks: "UGHT CONTEST" and "GXCELLENT" both resolve by
    fuzzy match. Anything that matches nothing is kept verbatim and flagged
    rather than forced onto a wrong value, so an unseen verdict shows up as
    unknown instead of silently becoming the closest known one.
"""

from __future__ import annotations

import collections
import difflib
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

# The band the banner is drawn in, normalized. Deliberately generous: the
# banner is centred and changes width with its panel count.
BANNER_BAND = (0.30, 0.016, 0.74, 0.080)   # x0, y0, x1, y1

# Known values. Extend as new ones are observed — anything unmatched is
# preserved verbatim rather than snapped to the nearest of these.
TIMING_VALUES = (
    "EXCELLENT", "GOOD", "SLIGHTLY EARLY", "SLIGHTLY LATE",
    "VERY EARLY", "VERY LATE", "EARLY", "LATE",
)
COVERAGE_VALUES = (
    "WIDE OPEN", "OPEN", "LIGHT CONTEST", "HEAVY CONTEST", "SMOTHERED",
)
# Loose, because OCR mangles these; the vocabulary is small enough that a low
# bar still separates its members from each other. Short terms need a higher
# bar, since a four-letter word matches far too much by chance.
MATCH_CUTOFF = 0.78
SHORT_TERM_CUTOFF = 0.88
SHORT_TERM_LENGTH = 6

DISTANCE_RE = re.compile(r"(\d{1,2})\s*['`’]\s*(\d{1,2})?")
LABELS = ("TIMING", "COVERAGE", "DISTANCE")


@dataclass
class ShotFeedback:
    """One shot's banner. Unread fields stay None."""

    timing: str | None = None
    coverage: str | None = None
    distance_feet: float | None = None
    raw_text: str = ""
    unmatched: list[str] = None            # values OCR'd but not recognised

    def __post_init__(self) -> None:
        if self.unmatched is None:
            self.unmatched = []

    @property
    def any_read(self) -> bool:
        return self.timing is not None or self.coverage is not None


class ShotFeedbackReader:
    """Detects and reads the shot feedback banner."""

    def __init__(self, *, present_threshold: float = 0.09) -> None:
        # Edge density in the band, above which a banner is considered present.
        self.present_threshold = present_threshold
        self._tesseract = None
        self._baseline: float | None = None

    def available(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract = pytesseract
            return True
        except Exception:                                   # noqa: BLE001
            return False

    def band(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x0, y0, x1, y1 = BANNER_BAND
        return image[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]

    def present(self, image: np.ndarray) -> bool:
        """Cheap check for whether a banner is on screen.

        Run on every sampled frame, so it must cost far less than reading one.
        The banner is dense synthetic text and rules on an otherwise
        low-detail strip of sky and crowd, so edge density separates it well.
        """
        band = self.band(image)
        if band.size == 0:
            return False
        return _edge_density(band) >= self.present_threshold

    def read_event(self, images: list[np.ndarray]) -> ShotFeedback:
        """Read one shot from several frames of its banner, by consensus.

        A single frame is not trustworthy: at capture resolution the banner
        text is small, and a marginal OCR pass fuzzy-matched against a closed
        vocabulary will confidently return the wrong verdict rather than
        nothing. But the banner stays up for the best part of a second, which
        is dozens of frames, and its errors are not correlated across them.
        Taking the most common reading turns many weak reads into one good one.

        A value is only accepted if it also wins by a margin, so an event whose
        frames genuinely disagree reports nothing rather than a coin flip.
        """
        readings = [self.read(image) for image in images]
        merged = ShotFeedback()
        merged.timing = _consensus([r.timing for r in readings])
        merged.coverage = _consensus([r.coverage for r in readings])
        distances = [r.distance_feet for r in readings if r.distance_feet]
        if distances:
            merged.distance_feet = float(np.median(distances))
        merged.raw_text = next((r.raw_text for r in readings if r.raw_text), "")
        if merged.timing is None or merged.coverage is None:
            merged.unmatched = sorted({t for r in readings for t in r.unmatched})
        return merged

    def read(self, image: np.ndarray) -> ShotFeedback:
        """Read the banner from one frame. Prefer `read_event` where possible."""
        if self._tesseract is None and not self.available():
            raise RuntimeError(
                "Tesseract is not available; the shot banner needs it.")

        band = self.band(image)
        text = self._ocr(band)
        feedback = ShotFeedback(raw_text=text)
        if not text:
            return feedback

        feedback.timing = _find_value(text, TIMING_VALUES)
        feedback.coverage = _find_value(text, COVERAGE_VALUES)
        feedback.distance_feet = _parse_distance(text)
        if feedback.timing is None or feedback.coverage is None:
            # Keep the leftovers so an unseen verdict can be spotted and the
            # vocabulary extended, rather than silently going missing.
            feedback.unmatched = _value_tokens(text)
        return feedback

    def _ocr(self, band: np.ndarray) -> str:
        """OCR the banner band.

        Inverting the upscaled greyscale outperforms thresholding here: the
        banner sits over crowd and sky, and any global threshold either keeps
        the background or eats the thinner strokes.
        """
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=4, fy=4,
                              interpolation=cv2.INTER_CUBIC)
        prepared = cv2.copyMakeBorder(cv2.bitwise_not(upscaled), 25, 25, 25, 25,
                                      cv2.BORDER_CONSTANT, value=255)
        best = ""
        for psm in (11, 6):
            try:
                text = self._tesseract.image_to_string(
                    prepared, config=f"--psm {psm}")
            except Exception:                               # noqa: BLE001
                log.exception("Shot banner OCR failed")
                continue
            text = " ".join(text.split())
            # Prefer whichever pass recovered more of the panel labels.
            if _label_hits(text) > _label_hits(best) or (
                    _label_hits(text) == _label_hits(best) and len(text) > len(best)):
                best = text
        return best


# A reading must be this share of the non-empty votes to be trusted.
CONSENSUS_SHARE = 0.5


def _consensus(values: list[str | None]) -> str | None:
    """Most common non-empty reading, if it commands a clear majority."""
    votes = [v for v in values if v]
    if not votes:
        return None
    value, count = collections.Counter(votes).most_common(1)[0]
    return value if count >= max(2, CONSENSUS_SHARE * len(votes)) else None


def _label_hits(text: str) -> int:
    upper = text.upper()
    return sum(1 for label in LABELS if label in upper)


def _value_tokens(text: str) -> list[str]:
    """Split banner text into candidate values, dropping the panel labels.

    Labels are fixed and carry no information; what matters is what sits under
    them. Anything short or numeric is dropped as OCR debris.
    """
    cleaned = re.sub(r"[^A-Za-z ]", " ", text.upper())
    for label in LABELS:
        cleaned = cleaned.replace(label, "|")
    tokens = []
    for chunk in cleaned.split("|"):
        chunk = " ".join(chunk.split())
        if len(chunk) >= 4:
            tokens.append(chunk)
    return tokens


def _find_value(text: str, vocabulary: tuple[str, ...]) -> str | None:
    """Find the best approximate occurrence of any known value in the text.

    Splitting the banner into panels first does not work: OCR reads the three
    labels as one line and the three values as another, so the values arrive
    concatenated with no label between them. Scanning the whole string for each
    known value sidesteps the layout entirely.

    Longer terms are tried first so "WIDE OPEN" wins over the "OPEN" inside it.
    Giving up matters as much as matching: an unrecognised verdict must surface
    as unknown rather than quietly become whichever known value is nearest.
    """
    cleaned = " ".join(re.sub(r"[^A-Za-z ]", " ", text.upper()).split())
    if not cleaned:
        return None

    # Score every term and take the best. Preferring longer terms instead
    # would read a plain "OPEN" as "WIDE OPEN", since the short term matches
    # perfectly inside the long one but not the reverse.
    best_term, best_score = None, 0.0
    for term in vocabulary:
        score = _best_window_ratio(cleaned, term)
        cutoff = (SHORT_TERM_CUTOFF if len(term) <= SHORT_TERM_LENGTH
                  else MATCH_CUTOFF)
        if score >= cutoff and score > best_score:
            best_term, best_score = term, score
    return best_term


def _best_window_ratio(text: str, term: str) -> float:
    """Best similarity between `term` and any similar-length window of `text`."""
    n = len(term)
    widths = {max(1, n + d) for d in (-2, -1, 0, 1, 2)}
    best = 0.0
    matcher = difflib.SequenceMatcher(a=term, autojunk=False)
    for start in range(max(1, len(text) - n + 3)):
        for width in widths:
            window = text[start:start + width]
            if len(window) < 3:
                continue
            matcher.set_seq2(window)
            # Cheap upper bound first; the real ratio is much more expensive.
            if matcher.real_quick_ratio() <= best:
                continue
            best = max(best, matcher.ratio())
    return best


def _parse_distance(text: str) -> float | None:
    """Read the distance panel, e.g. 24'6" -> 24.5 feet."""
    match = DISTANCE_RE.search(text)
    if not match:
        return None
    feet = int(match.group(1))
    inches = int(match.group(2)) if match.group(2) else 0
    if not (0 <= feet <= 94 and 0 <= inches < 12):
        return None
    return round(feet + inches / 12.0, 2)


def _edge_density(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    return float(np.count_nonzero(edges)) / float(edges.size)
