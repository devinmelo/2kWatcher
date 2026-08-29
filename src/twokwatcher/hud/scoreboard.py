"""Parsing the scoreboard bug.

A note on approach: the instinct is to reach for Tesseract or PaddleOCR here,
and it is the wrong instinct. General OCR is built for arbitrary fonts at
arbitrary scales and is both slow and unreliable on small HUD digits. 2K's
scoreboard is one fixed font, at one fixed size, in one fixed position, drawing
exactly eleven glyphs (0-9 and ':').

That is a template-matching problem, not an OCR problem. Build the glyph atlas
once from a handful of frames and matching becomes a few microseconds per field
with essentially perfect accuracy. `2kw atlas` is the intended path for
extracting those glyphs; until an atlas exists, `ScoreboardReader` returns None
rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import Config

DEFAULT_ATLAS_DIR = Path("data/atlas")


@dataclass
class Scoreboard:
    """One reading of the scoreboard. Any field may be None if unparsed."""

    score_home: int | None = None
    score_away: int | None = None
    game_clock: str | None = None
    quarter: int | None = None
    shot_clock: int | None = None

    @property
    def complete(self) -> bool:
        return None not in (self.score_home, self.score_away, self.game_clock)


def preprocess_digits(crop: np.ndarray, *, scale: int = 3) -> np.ndarray:
    """Normalize a HUD crop into a clean binary image of its glyphs.

    2K draws the scoreboard as bright text over a dark translucent plate, so a
    plain threshold on luminance separates glyph from background well. Upscaling
    first keeps thin strokes from being eaten by the threshold.
    """
    if crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Otsu adapts to the plate opacity, which varies between arenas and modes.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Glyphs should be white on black; invert if we got it backwards.
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    return binary


class ScoreboardReader:
    """Reads scoreboard fields from a frame using a glyph atlas.

    The atlas is a directory of single-glyph PNGs named for what they depict
    (`0.png` ... `9.png`, `colon.png`), cropped from your own footage. Without
    one, every read returns None — a scaffold that silently invents scores would
    be worse than one that admits it cannot read yet.
    """

    def __init__(self, config: Config, atlas_dir: Path = DEFAULT_ATLAS_DIR) -> None:
        self.config = config
        self.atlas_dir = Path(atlas_dir)
        self.atlas = self._load_atlas()

    @property
    def ready(self) -> bool:
        return bool(self.atlas)

    def _load_atlas(self) -> dict[str, np.ndarray]:
        if not self.atlas_dir.exists():
            return {}
        atlas: dict[str, np.ndarray] = {}
        for png in sorted(self.atlas_dir.glob("*.png")):
            glyph = ":" if png.stem == "colon" else png.stem
            image = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                atlas[glyph] = image
        return atlas

    def read(self, image: np.ndarray) -> Scoreboard:
        if not self.ready:
            return Scoreboard()
        return Scoreboard(
            score_home=self._read_int(image, "score_home"),
            score_away=self._read_int(image, "score_away"),
            game_clock=self._read_text(image, "game_clock"),
            shot_clock=self._read_int(image, "shot_clock"),
        )

    def _read_int(self, image: np.ndarray, region: str) -> int | None:
        text = self._read_text(image, region)
        if text is None:
            return None
        digits = "".join(c for c in text if c.isdigit())
        return int(digits) if digits else None

    def _read_text(self, image: np.ndarray, region: str) -> str | None:
        """Segment a field into glyphs and match each against the atlas."""
        binary = preprocess_digits(self.config.region(region).crop(image))
        return match_glyphs(binary, self.atlas)


# Below this a glyph is being matched against noise, and refusing to read beats
# guessing. Shared by every atlas-backed field, wherever it is drawn.
GLYPH_MATCH_CUTOFF = 0.55


def match_glyphs(binary: np.ndarray, atlas: dict[str, np.ndarray], *,
                 cutoff: float = GLYPH_MATCH_CUTOFF) -> str | None:
    """Segment a preprocessed field into glyphs and match each to the atlas.

    Returns None if any glyph fails to match convincingly: a field read as
    "1?" is worth nothing, so a partial read is treated as no read at all.
    """
    glyphs = _segment_glyphs(binary)
    if not glyphs:
        return None

    out = []
    for glyph in glyphs:
        best_score, best_char = 0.0, None
        for char, template in atlas.items():
            resized = cv2.resize(glyph, (template.shape[1], template.shape[0]))
            score = float(
                cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED).max()
            )
            if score > best_score:
                best_score, best_char = score, char
        if best_char is None or best_score < cutoff:
            return None
        out.append(best_char)
    return "".join(out)


def _segment_glyphs(binary: np.ndarray, min_area: int = 20) -> list[np.ndarray]:
    """Split a binary field image into individual glyph crops, left to right."""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    boxes.sort(key=lambda b: b[0])
    return [binary[y:y + h, x:x + w] for x, y, w, h in boxes]
