"""Locating Tesseract.

The Windows installer does not add Tesseract to PATH, so pytesseract cannot
find it even when it is correctly installed. Rather than making that a setup
step people have to discover from a stack trace, look in the standard install
locations and point pytesseract at whatever is there.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)

WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)
UNIX_CANDIDATES = (
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)

_configured = False


def find_tesseract() -> str | None:
    """Return a usable tesseract path, or None."""
    # An explicit override always wins.
    override = os.environ.get("TESSERACT_CMD")
    if override and Path(override).exists():
        return override

    on_path = shutil.which("tesseract")
    if on_path:
        return on_path

    candidates = (WINDOWS_CANDIDATES if sys.platform == "win32"
                  else UNIX_CANDIDATES)
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def configure() -> bool:
    """Point pytesseract at the binary. Returns whether OCR is usable.

    Safe to call repeatedly; the search only runs once.
    """
    global _configured
    try:
        import pytesseract
    except ImportError:
        return False

    if not _configured:
        _configured = True
        found = find_tesseract()
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
            log.debug("Using tesseract at %s", found)

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:                                       # noqa: BLE001
        return False
