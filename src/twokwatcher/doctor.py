"""Preflight checks.

The failure this exists to prevent is discovering a broken setup at 9pm with a
game already running. Everything here is cheap and read-only, and each failure
says what to do about it rather than only that something is wrong.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, EXAMPLE_CONFIG_PATH, Config

# Regions the pipeline needs before it can do anything beyond state tracking.
ESSENTIAL_REGIONS = ("scoreboard", "game_clock")

# A session of collection wants room; PNG frames are a few MB each.
MIN_FREE_GB = 2.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str | None = None
    warning: bool = False   # a problem worth saying, but not a blocker


def run_checks(config_path: Path | None = None,
               db_path: Path | None = None) -> list[Check]:
    checks: list[Check] = []

    checks.append(Check(
        "Python", sys.version_info >= (3, 11),
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}",
        fix="This project needs Python 3.11 or newer.",
    ))

    for module, extra in (("cv2", None), ("numpy", None), ("yaml", None),
                          ("webview", "app")):
        try:
            __import__(module)
            checks.append(Check(f"import {module}", True, "installed"))
        except ImportError:
            optional = extra is not None
            checks.append(Check(
                f"import {module}", False,
                "missing" + (" (optional)" if optional else ""),
                fix=(f'pip install -e ".[{extra}]"' if optional
                     else 'pip install -e "."'),
                warning=optional,
            ))

    # Capture devices. The usual failure is OBS not running, or its virtual
    # camera not started, both of which look identical from here.
    try:
        from .capture import list_devices
        devices = list_devices(max_index=6)
        checks.append(Check(
            "Capture device", bool(devices),
            (", ".join(f"[{d['index']}] {d['width']}x{d['height']}"
                       for d in devices) if devices else "none found"),
            fix="Start OBS, add the Elgato as a source, then click "
                "'Start Virtual Camera'.",
        ))
    except Exception as exc:                               # noqa: BLE001
        checks.append(Check("Capture device", False, f"probe failed: {exc}",
                            fix="Check that OpenCV installed correctly."))

    # Config and calibration.
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    using_example = not path.exists()
    checks.append(Check(
        "Region config", path.exists(),
        str(path) if path.exists() else f"missing, falling back to "
                                        f"{EXAMPLE_CONFIG_PATH}",
        fix=f"copy {EXAMPLE_CONFIG_PATH} {path}",
        warning=True,
    ))

    try:
        config = Config.load(config_path)
        missing = [r for r in ESSENTIAL_REGIONS if r not in config.regions]
        checks.append(Check(
            "Essential regions", not missing,
            "all present" if not missing else f"missing {', '.join(missing)}",
            fix="Open `2kw app` and drag boxes in the Calibrate panel.",
        ))
        # Placeholder coordinates are worse than missing ones: they look valid
        # and silently read the wrong pixels.
        checks.append(Check(
            "Regions calibrated", not using_example,
            "fitted to your capture" if not using_example
            else "using shipped placeholders",
            fix="Placeholder coordinates will read the wrong pixels. "
                "Calibrate in the app before trusting any value.",
            warning=True,
        ))
    except Exception as exc:                               # noqa: BLE001
        checks.append(Check("Region config parses", False, str(exc),
                            fix="Fix or delete config/regions.yaml."))

    # Glyph atlas — the reason values read as "not read".
    atlas = Path("data/atlas")
    glyphs = sorted(atlas.glob("*.png")) if atlas.exists() else []
    checks.append(Check(
        "Glyph atlas", bool(glyphs),
        f"{len(glyphs)} glyph(s)" if glyphs else "not built",
        fix="Expected for now — scoreboard values read as 'not read' until "
            "this exists. Collect frames tonight and it can be built from them.",
        warning=True,
    ))

    # Writable paths and room to collect into.
    target = Path(db_path) if db_path else Path("data/2kwatcher.db")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        probe = target.parent / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        checks.append(Check("Data directory", True, f"{target.parent} writable"))
    except OSError as exc:
        checks.append(Check("Data directory", False, str(exc),
                            fix="Run from a directory you can write to."))

    try:
        free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
        checks.append(Check(
            "Free disk", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB",
            fix=f"Frame collection wants at least {MIN_FREE_GB:.0f} GB free.",
            warning=True,
        ))
    except OSError:
        pass

    return checks


def format_report(checks: list[Check]) -> tuple[str, bool]:
    """Render the checks. Returns the text and whether anything blocking failed."""
    lines = []
    blocking = False
    for check in checks:
        if check.ok:
            mark = "PASS"
        elif check.warning:
            mark = "WARN"
        else:
            mark = "FAIL"
            blocking = True
        lines.append(f"  [{mark}]  {check.name:<20} {check.detail}")
        if not check.ok and check.fix:
            lines.append(f"           -> {check.fix}")

    lines.append("")
    lines.append("Ready to watch." if not blocking
                 else "Not ready — fix the FAIL items above.")
    return "\n".join(lines), blocking
