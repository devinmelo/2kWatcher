"""The native window.

pywebview renders the same local UI in an OS window, so there is one interface
to build rather than two — and the server stays reachable from a phone or a
tablet on the same machine's network stack, and later from OBS as a browser
source, without any of that being a separate build.

If pywebview is not installed the app still runs; it just opens in the default
browser instead. Degrading to a browser tab beats refusing to start.
"""

from __future__ import annotations

import logging
import webbrowser

from ..config import Config
from .server import AppServer

log = logging.getLogger(__name__)


def launch(config: Config, db_path, *, port: int = 8770,
           window: bool = True) -> int:
    app = AppServer(config, db_path, port=port)
    app.start_background()
    print(f"2kWatcher is serving at {app.url}")

    if not window:
        webbrowser.open(app.url)
        try:
            input("Running. Press Enter to quit.\n")
        except (EOFError, KeyboardInterrupt):
            pass
        app.shutdown()
        return 0

    try:
        import webview  # pywebview
    except ImportError:
        print("pywebview is not installed — opening in your browser instead.")
        print("For the standalone window:  pip install pywebview")
        return launch(config, db_path, port=port, window=False)

    webview.create_window("2kWatcher", app.url, width=1180, height=780,
                          min_size=(820, 560))
    try:
        webview.start()
    finally:
        app.shutdown()
    return 0
