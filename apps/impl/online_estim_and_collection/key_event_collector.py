from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Slot, Signal, QSize
from PySide6.QtGui import QGuiApplication, QMovie
from PySide6.QtWidgets import QLabel, QWidget

from pynput import keyboard


class _GifOverlay(QWidget):
    """Frameless, click-through, transparent window that shows only a GIF for a moment."""

    def __init__(self, gif_path: str):
        super().__init__(parent=None)  # GUI thread only, top-level window
        self.setWindowFlags(
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # Click-through (Qt ≥ 6.5)
        if hasattr(Qt, "WindowTransparentForInput"):
            self.setWindowFlag(Qt.WindowTransparentForInput, True)
        else:  # fallback
            self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._label = QLabel(self)
        self._label.setAttribute(Qt.WA_TranslucentBackground, True)
        self._label.setStyleSheet("background: transparent; border: none;")
        self._label.setAlignment(Qt.AlignCenter)

        self._movie = QMovie(gif_path)
        self._movie.setCacheMode(QMovie.CacheAll)
        self._label.setMovie(self._movie)

        # Pre-size to first frame
        self._movie.jumpToFrame(0)
        sz = self._frame_size()
        self.resize(sz)
        self._label.resize(sz)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._on_timeout)

    def _frame_size(self) -> QSize:
        img_sz = self._movie.currentImage().size()
        if not img_sz.isEmpty():
            return img_sz
        rect_sz = self._movie.frameRect().size()
        if not rect_sz.isEmpty():
            return rect_sz
        return QSize(200, 200)

    def show_for_one_second_on_screen_top(self, screen_index: int, top_margin: int = 10):
        screens = QGuiApplication.screens()
        if not screens:
            return
        if screen_index < 0 or screen_index >= len(screens):
            screen_index = 0

        screen_geo = screens[screen_index].geometry()
        size = self._frame_size()

        x = screen_geo.x() + (screen_geo.width() - size.width()) // 2
        y = screen_geo.y() + top_margin

        self._label.resize(size)
        self.setGeometry(x, y, size.width(), size.height())

        self._movie.stop()
        self._movie.jumpToFrame(0)
        self._movie.start()

        self.show()
        self.raise_()
        self._hide_timer.start(1000)

    def _on_timeout(self):
        self._movie.stop()
        self.hide()


class KeyEventTraceWorker(QObject):
    """
    Global Right-Arrow key trace + 1s GIF overlay on a chosen screen.
    - Works when the app window is unfocused (via pynput).
    - Records Unix ms timestamps.
    - @Slot(str) dump_to_json(filename): writes JSON list into the configured directory.

    Notes:
      * GUI (GIF overlay) stays in the main Qt thread.
      * Pynput runs a background (non-Qt) thread. We bounce to Qt via a signal.
    """

    # Use Python object (no 32-bit overflow)
    triggered = Signal(object)
    _hotkey_ping = Signal()  # thread-hop from pynput → Qt main thread

    def __init__(self, storage_dir: str | Path, screen_index: int, gif_path: str | Path, enabled: bool = True):
        super().__init__()
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._screen_index = int(screen_index)
        self._overlay = _GifOverlay(str(gif_path))
        self._timestamps_ms: list[int] = []

        # Deliver to main thread
        self._hotkey_ping.connect(self._handle_trigger, Qt.QueuedConnection)

        self._listener: keyboard.Listener | None = None
        if enabled:
            self.start()

    # ---------- Public API ----------

    def start(self):
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(on_press=self._on_key_press)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        try:
            if self._listener:
                self._listener.stop()
        finally:
            self._listener = None

    def set_screen_index(self, idx: int):
        self._screen_index = int(idx)

    def timestamps(self) -> list[int]:
        return list(self._timestamps_ms)

    def clear(self):
        self._timestamps_ms.clear()

    @Slot(str)
    def dump_to_json(self, filename: str) -> None:
        """
        Slot: write timestamps to <storage_dir>/<base>.json
        Accepts either a bare name ("session_001") or any path (".../session_001.npz").
        """
        if not filename:
            raise ValueError("filename must be a non-empty string")

        p = Path(filename)
        base = p.stem or p.name
        out_name = base if base.lower().endswith(".json") else f"{base}.json"
        out_path = self._storage_dir / out_name

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._timestamps_ms, f)
        self.clear()
        tmp_path.replace(out_path)

    # ---------- Non-Qt thread (pynput) ----------

    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.right:
                self._hotkey_ping.emit()  # queued to GUI thread
        except Exception:
            pass

    # ---------- Qt main thread ----------

    @Slot()
    def _handle_trigger(self):
        ts_ms = time.time_ns() // 1_000_000
        self._timestamps_ms.append(ts_ms)
        self.triggered.emit(ts_ms)
        print(f"Key event triggered at {ts_ms} ms")
        self._overlay.show_for_one_second_on_screen_top(self._screen_index)

# --- Minimal demo / manual test ---
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     worker = GlobalRightArrowWorker(
#         storage_dir="apps/impl/dataset_collection/traces/userstudy",
#         screen_index=0,
#         gif_path="apps/impl/online_estim_and_collection/dissatisfied.gif",  # Replace with a valid GIF path
#     )

#     print("Running. Press Right Arrow anywhere to trigger. Close this process to quit.")
#     sys.exit(app.exec())
