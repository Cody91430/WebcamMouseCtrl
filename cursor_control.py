"""
Standalone cursor controller with a fast Win32 backend (SetCursorPos) to reduce overhead.
Can be used independently of the hand-tracking pipeline.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import pyautogui

    pyautogui.FAILSAFE = False
except Exception:
    pyautogui = None

if sys.platform == "win32":
    try:
        import ctypes

        _user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    except Exception:
        _user32 = None
else:
    _user32 = None


@dataclass
class CursorSettings:
    """Configuration for smoothing and rate limiting."""

    max_rate_hz: float = 240.0  # 0 for unlimited
    smooth: bool = True
    smooth_alpha: float = 0.5   # higher = more reactive
    min_move_px: float = 0.2    # ignore tiny moves to cut jitter
    max_step_px: float = 0.0    # 0 to allow full jumps (faster travel)
    backend: str = "auto"      # "win32", "pyautogui", or "auto"


class CursorController:
    """Small helper to move the OS cursor from normalized coords (0..1)."""

    def __init__(self, settings: CursorSettings | None = None) -> None:
        self.settings = settings or CursorSettings()
        self.screen_size: Optional[Tuple[int, int]] = None
        self._last_move_ts: float = 0.0
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None
        self._last_sent: Optional[Tuple[float, float]] = None
        self._backend = self._choose_backend()

        self.screen_size = self._get_screen_size()
        if self.screen_size is None:
            print("[warn] No cursor backend available; cursor moves will be ignored.")

    def _choose_backend(self) -> str:
        if self.settings.backend == "win32" and _user32:
            return "win32"
        if self.settings.backend == "pyautogui" and pyautogui is not None:
            return "pyautogui"
        if _user32:
            return "win32"
        if pyautogui is not None:
            return "pyautogui"
        return "none"

    def _get_screen_size(self) -> Optional[Tuple[int, int]]:
        if self._backend == "win32" and _user32:
            try:
                w = _user32.GetSystemMetrics(0)
                h = _user32.GetSystemMetrics(1)
                return int(w), int(h)
            except Exception:
                pass
        if self._backend == "pyautogui" and pyautogui is not None:
            try:
                size = pyautogui.size()
                return size[0], size[1]
            except Exception:
                pass
        return None

    def _set_cursor_pos(self, x: float, y: float) -> None:
        if self._backend == "win32" and _user32:
            try:
                _user32.SetCursorPos(int(x), int(y))
                return
            except Exception:
                pass
        if self._backend == "pyautogui" and pyautogui is not None:
            try:
                pyautogui.moveTo(x, y, duration=0)
            except Exception:
                pass

    def move_normalized(self, norm_x: float, norm_y: float) -> None:
        """Move cursor given normalized coordinates in [0,1]."""
        if self.screen_size is None or self._backend == "none":
            return

        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        sx = norm_x * self.screen_size[0]
        sy = norm_y * self.screen_size[1]

        if self.settings.smooth:
            if self._smooth_x is None:
                self._smooth_x, self._smooth_y = sx, sy
            else:
                a = self.settings.smooth_alpha
                self._smooth_x = a * sx + (1 - a) * self._smooth_x
                self._smooth_y = a * sy + (1 - a) * self._smooth_y
            sx, sy = self._smooth_x, self._smooth_y

        now = time.time()
        if self.settings.max_rate_hz > 0:
            min_dt = 1.0 / self.settings.max_rate_hz
            if now - self._last_move_ts < min_dt:
                return

        if self._last_sent is not None:
            dx = sx - self._last_sent[0]
            dy = sy - self._last_sent[1]
            if (dx * dx + dy * dy) ** 0.5 < self.settings.min_move_px:
                return
            if self.settings.max_step_px > 0:
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self.settings.max_step_px:
                    scale = self.settings.max_step_px / dist
                    sx = self._last_sent[0] + dx * scale
                    sy = self._last_sent[1] + dy * scale

        self._set_cursor_pos(sx, sy)
        self._last_move_ts = now
        self._last_sent = (sx, sy)


if __name__ == "__main__":
    # Minimal demo: move cursor to center.
    ctrl = CursorController()
    ctrl.move_normalized(0.5, 0.5)
    print("Moved cursor to center (0.5, 0.5).")
