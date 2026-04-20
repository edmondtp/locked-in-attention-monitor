"""
audio.py — non-intrusive audio cues.

Generates two short sounds procedurally (no external files):

  • chime()  — soft two-note rising "ding" played when LOCKED IN engages.
  • buzz()   — low soft single-note hum played when LOCKED OUT engages.

Uses sounddevice if available, falls back to printed bell. Rate-limited
internally so it can't spam — at most one cue per 4 seconds.

Threaded so nothing blocks the main loop.
"""
from __future__ import annotations

import threading
import time

try:
    import numpy as np
    import sounddevice as sd
    _AUDIO_OK = True
except Exception:
    _AUDIO_OK = False


_SAMPLE_RATE    = 44100
_MIN_INTERVAL   = 4.0        # seconds between cues


class _Cue:
    """Pre-rendered audio buffer, played via sounddevice."""
    def __init__(self, samples):
        self.samples = samples

    def play(self):
        if not _AUDIO_OK:
            return
        try:
            sd.play(self.samples, _SAMPLE_RATE, blocking=False)
        except Exception:
            pass


def _fade(arr, fade_sec=0.03):
    n = int(fade_sec * _SAMPLE_RATE)
    n = min(n, len(arr) // 2)
    if n <= 1:
        return arr
    ramp = np.linspace(0, 1, n)
    arr[:n]       *= ramp
    arr[-n:]      *= ramp[::-1]
    return arr


def _make_chime() -> _Cue:
    """Two-note rising ding — E5 then A5, ~0.45s total."""
    if not _AUDIO_OK:
        return _Cue(None)

    def tone(freq, dur, amp=0.22):
        t = np.linspace(0, dur, int(_SAMPLE_RATE * dur), endpoint=False)
        wave = amp * np.sin(2 * np.pi * freq * t)
        # Slight exponential decay
        wave *= np.exp(-t * 4.5)
        return _fade(wave.astype(np.float32), 0.015)

    n1 = tone(659.25, 0.20)    # E5
    gap = np.zeros(int(0.02 * _SAMPLE_RATE), dtype=np.float32)
    n2 = tone(880.00, 0.25)    # A5
    samples = np.concatenate([n1, gap, n2])
    # Stereo
    samples = np.stack([samples, samples], axis=-1)
    return _Cue(samples)


def _make_buzz() -> _Cue:
    """Soft low hum — A3 sine, ~0.35s, low volume, gentle fade."""
    if not _AUDIO_OK:
        return _Cue(None)

    dur  = 0.35
    t    = np.linspace(0, dur, int(_SAMPLE_RATE * dur), endpoint=False)
    freq = 220.0  # A3
    # Slight detuned overtone for a warmer feel
    wave = 0.14 * np.sin(2 * np.pi * freq * t)
    wave += 0.05 * np.sin(2 * np.pi * freq * 0.5 * t)
    wave *= np.exp(-t * 2.8)
    wave = _fade(wave.astype(np.float32), 0.05)
    samples = np.stack([wave, wave], axis=-1)
    return _Cue(samples)


class AudioCues:
    """Singleton-style wrapper with rate limiting + thread dispatch."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and _AUDIO_OK
        self._chime  = _make_chime() if self.enabled else None
        self._buzz   = _make_buzz()  if self.enabled else None
        self._last_played_at = 0.0
        self._last_kind      = None
        self._lock           = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled and _AUDIO_OK

    def _play(self, cue: _Cue, kind: str):
        """Fire-and-forget playback in a thread."""
        if not self.enabled or cue is None:
            return

        with self._lock:
            now = time.time()
            # Rate limit — never more than one cue per _MIN_INTERVAL, and
            # never two cues of the same kind back-to-back.
            if (now - self._last_played_at) < _MIN_INTERVAL:
                return
            if kind == self._last_kind:
                # Only allow repeat if enough time has passed
                if (now - self._last_played_at) < _MIN_INTERVAL * 2:
                    return
            self._last_played_at = now
            self._last_kind      = kind

        threading.Thread(target=cue.play, daemon=True).start()

    def chime(self) -> None:
        self._play(self._chime, "chime")

    def buzz(self) -> None:
        self._play(self._buzz, "buzz")


def on_status_change(cues: AudioCues, old: str, new: str) -> None:
    """Helper: trigger the right cue when status transitions."""
    if old == new:
        return
    if new == "LOCKED IN":
        cues.chime()
    elif new == "LOCKED OUT":
        cues.buzz()