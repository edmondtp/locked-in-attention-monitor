"""
utils.py — shared signal processing primitives.
Import from here instead of duplicating across modules.
"""
from collections import deque


class EMA:
    """Exponential moving average."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class RollingMean:
    """Simple rolling mean over the last *n* samples."""

    def __init__(self, n: int):
        self.buf: deque[float] = deque(maxlen=n)

    def update(self, x: float) -> float:
        self.buf.append(x)
        return sum(self.buf) / len(self.buf) if self.buf else 0.0

    @property
    def value(self) -> float:
        return sum(self.buf) / len(self.buf) if self.buf else 0.0


class VotingWindow:
    """Binary voting window — returns True when enough recent frames vote yes."""

    def __init__(self, maxlen: int, threshold_frac: float = 0.65):
        self.buf: deque[int] = deque(maxlen=maxlen)
        self.threshold_frac = threshold_frac

    def push(self, val: bool) -> bool:
        self.buf.append(1 if val else 0)
        return self.active

    @property
    def votes(self) -> int:
        return sum(self.buf)

    @property
    def active(self) -> bool:
        if not self.buf:
            return False
        return self.votes >= int(len(self.buf) * self.threshold_frac)