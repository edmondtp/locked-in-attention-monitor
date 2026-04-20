"""
session.py — session lifecycle, distraction logging, and scoring.

States:  IDLE → RUNNING ⇄ PAUSED → ENDED
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


# ── session states ────────────────────────────────────────────────────────────
IDLE    = "IDLE"
RUNNING = "RUNNING"
PAUSED  = "PAUSED"
ENDED   = "ENDED"


@dataclass
class DistractionEvent:
    start_ts: float
    end_ts: float | None = None
    cause: str           = "unknown"
    phone:  bool         = False

    @property
    def duration(self) -> float:
        if self.end_ts is None:
            return time.time() - self.start_ts
        return self.end_ts - self.start_ts


@dataclass
class SessionReport:
    duration_sec:       float
    focused_sec:        float
    distracted_sec:     float
    efficiency_pct:     float
    longest_streak_sec: float
    distraction_count:  int
    phone_count:        int
    score:              int          # the big "Locked In Score" (0-100)
    grade:              str          # S, A, B, C, D, F
    breakdown: dict     = field(default_factory=dict)


class SessionManager:
    """
    Owns session lifecycle + distraction history. Does not own the state
    machine — the caller passes in observations every frame.
    """

    def __init__(self):
        self.state = IDLE

        self._started_at:  float | None = None
        self._ended_at:    float | None = None
        self._pause_start: float | None = None
        self._total_paused: float       = 0.0

        self.events:              list[DistractionEvent] = []
        self._current_distraction: DistractionEvent | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        now = time.time()
        self.state         = RUNNING
        self._started_at   = now
        self._ended_at     = None
        self._pause_start  = None
        self._total_paused = 0.0
        self.events.clear()
        self._current_distraction = None

    def pause(self) -> None:
        if self.state != RUNNING:
            return
        self.state        = PAUSED
        self._pause_start = time.time()
        # close any open distraction — don't count pause time as distracted
        self._close_distraction()

    def resume(self) -> None:
        if self.state != PAUSED:
            return
        if self._pause_start is not None:
            self._total_paused += time.time() - self._pause_start
        self._pause_start = None
        self.state = RUNNING

    def end(self) -> SessionReport:
        if self.state == PAUSED and self._pause_start is not None:
            self._total_paused += time.time() - self._pause_start
            self._pause_start = None

        self._close_distraction()
        self._ended_at = time.time()
        self.state     = ENDED
        return self.report()

    def reset(self) -> None:
        self.__init__()

    # ── per-frame observation ─────────────────────────────────────────────────

    def observe(self, status: str, reasons: list[str], phone_active: bool) -> None:
        """Call this every frame while RUNNING. Tracks distraction intervals."""
        if self.state != RUNNING:
            return

        now = time.time()
        if status == "LOCKED OUT":
            if self._current_distraction is None:
                cause = reasons[0] if reasons else "off-task"
                self._current_distraction = DistractionEvent(
                    start_ts = now,
                    cause    = cause,
                    phone    = phone_active,
                )
        else:
            self._close_distraction()

    def _close_distraction(self) -> None:
        if self._current_distraction is not None:
            self._current_distraction.end_ts = time.time()
            # Only keep distractions longer than 0.3s to avoid noise
            if self._current_distraction.duration >= 0.3:
                self.events.append(self._current_distraction)
            self._current_distraction = None

    # ── derived props ─────────────────────────────────────────────────────────

    @property
    def elapsed(self) -> float:
        """Active session time, excluding pauses."""
        if self._started_at is None:
            return 0.0
        end = self._ended_at if self._ended_at else time.time()
        total = end - self._started_at - self._total_paused
        if self.state == PAUSED and self._pause_start is not None:
            total -= (time.time() - self._pause_start)
        return max(0.0, total)

    @property
    def is_running(self) -> bool:
        return self.state == RUNNING

    @property
    def is_paused(self) -> bool:
        return self.state == PAUSED

    # ── scoring ───────────────────────────────────────────────────────────────

    def compute_score(
        self,
        focused_sec: float,
        distracted_sec: float,
        longest_streak_sec: float,
    ) -> tuple[int, str, dict]:
        """
        The Locked-In Score algorithm.

        Combines three weighted components:

          • efficiency (60%) — focused_time / total_time
          • streak    (25%) — rewards long uninterrupted focus.
                               Cap scales with session length so short sessions
                               aren't punished: cap = min(session * 0.7, 600s).
          • consistency (15%) — penalises many distraction events
                                 (saturates around 2 distractions/min = 0 pts)

        Returns (score_0_to_100, letter_grade, breakdown_dict).
        """
        total = max(focused_sec + distracted_sec, 1.0)

        # 1) efficiency
        eff_ratio = focused_sec / total
        eff_score = eff_ratio * 100

        # 2) streak — adaptive cap.
        # For a 1-min session, hitting 42s of streak = 100pts.
        # For a 30-min session, you need 600s (10min) to max out.
        streak_cap = max(15.0, min(total * 0.7, 600.0))
        streak_score = min(longest_streak_sec / streak_cap, 1.0) * 100

        # 3) consistency — distractions per minute.
        # Also scales with session length: very short sessions shouldn't
        # be dinged for having 1 distraction.
        distractions_per_min = len(self.events) / max(total / 60.0, 0.5)
        if distractions_per_min <= 0.5:
            consistency_score = 100.0
        elif distractions_per_min >= 2.0:
            consistency_score = 0.0
        else:
            consistency_score = 100.0 * (1 - (distractions_per_min - 0.5) / 1.5)

        final = (
            0.60 * eff_score +
            0.25 * streak_score +
            0.15 * consistency_score
        )
        final_int = max(0, min(100, int(round(final))))

        if   final_int >= 90: grade = "S"
        elif final_int >= 80: grade = "A"
        elif final_int >= 70: grade = "B"
        elif final_int >= 60: grade = "C"
        elif final_int >= 50: grade = "D"
        else:                 grade = "F"

        breakdown = {
            "efficiency":  round(eff_score, 1),
            "streak":      round(streak_score, 1),
            "consistency": round(consistency_score, 1),
        }
        return final_int, grade, breakdown

    # ── report ────────────────────────────────────────────────────────────────

    def report(
        self,
        focused_sec: float       = None,
        longest_streak_sec: float = None,
    ) -> SessionReport:
        """
        Build a final report. If focused_sec / longest_streak_sec aren't provided,
        they're derived from distraction events.
        """
        elapsed = self.elapsed

        if focused_sec is None:
            distracted = sum(e.duration for e in self.events)
            focused_sec = max(0.0, elapsed - distracted)
        else:
            distracted = max(0.0, elapsed - focused_sec)

        if longest_streak_sec is None:
            longest_streak_sec = 0.0  # unknown

        eff_pct = 100.0 * focused_sec / max(elapsed, 1.0)
        phone_count = sum(1 for e in self.events if e.phone)

        score, grade, breakdown = self.compute_score(
            focused_sec, distracted, longest_streak_sec
        )

        return SessionReport(
            duration_sec       = elapsed,
            focused_sec        = focused_sec,
            distracted_sec     = distracted,
            efficiency_pct     = eff_pct,
            longest_streak_sec = longest_streak_sec,
            distraction_count  = len(self.events),
            phone_count        = phone_count,
            score              = score,
            grade              = grade,
            breakdown          = breakdown,
        )