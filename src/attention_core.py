"""
attention_core.py — state machine, classification, and phone detection.
"""
import time
from ultralytics import YOLO

# ── tuning constants ──────────────────────────────────────────────────────────
VOTE_CAP               = 16
LOCK_ENTER_THRESHOLD   = 11
LOCK_EXIT_THRESHOLD    = 3
HOLD_TO_LOCKED_IN      = 0.85   # seconds
HOLD_TO_NOT_LOCKED     = 0.18   # seconds
REENTRY_COOLDOWN_SEC   = 0.85

PHONE_CONF_THRESHOLD     = 0.35
PHONE_OVERRIDE_STICKY_SEC = 1.10


# ── phone detection ───────────────────────────────────────────────────────────

def detect_phone(model, frame, conf_threshold: float = PHONE_CONF_THRESHOLD):
    """Return (detected: bool, best_conf: float, best_box: tuple|None)."""
    phone_detected = False
    best_conf      = 0.0
    best_box       = None

    results = model.predict(frame, verbose=False, conf=conf_threshold)
    for result in results:
        boxes = result.boxes
        names = result.names
        if boxes is None:
            continue
        for box in boxes:
            cls_id   = int(box.cls.item())
            conf     = float(box.conf.item())
            cls_name = names[cls_id]
            if cls_name == "cell phone" and conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                best_conf      = conf
                best_box       = (x1, y1, x2, y2)
                phone_detected = True

    return phone_detected, best_conf, best_box


# ── raw state classification ──────────────────────────────────────────────────

def classify_state(pose_data: dict, face_data: dict) -> str:
    """
    Returns one of: FACING | READING | WRITING | OFF_TASK

    Priority order:
      1. Face-based distractions (gaze away / talking)  → OFF_TASK
      2. No pose                                         → OFF_TASK
      3. Looking at screen (FACING)
      4. Writing motion
      5. Reading posture
      6. Fallthrough                                     → OFF_TASK
    """
    if face_data["face_detected"]:
        if face_data["gaze_away_active"] or face_data["talking_active"]:
            return "OFF_TASK"

    if not pose_data["pose_detected"]:
        return "OFF_TASK"

    nv = pose_data["norm_vert"]
    nh = pose_data["norm_horiz"]

    if nv < -0.03 and abs(nh) < 0.16 and not face_data["gaze_away_active"]:
        return "FACING"

    if pose_data["writing_active"]:
        return "WRITING"

    if (pose_data["reading_candidate"]
            and not face_data["gaze_away_active"]
            and not face_data["talking_active"]):
        return "READING"

    return "OFF_TASK"


# ── reason builder ────────────────────────────────────────────────────────────

_STATE_REASONS = {
    "FACING":  ["Focused on screen", "Eye contact maintained", "Good posture"],
    "READING": ["Reading posture", "Head angle correct", "Eyes tracking text"],
    "WRITING": ["Writing motion detected", "Wrist active", "Head angle correct"],
}


def build_reasons(raw_state: str, pose_data: dict, face_data: dict, phone_active: bool) -> list[str]:
    if phone_active:
        return ["Phone detected", "Device distraction active"]

    if raw_state in _STATE_REASONS:
        return _STATE_REASONS[raw_state]

    # OFF_TASK — be specific about why
    reasons = []
    if face_data["gaze_away_active"]:
        reasons.append("Looking away from screen")
    if face_data["talking_active"]:
        reasons.append("Talking detected")
    if face_data["eyes_closed"]:
        reasons.append("Eyes closed")
    if not pose_data["pose_detected"]:
        reasons.append("No pose detected")
    if not reasons:
        reasons.append("Off-task posture")
    return reasons


# ── state machine ─────────────────────────────────────────────────────────────

class AttentionStateMachine:
    def __init__(self):
        self.status               = "LOCKED OUT"
        self.vote_score           = 0
        self.candidate            = None
        self.candidate_since      = None
        self.last_not_locked_time = 0.0

        # Session & streak tracking
        self.session_start        = time.time()
        self.locked_in_seconds    = 0.0
        self.last_update_time     = time.time()
        self.current_streak_start = None   # when the current streak began
        self.longest_streak_sec   = 0.0

    def reset(self) -> None:
        """Reset all session-level counters. Call when starting a new session."""
        now = time.time()
        self.status               = "LOCKED OUT"
        self.vote_score           = 0
        self.candidate            = None
        self.candidate_since      = None
        self.last_not_locked_time = 0.0
        self.session_start        = now
        self.locked_in_seconds    = 0.0
        self.last_update_time     = now
        self.current_streak_start = None
        self.longest_streak_sec   = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def attention_score(self) -> int:
        """0-100 score derived from vote_score. Useful for the UI meter."""
        raw = (self.vote_score + VOTE_CAP) / (2 * VOTE_CAP)   # 0.0 – 1.0
        return max(0, min(100, int(raw * 100)))

    @property
    def session_elapsed(self) -> float:
        return time.time() - self.session_start

    @property
    def current_streak(self) -> float:
        """How long the user has been continuously locked-in (seconds)."""
        if self.status == "LOCKED IN" and self.current_streak_start is not None:
            return time.time() - self.current_streak_start
        return 0.0

    # ── main update ───────────────────────────────────────────────────────────

    def update(self, raw_state: str, now: float, hard_off: bool = False) -> str:
        dt = now - self.last_update_time
        self.last_update_time = now

        # Accumulate locked-in time
        if self.status == "LOCKED IN":
            self.locked_in_seconds += dt

        if hard_off:
            self._force_out(now)
            return self.status

        good_state  = raw_state in ("FACING", "READING", "WRITING")
        can_reenter = (now - self.last_not_locked_time) >= REENTRY_COOLDOWN_SEC

        if self.status == "LOCKED OUT":
            self.vote_score = min(self.vote_score + 1, VOTE_CAP) if good_state \
                else max(self.vote_score - 2, -VOTE_CAP)
            intended = "LOCKED IN" if (self.vote_score >= LOCK_ENTER_THRESHOLD and can_reenter) \
                else "LOCKED OUT"
        else:
            self.vote_score = min(self.vote_score + 1, VOTE_CAP) if good_state \
                else max(self.vote_score - 4, -VOTE_CAP)
            intended = "LOCKED OUT" if self.vote_score <= LOCK_EXIT_THRESHOLD else "LOCKED IN"

        self._apply_transition(intended, now)
        return self.status

    # ── private ───────────────────────────────────────────────────────────────

    def _force_out(self, now: float) -> None:
        self._end_streak()
        self.status      = "LOCKED OUT"
        self.vote_score  = max(self.vote_score - 5, -VOTE_CAP)
        self.candidate   = None
        self.candidate_since = None
        self.last_not_locked_time = now

    def _apply_transition(self, intended: str, now: float) -> None:
        if intended != self.status:
            if self.candidate != intended:
                self.candidate       = intended
                self.candidate_since = now

            hold = HOLD_TO_LOCKED_IN if intended == "LOCKED IN" else HOLD_TO_NOT_LOCKED
            if (now - self.candidate_since) >= hold:
                prev_status    = self.status
                self.status    = intended
                self.candidate = None
                self.candidate_since = None

                if self.status == "LOCKED IN":
                    self.current_streak_start = now
                elif prev_status == "LOCKED IN":
                    self._end_streak()
                    self.last_not_locked_time = now
        else:
            self.candidate       = None
            self.candidate_since = None

    def _end_streak(self) -> None:
        if self.current_streak_start is not None:
            streak = time.time() - self.current_streak_start
            if streak > self.longest_streak_sec:
                self.longest_streak_sec = streak
        self.current_streak_start = None


# ── model factory ─────────────────────────────────────────────────────────────

def create_phone_model():
    return YOLO("yolo11n.pt")