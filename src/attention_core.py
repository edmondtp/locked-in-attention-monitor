import time
from ultralytics import YOLO

VOTE_CAP = 16
LOCK_ENTER_THRESHOLD = 11
LOCK_EXIT_THRESHOLD = 3
HOLD_TO_LOCKED_IN = 0.85
HOLD_TO_NOT_LOCKED = 0.18
REENTRY_COOLDOWN_SEC = 0.85

PHONE_CONF_THRESHOLD = 0.35
PHONE_OVERRIDE_STICKY_SEC = 1.10


def detect_phone(model, frame, conf_threshold=PHONE_CONF_THRESHOLD):
    phone_detected = False
    best_conf = 0.0
    best_box = None

    results = model.predict(frame, verbose=False, conf=conf_threshold)

    for result in results:
        boxes = result.boxes
        names = result.names
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            cls_name = names[cls_id]

            if cls_name == "cell phone" and conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                best_conf = conf
                best_box = (x1, y1, x2, y2)
                phone_detected = True

    return phone_detected, best_conf, best_box


def classify_state(pose_data, face_data):
    if face_data["face_detected"]:
        if face_data["gaze_away_active"]:
            return "OFF_TASK"
        if face_data["talking_active"]:
            return "OFF_TASK"

    if not pose_data["pose_detected"]:
        return "OFF_TASK"

    norm_vert = pose_data["norm_vert"]
    norm_horiz = pose_data["norm_horiz"]

    if norm_vert < -0.03 and abs(norm_horiz) < 0.16 and not face_data["gaze_away_active"]:
        return "FACING"

    if pose_data["writing_active"]:
        return "WRITING"

    if pose_data["reading_candidate"] and not face_data["gaze_away_active"] and not face_data["talking_active"]:
        return "READING"

    return "OFF_TASK"


def build_reasons(raw_state, pose_data, face_data, phone_active):
    reasons = []

    if phone_active:
        reasons.append("Phone detected")
        return reasons

    if raw_state == "FACING":
        reasons.append("Focused on screen")
    elif raw_state == "READING":
        reasons.append("Reading posture")
    elif raw_state == "WRITING":
        reasons.append("Writing detected")
    else:
        if face_data["gaze_away_active"]:
            reasons.append("Looking away")
        if face_data["talking_active"]:
            reasons.append("Talking too much")
        if not reasons:
            reasons.append("Off task posture")

    return reasons


class AttentionStateMachine:
    def __init__(self):
        self.status = "NOT LOCKED IN"
        self.vote_score = 0
        self.candidate = None
        self.candidate_since = None
        self.last_not_locked_time = 0.0

    def update(self, raw_state, now, hard_off=False):
        if hard_off:
            self.status = "NOT LOCKED IN"
            self.vote_score = max(self.vote_score - 5, -VOTE_CAP)
            self.candidate = None
            self.candidate_since = None
            self.last_not_locked_time = now
            return self.status

        good_state = raw_state in ("FACING", "READING", "WRITING")
        can_reenter = (now - self.last_not_locked_time) >= REENTRY_COOLDOWN_SEC

        if self.status == "NOT LOCKED IN":
            if good_state:
                self.vote_score = min(self.vote_score + 1, VOTE_CAP)
            else:
                self.vote_score = max(self.vote_score - 2, -VOTE_CAP)

            intended = "LOCKED IN" if (self.vote_score >= LOCK_ENTER_THRESHOLD and can_reenter) else "NOT LOCKED IN"
        else:
            if good_state:
                self.vote_score = min(self.vote_score + 1, VOTE_CAP)
            else:
                self.vote_score = max(self.vote_score - 4, -VOTE_CAP)

            intended = "NOT LOCKED IN" if self.vote_score <= LOCK_EXIT_THRESHOLD else "LOCKED IN"

        if intended != self.status:
            if self.candidate != intended:
                self.candidate = intended
                self.candidate_since = now

            hold = HOLD_TO_LOCKED_IN if intended == "LOCKED IN" else HOLD_TO_NOT_LOCKED
            if (now - self.candidate_since) >= hold:
                self.status = intended
                if self.status == "NOT LOCKED IN":
                    self.last_not_locked_time = now
                self.candidate = None
                self.candidate_since = None
        else:
            self.candidate = None
            self.candidate_since = None

        return self.status


def create_phone_model():
    return YOLO("yolo11n.pt")