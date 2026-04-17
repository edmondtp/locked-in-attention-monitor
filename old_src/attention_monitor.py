# src/attention_monitor.py
import cv2
import time
from collections import deque
from ultralytics import YOLO

from face_features import FaceFeatureExtractor
from pose_features import PoseFeatureExtractor
from ui import render_ui, GREEN, RED, YELLOW, BLUE, PURPLE

VOTE_CAP = 16
LOCK_ENTER_THRESHOLD = 11
LOCK_EXIT_THRESHOLD = 3
HOLD_TO_LOCKED_IN = 0.85
HOLD_TO_NOT_LOCKED = 0.18
REENTRY_COOLDOWN_SEC = 0.85

PHONE_CONF_THRESHOLD = 0.35
PHONE_OVERRIDE_STICKY_SEC = 1.10

WINDOW_TITLE = "Locked In"

DRAW_FACE_DEBUG = True


def detect_phone(model, frame, conf_threshold):
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


def classify_state(pose_data, face_data):
    # Face negatives first
    if face_data["face_detected"]:
        if face_data["gaze_away_active"]:
            return "OFF_TASK"
        if face_data["talking_active"]:
            return "OFF_TASK"

    if not pose_data["pose_detected"]:
        return "OFF_TASK"

    norm_vert = pose_data["norm_vert"]
    norm_horiz = pose_data["norm_horiz"]

    # Focused on screen
    if norm_vert < -0.03 and abs(norm_horiz) < 0.16 and not face_data["gaze_away_active"]:
        return "FACING"

    # Writing: explicit
    if pose_data["writing_active"]:
        return "WRITING"

    # Reading
    if pose_data["reading_candidate"] and not face_data["gaze_away_active"] and not face_data["talking_active"]:
        return "READING"

    return "OFF_TASK"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    phone_model = YOLO("yolo11n.pt")
    face_extractor = FaceFeatureExtractor()
    pose_extractor = PoseFeatureExtractor()
    state_machine = AttentionStateMachine()

    fps_buf = deque(maxlen=30)
    last_time = time.time()

    last_phone_seen_time = 0.0
    last_phone_conf = 0.0
    last_phone_box = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        fps_buf.append(1.0 / max(now - last_time, 1e-6))
        last_time = now
        fps = sum(fps_buf) / len(fps_buf)

        # Phone
        phone_detected_now, phone_conf_now, phone_box_now = detect_phone(
            phone_model,
            frame,
            PHONE_CONF_THRESHOLD,
        )
        if phone_detected_now:
            last_phone_seen_time = now
            last_phone_conf = phone_conf_now
            last_phone_box = phone_box_now

        phone_active = (now - last_phone_seen_time) <= PHONE_OVERRIDE_STICKY_SEC

        # Features
        face_data = face_extractor.process(frame)
        pose_data = pose_extractor.process(frame)

        if DRAW_FACE_DEBUG:
            frame = face_extractor.draw_debug(frame, face_data)

        raw_state = classify_state(pose_data, face_data)
        status = state_machine.update(raw_state, now, hard_off=phone_active)
        internal_mode = "PHONE" if phone_active else raw_state

        reasons = []
        if phone_active:
            reasons.append(("Phone Detected", RED))
        else:
            if raw_state == "FACING":
                reasons.append(("Screen Focus", GREEN))
            elif raw_state == "READING":
                reasons.append(("Reading", BLUE))
            elif raw_state == "WRITING":
                reasons.append(("Writing Action", PURPLE))
            else:
                reasons.append(("Off Task", RED))

            if pose_data["writing_active"]:
                reasons.append(("One Arm Moving", PURPLE))

            if face_data["gaze_away_active"]:
                reasons.append(("Looking Away", RED))
            elif face_data["face_detected"] and face_data["gaze_horizontal"] == "center":
                reasons.append(("Eyes Centered", GREEN))

            if face_data["talking_active"]:
                reasons.append(("Talking", RED))

            if pose_data["reading_candidate"]:
                reasons.append(("Reading Posture", YELLOW))

        debug_lines = [
            f"raw: {raw_state:<8}  final: {status:<14}  mode: {internal_mode}",
            f"pose vert: {pose_data['norm_vert']:+.3f}  horiz: {pose_data['norm_horiz']:+.3f}  motion: {pose_data['smooth_motion']:.4f}",
            f"writing_votes: {pose_data['writing_votes']}  writing_active: {pose_data['writing_active']}  reading_candidate: {pose_data['reading_candidate']}",
            f"gaze: {face_data['gaze_horizontal']:<6}  gaze_away: {face_data['gaze_away_active']}  mouth: {face_data['mouth_open_ratio']:.3f}",
            f"talking_active: {face_data['talking_active']}  eye_open: {face_data['eye_open_ratio']:.3f}  phone_active: {phone_active}",
        ]

        frame = render_ui(
            frame=frame,
            status=status,
            internal_mode=internal_mode,
            vote_score=state_machine.vote_score,
            vote_cap=VOTE_CAP,
            fps=fps,
            phone_active=phone_active,
            phone_conf=last_phone_conf,
            phone_box=last_phone_box if phone_active else None,
            reasons=reasons,
            debug_lines=debug_lines,
        )

        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()