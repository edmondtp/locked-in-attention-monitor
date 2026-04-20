"""
main.py — Locked In production entry point.
"""
import cv2
import time
from collections import deque

from face_features import FaceFeatureExtractor
from pose_features import PoseFeatureExtractor
from attention_core import (
    AttentionStateMachine,
    classify_state,
    build_reasons,
    detect_phone,
    create_phone_model,
    PHONE_CONF_THRESHOLD,
    PHONE_OVERRIDE_STICKY_SEC,
)
from ui_app import render_app_ui

WINDOW_TITLE    = "Locked In"
DRAW_FACE_DEBUG = False


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    phone_model    = create_phone_model()
    face_extractor = FaceFeatureExtractor()
    pose_extractor = PoseFeatureExtractor()
    state_machine  = AttentionStateMachine()

    last_phone_seen_time = 0.0
    last_phone_box       = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now   = time.time()

        # ── phone detection ───────────────────────────────────────────────────
        phone_detected_now, _, phone_box_now = detect_phone(
            phone_model, frame, PHONE_CONF_THRESHOLD
        )
        if phone_detected_now:
            last_phone_seen_time = now
            last_phone_box       = phone_box_now

        phone_active = (now - last_phone_seen_time) <= PHONE_OVERRIDE_STICKY_SEC

        # ── feature extraction ────────────────────────────────────────────────
        face_data = face_extractor.process(frame)
        pose_data = pose_extractor.process(frame)

        if DRAW_FACE_DEBUG:
            frame = face_extractor.draw_debug(frame, face_data)

        # ── classification ────────────────────────────────────────────────────
        raw_state = classify_state(pose_data, face_data)
        status    = state_machine.update(raw_state, now, hard_off=phone_active)
        reasons   = build_reasons(raw_state, pose_data, face_data, phone_active)

        # ── render ────────────────────────────────────────────────────────────
        frame = render_app_ui(
            frame            = frame,
            status           = status,
            reasons          = reasons,
            attention_score  = state_machine.attention_score,
            session_elapsed  = state_machine.session_elapsed,
            locked_in_seconds = state_machine.locked_in_seconds,
            current_streak   = state_machine.current_streak,
            longest_streak   = state_machine.longest_streak_sec,
            phone_active     = phone_active,
            phone_box        = last_phone_box if phone_active else None,
        )

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()