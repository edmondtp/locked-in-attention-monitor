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
    VOTE_CAP,
)
from ui_debug import render_debug_ui

WINDOW_TITLE = "Locked In Debug"
DRAW_FACE_DEBUG = True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    phone_model = create_phone_model()
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
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        fps_buf.append(1.0 / max(now - last_time, 1e-6))
        last_time = now
        fps = sum(fps_buf) / len(fps_buf)

        phone_detected_now, phone_conf_now, phone_box_now = detect_phone(
            phone_model, frame, PHONE_CONF_THRESHOLD
        )

        if phone_detected_now:
            last_phone_seen_time = now
            last_phone_conf = phone_conf_now
            last_phone_box = phone_box_now

        phone_active = (now - last_phone_seen_time) <= PHONE_OVERRIDE_STICKY_SEC

        face_data = face_extractor.process(frame)
        pose_data = pose_extractor.process(frame)

        if DRAW_FACE_DEBUG:
            frame = face_extractor.draw_debug(frame, face_data)

        raw_state = classify_state(pose_data, face_data)
        status = state_machine.update(raw_state, now, hard_off=phone_active)
        internal_mode = "PHONE" if phone_active else raw_state

        reasons = build_reasons(raw_state, pose_data, face_data, phone_active)

        debug_lines = [
            f"pose vert: {pose_data['norm_vert']:+.3f}  horiz: {pose_data['norm_horiz']:+.3f}",
            f"motion: {pose_data['smooth_motion']:.4f}  writing_votes: {pose_data['writing_votes']}",
            f"writing_active: {pose_data['writing_active']}  reading_candidate: {pose_data['reading_candidate']}",
            f"gaze: {face_data['gaze_horizontal']}  gaze_away: {face_data['gaze_away_active']}",
            f"mouth: {face_data['mouth_open_ratio']:.3f}  talking: {face_data['talking_active']}",
            f"eye_open: {face_data['eye_open_ratio']:.3f}  phone_active: {phone_active}",
            f"vote_score: {state_machine.vote_score:+d}/{VOTE_CAP}  fps: {fps:.0f}",
        ]

        frame = render_debug_ui(
            frame=frame,
            status=status,
            internal_mode=internal_mode,
            reasons=reasons,
            debug_lines=debug_lines,
            phone_active=phone_active,
            phone_box=last_phone_box if phone_active else None,
        )

        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()