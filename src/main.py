"""
main.py — Locked In production entry point.

Controls:
  • Click buttons on screen (Start, Pause/Resume, End, New, Quit)
  • SPACE  — toggle pause
  • S      — start (from idle) / new session (from ended)
  • E      — end current session
  • Q      — quit
"""
import cv2
import time

from face_features     import FaceFeatureExtractor
from pose_features     import PoseFeatureExtractor
from attention_core    import (
    AttentionStateMachine, classify_state, build_reasons,
    detect_phone, create_phone_model,
    PHONE_CONF_THRESHOLD, PHONE_OVERRIDE_STICKY_SEC,
)
from ui_app  import render_app_ui
from session import SessionManager, IDLE, RUNNING, PAUSED, ENDED
from audio   import AudioCues, on_status_change


WINDOW_TITLE = "Locked In"

# ── mouse click state (module-level so callback can mutate) ───────────────────
_click_pending = {"pos": None}


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_pending["pos"] = (x, y)


def _hit_test(pos, buttons):
    if pos is None or not buttons:
        return None
    x, y = pos
    for b in buttons:
        bx1, by1, bx2, by2 = b["rect"]
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            return b["id"]
    return None


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
    session        = SessionManager()
    audio          = AudioCues(enabled=True)

    last_phone_seen_time = 0.0
    last_phone_box       = None
    prev_status          = "LOCKED OUT"
    end_report           = None

    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, _on_mouse)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now   = time.time()

        # Only run detection/classification while session is RUNNING
        reasons      = []
        phone_active = False
        status       = state_machine.status

        if session.state == RUNNING:
            # ── phone detection ───────────────────────────────────────────────
            phone_detected_now, _, phone_box_now = detect_phone(
                phone_model, frame, PHONE_CONF_THRESHOLD
            )
            if phone_detected_now:
                last_phone_seen_time = now
                last_phone_box       = phone_box_now

            phone_active = (now - last_phone_seen_time) <= PHONE_OVERRIDE_STICKY_SEC

            # ── features + classify ───────────────────────────────────────────
            face_data = face_extractor.process(frame)
            pose_data = pose_extractor.process(frame)
            raw_state = classify_state(pose_data, face_data)
            status    = state_machine.update(raw_state, now, hard_off=phone_active)
            reasons   = build_reasons(raw_state, pose_data, face_data, phone_active)

            # ── audio cue on status change ────────────────────────────────────
            on_status_change(audio, prev_status, status)
            prev_status = status

            # ── track distraction events for the session report ───────────────
            session.observe(status, reasons, phone_active)

        # ── render ────────────────────────────────────────────────────────────
        frame, buttons = render_app_ui(
            frame             = frame,
            status            = status,
            reasons           = reasons,
            attention_score   = state_machine.attention_score,
            session_elapsed   = session.elapsed,
            locked_in_seconds = state_machine.locked_in_seconds,
            current_streak    = state_machine.current_streak,
            longest_streak    = state_machine.longest_streak_sec,
            phone_active      = phone_active,
            phone_box         = last_phone_box if phone_active else None,
            session_state     = session.state,
            end_report        = end_report,
        )

        cv2.imshow(WINDOW_TITLE, frame)

        # ── mouse / keyboard input ────────────────────────────────────────────
        action = _hit_test(_click_pending["pos"], buttons)
        _click_pending["pos"] = None

        key = cv2.waitKey(1) & 0xFF
        if   key == ord("q"):
            action = action or "quit"
        elif key == ord("s"):
            if session.state == IDLE:   action = action or "start"
            elif session.state == ENDED: action = action or "new"
        elif key == ord("e"):
            if session.state in (RUNNING, PAUSED):
                action = action or "end"
        elif key == ord(" "):
            if session.state == RUNNING: action = action or "pause"
            elif session.state == PAUSED: action = action or "resume"

        # ── apply action ──────────────────────────────────────────────────────
        if action == "start":
            state_machine.reset()
            session.start()
            end_report           = None
            prev_status          = "LOCKED OUT"
            last_phone_seen_time = 0.0
        elif action == "pause":
            session.pause()
        elif action == "resume":
            session.resume()
        elif action == "end":
            # Use the state machine's accurate focused-time counter.
            end_report = session.report(
                focused_sec        = state_machine.locked_in_seconds,
                longest_streak_sec = state_machine.longest_streak_sec,
            )
            session.state = ENDED
            print("── SESSION COMPLETE ──")
            print(f"  score:        {end_report.score} ({end_report.grade})")
            print(f"  duration:     {end_report.duration_sec:.0f}s")
            print(f"  focused:      {end_report.focused_sec:.0f}s ({end_report.efficiency_pct:.0f}%)")
            print(f"  best streak:  {end_report.longest_streak_sec:.0f}s")
            print(f"  distractions: {end_report.distraction_count}")
            print(f"  breakdown:    {end_report.breakdown}")
        elif action == "new":
            session.reset()
            state_machine.reset()
            end_report = None
        elif action == "quit":
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()