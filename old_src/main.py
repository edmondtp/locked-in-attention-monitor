import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from face_features import FaceFeatureExtractor
from ui_engine import draw_hud, COLORS
from logic_helpers import AttentionStateMachine, PostureEMA

def main():
    cap = cv2.VideoCapture(0)
    yolo = YOLO("yolo11n.pt")
    face_ex = FaceFeatureExtractor()
    state_m = AttentionStateMachine()
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.6)
    
    # EMAs for smoothing
    yaw_ema = PostureEMA(0.25)
    pitch_ema = PostureEMA(0.25)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        now = time.time()

        # 1. Feature Extraction
        face = face_ex.process(frame)
        pose_res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 2. YOLO Phone Check
        yolo_res = yolo.predict(frame, conf=0.4, verbose=False)
        phone_active = any(yolo.names[int(box.cls)] == 'cell phone' for r in yolo_res for box in r.boxes)

        # 3. Classify Raw State
        raw_state = "OFF_TASK"
        reasons = []
        
        if face["face_detected"]:
            # Smooth the face orientation
            yaw = yaw_ema.update(face["yaw"])
            pitch = pitch_ema.update(face["pitch"])
            
            # --- WRITING LOGIC ---
            # Head down (pitch) + potential for looking at desk
            if pitch < -0.4: # Adjust threshold based on laptop height
                raw_state = "WRITING"
                reasons.append(("WRITING MODE", COLORS["NEON_BLUE"]))
            
            # --- FACING LOGIC ---
            elif abs(yaw) < 0.2:
                if face["gaze_away"]:
                    raw_state = "OFF_TASK"
                    reasons.append(("GAZE AWAY", COLORS["NEON_RED"]))
                elif face["eyes_closed"]:
                    raw_state = "OFF_TASK"
                    reasons.append(("EYES CLOSED", COLORS["NEON_RED"]))
                else:
                    raw_state = "FACING"
                    reasons.append(("SCREEN FOCUS", COLORS["NEON_GREEN"]))
            else:
                reasons.append(("TURNED AWAY", COLORS["NEON_RED"]))
        else:
            reasons.append(("NO FACE", COLORS["NEON_RED"]))

        if phone_active:
            reasons.append(("PHONE DETECTED", COLORS["NEON_RED"]))

        # 4. State Update
        status = state_m.update(raw_state, now, hard_off=phone_active)
        
        # 5. UI Rendering
        debug_lines = [
            f"Mode: {raw_state}",
            f"Gaze Score: {face['gaze_score']:.2f}",
            f"Face Yaw: {face['yaw']:.2f}",
            f"Face Pitch: {face['pitch']:.2f}",
            f"Vote Score: {state_m.vote_score}"
        ]
        
        out_frame = draw_hud(frame, status, raw_state, state_m.vote_score, phone_active, reasons, debug_lines)
        
        cv2.imshow("Locked In HUD", out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()