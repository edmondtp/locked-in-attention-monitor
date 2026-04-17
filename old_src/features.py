import math
import cv2
import mediapipe as mp
from collections import deque

class TrackerFeatures:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7)
        
        self.gaze_hist = deque(maxlen=8)
        self.arm_motion_hist = deque(maxlen=15)
        self.prev_wrist_pos = None

    def get_metrics(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- Face Logic ---
        face_res = self.face_mesh.process(rgb)
        face_data = {"detected": False, "gaze_status": "center", "eyes_closed": False, "pitch": 0}
        
        if face_res.multi_face_landmarks:
            face_data["detected"] = True
            lm = face_res.multi_face_landmarks[0].landmark
            
            # Iris Tracking: Ratio of iris position between inner and outer eye corners
            # Left Eye: Outer(33), Inner(133), Iris(468)
            l_iris = lm[468].x
            l_ratio = (l_iris - lm[33].x) / (lm[133].x - lm[33].x)
            self.gaze_hist.append(l_ratio)
            avg_gaze = sum(self.gaze_hist) / len(self.gaze_hist)

            if avg_gaze < 0.35: face_data["gaze_status"] = "left"
            elif avg_gaze > 0.65: face_data["gaze_status"] = "right"
            
            # Pitch (Looking down)
            face_data["pitch"] = lm[1].y - (lm[33].y + lm[263].y)/2

        # --- Pose / Writing Logic ---
        pose_res = self.pose.process(rgb)
        writing_score = 0
        if pose_res.pose_landmarks:
            wrist = pose_res.pose_landmarks.landmark[15] # Left or Right wrist
            if self.prev_wrist_pos:
                dist = math.hypot(wrist.x - self.prev_wrist_pos[0], wrist.y - self.prev_wrist_pos[1])
                self.arm_motion_hist.append(dist)
                writing_score = sum(self.arm_motion_hist) / len(self.arm_motion_hist)
            self.prev_wrist_pos = (wrist.x, wrist.y)

        return face_data, writing_score