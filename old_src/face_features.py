# src/face_features.py
import cv2
import mediapipe as mp
import math
from collections import deque


class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class FaceFeatureExtractor:
    def __init__(
        self,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Eye / iris
        self.LEFT_EYE_OUTER = 33
        self.LEFT_EYE_INNER = 133
        self.RIGHT_EYE_INNER = 362
        self.RIGHT_EYE_OUTER = 263
        self.LEFT_IRIS = 468
        self.RIGHT_IRIS = 473

        # Eyelids
        self.LEFT_UPPER_LID = 159
        self.LEFT_LOWER_LID = 145
        self.RIGHT_UPPER_LID = 386
        self.RIGHT_LOWER_LID = 374

        # Mouth
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14

        # Orientation
        self.NOSE_TIP = 1
        self.CHIN = 152

        self.gaze_ema = EMA(0.30)
        self.yaw_ema = EMA(0.25)
        self.pitch_ema = EMA(0.25)
        self.mouth_ema = EMA(0.20)
        self.eye_open_ema = EMA(0.20)

        self.talking_hist = deque(maxlen=24)
        self.gaze_left_hist = deque(maxlen=10)
        self.gaze_right_hist = deque(maxlen=10)

    def _distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        output = {
            "face_detected": False,
            "gaze_score": 0.5,
            "gaze_horizontal": "unknown",
            "gaze_away_active": False,
            "mouth_open_ratio": 0.0,
            "talking_active": False,
            "face_yaw_ratio": 0.0,
            "face_pitch_ratio": 0.0,
            "eye_open_ratio": 0.0,
            "eyes_closed": False,
            "landmarks_px": None,
        }

        if not results.multi_face_landmarks:
            self.talking_hist.append(0)
            self.gaze_left_hist.append(0)
            self.gaze_right_hist.append(0)
            return output

        h, w = frame_bgr.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

        output["face_detected"] = True
        output["landmarks_px"] = pts

        # Eyes / gaze
        left_eye_outer = pts[self.LEFT_EYE_OUTER]
        left_eye_inner = pts[self.LEFT_EYE_INNER]
        left_iris = pts[self.LEFT_IRIS]

        right_eye_inner = pts[self.RIGHT_EYE_INNER]
        right_eye_outer = pts[self.RIGHT_EYE_OUTER]
        right_iris = pts[self.RIGHT_IRIS]

        left_eye_width = max(1.0, self._distance(left_eye_outer, left_eye_inner))
        right_eye_width = max(1.0, self._distance(right_eye_inner, right_eye_outer))

        left_ratio = (left_iris[0] - left_eye_outer[0]) / left_eye_width
        right_ratio = (right_iris[0] - right_eye_inner[0]) / right_eye_width
        gaze_score = (left_ratio + right_ratio) / 2.0
        gaze_score = self.gaze_ema.update(gaze_score)

        output["gaze_score"] = gaze_score

        if gaze_score < 0.36:
            gaze_horizontal = "left"
        elif gaze_score > 0.64:
            gaze_horizontal = "right"
        else:
            gaze_horizontal = "center"

        output["gaze_horizontal"] = gaze_horizontal

        # Mouth
        mouth_left = pts[self.MOUTH_LEFT]
        mouth_right = pts[self.MOUTH_RIGHT]
        upper_lip = pts[self.UPPER_LIP]
        lower_lip = pts[self.LOWER_LIP]

        mouth_width = max(1.0, self._distance(mouth_left, mouth_right))
        mouth_open = self._distance(upper_lip, lower_lip)
        mouth_open_ratio = self.mouth_ema.update(mouth_open / mouth_width)
        output["mouth_open_ratio"] = mouth_open_ratio

        talking_now = mouth_open_ratio > 0.12
        self.talking_hist.append(1 if talking_now else 0)
        output["talking_active"] = sum(self.talking_hist) >= int(len(self.talking_hist) * 0.65)

        # Eye openness
        left_upper = pts[self.LEFT_UPPER_LID]
        left_lower = pts[self.LEFT_LOWER_LID]
        right_upper = pts[self.RIGHT_UPPER_LID]
        right_lower = pts[self.RIGHT_LOWER_LID]

        left_eye_open = self._distance(left_upper, left_lower) / left_eye_width
        right_eye_open = self._distance(right_upper, right_lower) / right_eye_width
        eye_open_ratio = self.eye_open_ema.update((left_eye_open + right_eye_open) / 2.0)
        output["eye_open_ratio"] = eye_open_ratio
        output["eyes_closed"] = eye_open_ratio < 0.10

        # Orientation
        nose_tip = pts[self.NOSE_TIP]
        chin = pts[self.CHIN]

        face_width = max(1.0, self._distance(pts[self.LEFT_EYE_OUTER], pts[self.RIGHT_EYE_OUTER]))
        face_height = max(1.0, self._distance(nose_tip, chin))

        eye_center_x = (pts[self.LEFT_EYE_OUTER][0] + pts[self.RIGHT_EYE_OUTER][0]) / 2.0
        eye_center_y = (pts[self.LEFT_EYE_OUTER][1] + pts[self.RIGHT_EYE_OUTER][1]) / 2.0

        yaw_ratio = self.yaw_ema.update((nose_tip[0] - eye_center_x) / face_width)
        pitch_ratio = self.pitch_ema.update((nose_tip[1] - eye_center_y) / face_height)

        output["face_yaw_ratio"] = yaw_ratio
        output["face_pitch_ratio"] = pitch_ratio

        # Smoothed gaze-away
        self.gaze_left_hist.append(1 if gaze_horizontal == "left" else 0)
        self.gaze_right_hist.append(1 if gaze_horizontal == "right" else 0)
        output["gaze_away_active"] = (
            sum(self.gaze_left_hist) >= int(len(self.gaze_left_hist) * 0.7)
            or sum(self.gaze_right_hist) >= int(len(self.gaze_right_hist) * 0.7)
            or abs(yaw_ratio) > 0.13
        )

        return output

    def draw_debug(self, frame, face_data):
        if not face_data["face_detected"] or face_data["landmarks_px"] is None:
            return frame

        pts = face_data["landmarks_px"]

        key_indices = [
            self.LEFT_EYE_OUTER, self.LEFT_EYE_INNER,
            self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER,
            self.LEFT_IRIS, self.RIGHT_IRIS,
            self.LEFT_UPPER_LID, self.LEFT_LOWER_LID,
            self.RIGHT_UPPER_LID, self.RIGHT_LOWER_LID,
            self.MOUTH_LEFT, self.MOUTH_RIGHT,
            self.UPPER_LIP, self.LOWER_LIP,
            self.NOSE_TIP, self.CHIN,
        ]

        for idx in key_indices:
            cv2.circle(frame, pts[idx], 2, (0, 255, 255), -1)

        return frame