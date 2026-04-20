"""
face_features.py — MediaPipe FaceLandmarker feature extraction.
Compatible with mediapipe >= 0.10.30 (Tasks API, no mp.solutions).
Model file is auto-downloaded on first run.
"""
from __future__ import annotations

import cv2
import math
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils import EMA, VotingWindow

# ── model download ────────────────────────────────────────────────────────────
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"[face_features] Downloading face landmarker model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[face_features] Download complete.")


# ── landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE_OUTER  = 33
LEFT_EYE_INNER  = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_IRIS       = 468
RIGHT_IRIS      = 473
LEFT_UPPER_LID  = 159
LEFT_LOWER_LID  = 145
RIGHT_UPPER_LID = 386
RIGHT_LOWER_LID = 374
MOUTH_LEFT      = 61
MOUTH_RIGHT     = 291
UPPER_LIP       = 13
LOWER_LIP       = 14
NOSE_TIP        = 1
CHIN            = 152


class FaceFeatureExtractor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float  = 0.5,
    ):
        _ensure_model()

        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options                          = base_opts,
            running_mode                          = mp_vision.RunningMode.VIDEO,
            num_faces                             = 1,
            min_face_detection_confidence         = min_detection_confidence,
            min_face_presence_confidence          = min_detection_confidence,
            min_tracking_confidence               = min_tracking_confidence,
            output_face_blendshapes               = False,
            output_facial_transformation_matrixes = False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._frame_ts  = 0

        self.gaze_ema      = EMA(0.30)
        self.gaze_v_ema    = EMA(0.30)
        self.yaw_ema       = EMA(0.25)
        self.pitch_ema     = EMA(0.25)
        self.mouth_ema     = EMA(0.20)
        self.eye_open_ema  = EMA(0.20)

        # Windows — "balanced" policy: require ~0.5s of sustained away
        # before flagging. At ~20 fps, 10 frames @ 70% threshold ≈ 0.5s.
        self.talking_window    = VotingWindow(maxlen=24, threshold_frac=0.65)
        self.gaze_left_window  = VotingWindow(maxlen=14, threshold_frac=0.70)
        self.gaze_right_window = VotingWindow(maxlen=14, threshold_frac=0.70)
        self.gaze_up_window    = VotingWindow(maxlen=14, threshold_frac=0.70)
        self.gaze_down_window  = VotingWindow(maxlen=14, threshold_frac=0.70)
        self.pitch_away_window = VotingWindow(maxlen=14, threshold_frac=0.70)

    @staticmethod
    def _dist(p1, p2) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def process(self, frame_bgr) -> dict:
        output = {
            "face_detected":    False,
            "gaze_score":       0.5,
            "gaze_horizontal":  "unknown",
            "gaze_away_active": False,
            "mouth_open_ratio": 0.0,
            "talking_active":   False,
            "face_yaw_ratio":   0.0,
            "face_pitch_ratio": 0.0,
            "eye_open_ratio":   0.0,
            "eyes_closed":      False,
            "landmarks_px":     None,
        }

        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.face_landmarks:
            self.talking_window.push(False)
            self.gaze_left_window.push(False)
            self.gaze_right_window.push(False)
            self.gaze_up_window.push(False)
            self.gaze_down_window.push(False)
            self.pitch_away_window.push(False)
            return output

        lms = result.face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        has_iris = len(pts) > max(LEFT_IRIS, RIGHT_IRIS)

        output["face_detected"] = True
        output["landmarks_px"]  = pts

        lo = pts[LEFT_EYE_OUTER];  li = pts[LEFT_EYE_INNER]
        ri = pts[RIGHT_EYE_INNER]; ro = pts[RIGHT_EYE_OUTER]
        lu = pts[LEFT_UPPER_LID];  ld = pts[LEFT_LOWER_LID]
        ru = pts[RIGHT_UPPER_LID]; rd = pts[RIGHT_LOWER_LID]

        lw = max(1.0, self._dist(lo, li))
        rw = max(1.0, self._dist(ri, ro))
        lh = max(1.0, self._dist(lu, ld))
        rh = max(1.0, self._dist(ru, rd))

        # ── horizontal gaze (iris X relative to eye corners) ────────────────
        if has_iris:
            li_pt = pts[LEFT_IRIS]; ri_pt = pts[RIGHT_IRIS]
            left_ratio_x  = (li_pt[0] - lo[0]) / lw
            right_ratio_x = (ri_pt[0] - ri[0]) / rw
            gaze_score_x  = self.gaze_ema.update((left_ratio_x + right_ratio_x) / 2.0)

            # ── vertical gaze (iris Y relative to eyelids) ──────────────────
            # 0.0 = iris at upper lid, 1.0 = iris at lower lid. ~0.5 = neutral.
            left_eye_mid_y  = (lu[1] + ld[1]) / 2.0
            right_eye_mid_y = (ru[1] + rd[1]) / 2.0
            left_ratio_y    = (li_pt[1] - lu[1]) / lh
            right_ratio_y   = (ri_pt[1] - ru[1]) / rh
            gaze_score_y    = self.gaze_v_ema.update((left_ratio_y + right_ratio_y) / 2.0)
        else:
            gaze_score_x = self.gaze_ema.update(0.5)
            gaze_score_y = self.gaze_v_ema.update(0.5)

        output["gaze_score"]   = gaze_score_x
        output["gaze_score_y"] = gaze_score_y

        # Horizontal band
        gaze_h = "left" if gaze_score_x < 0.36 else ("right" if gaze_score_x > 0.64 else "center")
        output["gaze_horizontal"] = gaze_h

        # Vertical band — tuned so "looking at screen while sitting normally"
        # stays in "center". Iris sits slightly high when reading, so neutral
        # band is 0.28–0.68.
        if gaze_score_y < 0.28:
            gaze_v = "up"
        elif gaze_score_y > 0.68:
            gaze_v = "down"
        else:
            gaze_v = "center"
        output["gaze_vertical"] = gaze_v

        ml, mr = pts[MOUTH_LEFT], pts[MOUTH_RIGHT]
        ul_pt, ll_pt = pts[UPPER_LIP],  pts[LOWER_LIP]
        mouth_w          = max(1.0, self._dist(ml, mr))
        mouth_open_ratio = self.mouth_ema.update(self._dist(ul_pt, ll_pt) / mouth_w)
        output["mouth_open_ratio"] = mouth_open_ratio
        output["talking_active"]   = self.talking_window.push(mouth_open_ratio > 0.12)

        # Eye openness (reuse lh/rh we already computed)
        eye_open_ratio          = self.eye_open_ema.update(((lh / lw) + (rh / rw)) / 2.0)
        output["eye_open_ratio"] = eye_open_ratio
        output["eyes_closed"]    = eye_open_ratio < 0.10

        nose   = pts[NOSE_TIP]; chin = pts[CHIN]
        face_w = max(1.0, self._dist(pts[LEFT_EYE_OUTER], pts[RIGHT_EYE_OUTER]))
        eye_cx = (pts[LEFT_EYE_OUTER][0] + pts[RIGHT_EYE_OUTER][0]) / 2.0
        eye_cy = (pts[LEFT_EYE_OUTER][1] + pts[RIGHT_EYE_OUTER][1]) / 2.0

        yaw_ratio   = self.yaw_ema.update((nose[0] - eye_cx) / face_w)
        pitch_ratio = self.pitch_ema.update((nose[1] - eye_cy) / max(1.0, self._dist(nose, chin)))
        output["face_yaw_ratio"]   = yaw_ratio
        output["face_pitch_ratio"] = pitch_ratio

        # ── gaze-away decision (balanced) ────────────────────────────────────
        # Flag if ANY of these sustain for ~0.5s:
        #   • horizontal gaze pulled left/right
        #   • vertical gaze pulled up (looking above monitor) or down
        #     (looking past monitor — but allow reading/writing posture)
        #   • head pitched up significantly (chin-up, looking up)
        #   • head yawed hard left/right
        #
        # "Looking down" alone is NOT flagged because that's reading/writing.
        # But "head pitched up" IS flagged — that's clearly looking away.
        horiz_away = (
            self.gaze_left_window.push(gaze_h == "left")
            or self.gaze_right_window.push(gaze_h == "right")
        )
        vert_up_away = self.gaze_up_window.push(gaze_v == "up")
        # Head pitched up (nose above eye line) — clear look-away signal
        pitch_up_away = self.pitch_away_window.push(pitch_ratio < -0.18)

        yaw_away = abs(yaw_ratio) > 0.18
        output["gaze_away_active"] = horiz_away or vert_up_away or pitch_up_away or yaw_away

        return output

    def draw_debug(self, frame, face_data):
        if not face_data["face_detected"] or face_data["landmarks_px"] is None:
            return frame
        pts = face_data["landmarks_px"]
        for idx in [LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
                    LEFT_UPPER_LID, LEFT_LOWER_LID, RIGHT_UPPER_LID, RIGHT_LOWER_LID,
                    MOUTH_LEFT, MOUTH_RIGHT, UPPER_LIP, LOWER_LIP, NOSE_TIP, CHIN]:
            if idx < len(pts):
                cv2.circle(frame, pts[idx], 2, (0, 255, 200), -1)
        return frame