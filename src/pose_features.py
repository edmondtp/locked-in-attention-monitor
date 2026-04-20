"""
pose_features.py — MediaPipe PoseLandmarker feature extraction.
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

from utils import EMA, RollingMean, VotingWindow

# ── model download ────────────────────────────────────────────────────────────
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"[pose_features] Downloading pose landmarker model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[pose_features] Download complete.")


# ── landmark indices ──────────────────────────────────────────────────────────
# From mediapipe.tasks.python.vision.PoseLandmark enum
NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_WRIST     = 15
RIGHT_WRIST    = 16


class PoseFeatureExtractor:
    def __init__(self):
        _ensure_model()

        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options              = base_opts,
            running_mode              = mp_vision.RunningMode.VIDEO,
            num_poses                 = 1,
            min_pose_detection_confidence = 0.5,
            min_pose_presence_confidence  = 0.5,
            min_tracking_confidence       = 0.5,
            output_segmentation_masks     = False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        self._frame_ts  = 0

        self.vert_ema      = EMA(0.22)
        self.horiz_ema     = EMA(0.22)
        self.motion_roller = RollingMean(10)
        self.writing_votes = VotingWindow(maxlen=18, threshold_frac=7 / 18)

        self.prev_left_sh     = None
        self.prev_right_sh    = None
        self.prev_left_wrist  = None
        self.prev_right_wrist = None

    @staticmethod
    def _dist(p1, p2) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _px(self, lms, idx: int, w: int, h: int) -> tuple[int, int]:
        lm = lms[idx]
        return int(lm.x * w), int(lm.y * h)

    def process(self, frame_bgr) -> dict:
        output = {
            "pose_detected":     False,
            "raw_results":       None,
            "norm_vert":         0.0,
            "norm_horiz":        0.0,
            "smooth_motion":     0.0,
            "writing_votes":     0,
            "writing_active":    False,
            "reading_candidate": False,
        }

        # Resize for speed (same as before)
        small = cv2.resize(frame_bgr, (640, 480))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts += 33
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)
        output["raw_results"] = result

        if not result.pose_landmarks:
            self.prev_left_sh     = None
            self.prev_right_sh    = None
            self.prev_left_wrist  = None
            self.prev_right_wrist = None
            self.writing_votes.push(False)
            return output

        output["pose_detected"] = True
        lms = result.pose_landmarks[0]
        pw, ph = 640, 480

        nose        = self._px(lms, NOSE,           pw, ph)
        left_sh     = self._px(lms, LEFT_SHOULDER,  pw, ph)
        right_sh    = self._px(lms, RIGHT_SHOULDER, pw, ph)
        left_wrist  = self._px(lms, LEFT_WRIST,     pw, ph)
        right_wrist = self._px(lms, RIGHT_WRIST,    pw, ph)

        sh_mid = ((left_sh[0] + right_sh[0]) // 2, (left_sh[1] + right_sh[1]) // 2)
        sh_w   = max(self._dist(left_sh, right_sh), 1.0)

        norm_vert  = self.vert_ema.update((nose[1] - sh_mid[1]) / sh_w)
        norm_horiz = self.horiz_ema.update((nose[0] - sh_mid[0]) / sh_w)

        if self.prev_left_sh is not None and self.prev_right_sh is not None:
            raw_motion = (
                self._dist(left_sh,  self.prev_left_sh) +
                self._dist(right_sh, self.prev_right_sh)
            ) / (2 * sh_w)
        else:
            raw_motion = 0.0
        smooth_motion = self.motion_roller.update(raw_motion)

        if self.prev_left_wrist is not None and self.prev_right_wrist is not None:
            lws = self._dist(left_wrist,  self.prev_left_wrist)  / sh_w
            rws = self._dist(right_wrist, self.prev_right_wrist) / sh_w
        else:
            lws = rws = 0.0

        self.prev_left_sh     = left_sh
        self.prev_right_sh    = right_sh
        self.prev_left_wrist  = left_wrist
        self.prev_right_wrist = right_wrist

        looking_down = 0.04 <= norm_vert <= 0.24
        lw_below_sh  = left_wrist[1]  > left_sh[1]  - 10
        rw_below_sh  = right_wrist[1] > right_sh[1] - 10
        one_arm_writing = (
            (0.01 <= lws <= 0.16 and lw_below_sh) or
            (0.01 <= rws <= 0.16 and rw_below_sh)
        )

        writing_active    = self.writing_votes.push(looking_down and one_arm_writing)
        reading_candidate = (-0.01 <= norm_vert <= 0.12) and abs(norm_horiz) < 0.16

        output.update(
            norm_vert         = norm_vert,
            norm_horiz        = norm_horiz,
            smooth_motion     = smooth_motion,
            writing_votes     = self.writing_votes.votes,
            writing_active    = writing_active,
            reading_candidate = reading_candidate,
        )
        return output