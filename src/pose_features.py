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


class RollingMean:
    def __init__(self, n):
        self.buf = deque(maxlen=n)

    def update(self, x):
        self.buf.append(x)
        return sum(self.buf) / len(self.buf) if self.buf else 0.0


class PoseFeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
        )

        self.vert_ema = EMA(0.22)
        self.horiz_ema = EMA(0.22)
        self.motion_roller = RollingMean(10)

        self.prev_left_sh = None
        self.prev_right_sh = None
        self.prev_left_wrist = None
        self.prev_right_wrist = None

        self.writing_hist = deque(maxlen=18)

    def _distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _landmark_px(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    def process(self, frame_bgr):
        small = cv2.resize(frame_bgr, (640, 480))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        output = {
            "pose_detected": False,
            "raw_results": res,
            "norm_vert": 0.0,
            "norm_horiz": 0.0,
            "smooth_motion": 0.0,
            "writing_votes": 0,
            "writing_active": False,
            "reading_candidate": False,
        }

        if not res.pose_landmarks:
            self.prev_left_sh = None
            self.prev_right_sh = None
            self.prev_left_wrist = None
            self.prev_right_wrist = None
            self.writing_hist.append(0)
            return output

        output["pose_detected"] = True

        lm = res.pose_landmarks.landmark
        pw, ph = 640, 480

        pose_enum = self.mp_pose.PoseLandmark

        nose = self._landmark_px(lm, pose_enum.NOSE.value, pw, ph)
        left_sh = self._landmark_px(lm, pose_enum.LEFT_SHOULDER.value, pw, ph)
        right_sh = self._landmark_px(lm, pose_enum.RIGHT_SHOULDER.value, pw, ph)
        left_wrist = self._landmark_px(lm, pose_enum.LEFT_WRIST.value, pw, ph)
        right_wrist = self._landmark_px(lm, pose_enum.RIGHT_WRIST.value, pw, ph)

        sh_mid = ((left_sh[0] + right_sh[0]) // 2, (left_sh[1] + right_sh[1]) // 2)
        sh_w = max(self._distance(left_sh, right_sh), 1)

        norm_vert = self.vert_ema.update((nose[1] - sh_mid[1]) / sh_w)
        norm_horiz = self.horiz_ema.update((nose[0] - sh_mid[0]) / sh_w)

        if self.prev_left_sh is not None and self.prev_right_sh is not None:
            raw_motion = (
                self._distance(left_sh, self.prev_left_sh) +
                self._distance(right_sh, self.prev_right_sh)
            ) / (2 * sh_w)
        else:
            raw_motion = 0.0

        smooth_motion = self.motion_roller.update(raw_motion)

        # Wrist motion for writing
        if self.prev_left_wrist is not None and self.prev_right_wrist is not None:
            left_wrist_speed = self._distance(left_wrist, self.prev_left_wrist) / sh_w
            right_wrist_speed = self._distance(right_wrist, self.prev_right_wrist) / sh_w
        else:
            left_wrist_speed = 0.0
            right_wrist_speed = 0.0

        self.prev_left_sh = left_sh
        self.prev_right_sh = right_sh
        self.prev_left_wrist = left_wrist
        self.prev_right_wrist = right_wrist

        looking_down = 0.04 <= norm_vert <= 0.24

        left_wrist_below_shoulder = left_wrist[1] > left_sh[1] - 10
        right_wrist_below_shoulder = right_wrist[1] > right_sh[1] - 10

        one_arm_writing_motion = (
            (0.01 <= left_wrist_speed <= 0.16 and left_wrist_below_shoulder) or
            (0.01 <= right_wrist_speed <= 0.16 and right_wrist_below_shoulder)
        )

        writing_now = looking_down and one_arm_writing_motion
        self.writing_hist.append(1 if writing_now else 0)

        writing_votes = sum(self.writing_hist)
        writing_active = writing_votes >= 7

        reading_candidate = (-0.01 <= norm_vert <= 0.12) and abs(norm_horiz) < 0.16

        output["norm_vert"] = norm_vert
        output["norm_horiz"] = norm_horiz
        output["smooth_motion"] = smooth_motion
        output["writing_votes"] = writing_votes
        output["writing_active"] = writing_active
        output["reading_candidate"] = reading_candidate

        return output