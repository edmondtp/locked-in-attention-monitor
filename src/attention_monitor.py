import cv2
import mediapipe as mp
import math
import time
from collections import deque
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# =============================================================================
# LOCKED IN
# Final sleek UI version
# =============================================================================

# ---------------------------------
# CONFIG
# ---------------------------------

WINDOW_TITLE = "Locked In"

# Camera / processing
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# Calibration
CALIBRATION_SECONDS = 3.0
CALIBRATION_SLACK = 0.07
FACING_VERT_FALLBACK = -0.10

# Pose interpretation
CENTERED_HORIZ_MAX = 0.16
STRONG_TURN_HORIZ = 0.24

READING_DOWN_MIN = -0.01
READING_DOWN_MAX = 0.12

WRITING_DOWN_MIN = 0.04
WRITING_DOWN_MAX = 0.24

WRITING_MOTION_LOW = 0.004
WRITING_MOTION_HIGH = 0.055
READING_MOTION_MAX = 0.014

WRITING_VOTE_WINDOW = 20
WRITING_VOTE_MIN_HITS = 7

# Smoothing
VERT_EMA_ALPHA = 0.22
HORIZ_EMA_ALPHA = 0.22
MOTION_SMOOTH_WINDOW = 10

# State machine
VOTE_CAP = 16
LOCK_ENTER_THRESHOLD = 11
LOCK_EXIT_THRESHOLD = 3

HOLD_TO_LOCKED_IN = 0.85
HOLD_TO_NOT_LOCKED = 0.18

REENTRY_COOLDOWN_SEC = 0.85

# Phone override
PHONE_CONF_THRESHOLD = 0.35
PHONE_OVERRIDE_STICKY_SEC = 1.10

# UI
SIDE_PANEL_WIDTH = 300
BOTTOM_PANEL_HEIGHT = 132

STATUS_FONT_SIZE = 52
TITLE_FONT_SIZE = 24
SMALL_FONT_SIZE = 20
DEBUG_FONT_SIZE = 17
CHIP_FONT_SIZE = 18
ALERT_FONT_SIZE = 260

# Colors (RGBA for PIL overlay)
WHITE = (245, 247, 250, 255)
MUTED = (185, 192, 203, 255)

GREEN = (39, 220, 124, 255)
GREEN_SOFT = (39, 220, 124, 82)
GREEN_DARK = (28, 168, 96, 255)

RED = (255, 71, 87, 255)
RED_SOFT = (255, 71, 87, 88)
RED_DARK = (214, 57, 74, 255)

YELLOW = (255, 193, 7, 255)
BLUE = (86, 156, 214, 255)
PURPLE = (163, 112, 255, 255)
CYAN = (59, 224, 224, 255)

BG_PANEL = (16, 18, 24, 220)
BG_PANEL_SOFT = (20, 23, 30, 198)
BG_CHIP_DARK = (28, 32, 42, 228)
BORDER = (255, 255, 255, 34)

# ---------------------------------
# MEDIAPIPE
# ---------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------
# HELPERS
# ---------------------------------


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def landmark_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


class EMA:
    def __init__(self, alpha):
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


def get_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_STATUS = get_font(STATUS_FONT_SIZE)
FONT_TITLE = get_font(TITLE_FONT_SIZE)
FONT_SMALL = get_font(SMALL_FONT_SIZE)
FONT_DEBUG = get_font(DEBUG_FONT_SIZE)
FONT_CHIP = get_font(CHIP_FONT_SIZE)
FONT_ALERT = get_font(ALERT_FONT_SIZE)


def hex_to_rgba(hex_color, alpha=255):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def draw_glow_text(draw, pos, text, font, fill, glow_fill, glow_radius=2):
    x, y = pos
    for dx in range(-glow_radius, glow_radius + 1):
        for dy in range(-glow_radius, glow_radius + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=glow_fill)
    draw.text((x, y), text, font=font, fill=fill)


def draw_rounded(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_chip(draw, x, y, text, fill, text_fill=WHITE, outline=None):
    bbox = draw.textbbox((0, 0), text, font=FONT_CHIP)
    w = bbox[2] - bbox[0] + 22
    h = bbox[3] - bbox[1] + 12
    draw_rounded(draw, (x, y, x + w, y + h), 16, fill, outline=outline, width=1 if outline else 0)
    draw.text((x + 11, y + 5), text, font=FONT_CHIP, fill=text_fill)
    return w, h


def draw_gradient_bar(draw, x1, y1, x2, y2, progress, color_left, color_right, bg=(50, 54, 66, 220)):
    draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=bg)
    progress = max(0.0, min(1.0, progress))
    fill_x2 = int(x1 + (x2 - x1) * progress)

    if fill_x2 <= x1:
        return

    width = fill_x2 - x1
    for i in range(width):
        t = i / max(1, width - 1)
        r = int(color_left[0] * (1 - t) + color_right[0] * t)
        g = int(color_left[1] * (1 - t) + color_right[1] * t)
        b = int(color_left[2] * (1 - t) + color_right[2] * t)
        draw.line((x1 + i, y1, x1 + i, y2), fill=(r, g, b, 255), width=1)


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


# ---------------------------------
# CALIBRATION
# ---------------------------------


def run_calibration(cap, pose):
    print(f"[Calibration] Sit normally and face the screen for {CALIBRATION_SECONDS:.0f}s...")
    deadline = time.time() + CALIBRATION_SECONDS
    samples = []
    vert_ema = EMA(VERT_EMA_ALPHA)

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            pw, ph = PROCESS_WIDTH, PROCESS_HEIGHT

            nose = landmark_px(lm, mp_pose.PoseLandmark.NOSE.value, pw, ph)
            left_sh = landmark_px(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, pw, ph)
            right_sh = landmark_px(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, pw, ph)

            sh_mid = ((left_sh[0] + right_sh[0]) // 2, (left_sh[1] + right_sh[1]) // 2)
            sh_w = max(dist(left_sh, right_sh), 1)

            norm_vert = vert_ema.update((nose[1] - sh_mid[1]) / sh_w)
            samples.append(norm_vert)

        h, w, _ = frame.shape
        remaining = max(0.0, deadline - time.time())
        pct = 1.0 - (remaining / CALIBRATION_SECONDS)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (15, 18, 24), -1)
        frame = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)

        bar_x1, bar_y1, bar_x2, bar_y2 = 30, h - 50, w - 30, h - 24
        fill_x2 = int(bar_x1 + (bar_x2 - bar_x1) * pct)

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (55, 60, 70), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y2), (39, 220, 124), -1)

        cv2.putText(frame, "LOCKED IN", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (245, 247, 250), 3)
        cv2.putText(frame, "Calibrating posture baseline...", (30, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (185, 192, 203), 2)
        cv2.putText(frame, f"{remaining:.1f}s", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 193, 7), 2)

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not samples:
        print("[Calibration] No pose detected. Using fallback threshold.")
        return FACING_VERT_FALLBACK

    samples.sort()
    median = samples[len(samples) // 2]
    threshold = median + CALIBRATION_SLACK
    print(f"[Calibration] median={median:.3f} -> threshold={threshold:.3f}")
    return threshold


# ---------------------------------
# CLASSIFIER
# ---------------------------------


def classify_raw(norm_vert, norm_horiz, smooth_motion, writing_votes, facing_threshold):
    if abs(norm_horiz) > STRONG_TURN_HORIZ:
        return "OFF_TASK"

    if norm_vert < facing_threshold and abs(norm_horiz) < CENTERED_HORIZ_MAX:
        return "FACING"

    if WRITING_DOWN_MIN <= norm_vert <= WRITING_DOWN_MAX:
        if WRITING_MOTION_LOW <= smooth_motion <= WRITING_MOTION_HIGH and writing_votes >= WRITING_VOTE_MIN_HITS:
            return "WRITING"

    if READING_DOWN_MIN <= norm_vert <= READING_DOWN_MAX:
        if abs(norm_horiz) < CENTERED_HORIZ_MAX and smooth_motion <= READING_MOTION_MAX:
            return "READING"

    return "OFF_TASK"


# ---------------------------------
# STATE MACHINE
# ---------------------------------


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


# ---------------------------------
# UI
# ---------------------------------


def render_ui(
    frame,
    status,
    internal_mode,
    vote_score,
    fps,
    phone_active,
    phone_conf,
    phone_box,
    reasons,
    debug_lines,
):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = base.size
    locked = status == "LOCKED IN"

    neon_green = hex_to_rgba("#22E38E")
    soft_green = hex_to_rgba("#22E38E", 70)
    dark_green = hex_to_rgba("#1CAA6A")
    neon_red = hex_to_rgba("#FF4D6D")
    soft_red = hex_to_rgba("#FF4D6D", 82)
    dark_red = hex_to_rgba("#D7445A")
    neon_blue = hex_to_rgba("#58A6FF")
    neon_purple = hex_to_rgba("#9B6DFF")
    soft_panel = (12, 14, 20, 175)
    hard_panel = (10, 12, 18, 225)
    border = (255, 255, 255, 30)
    white = (245, 247, 250, 255)
    muted = (180, 188, 198, 255)
    amber = hex_to_rgba("#FFC857")

    status_color = neon_green if locked else neon_red
    status_soft = soft_green if locked else soft_red

    # Top status banner
    top_x1, top_y1, top_x2, top_y2 = 18, 18, w - 18, 108
    draw.rounded_rectangle((top_x1, top_y1, top_x2, top_y2), radius=30, fill=hard_panel, outline=border, width=2)
    draw.rounded_rectangle((top_x1, top_y1, top_x2, top_y2), radius=30, fill=status_soft)

    draw_glow_text(
        draw,
        (42, 34),
        status,
        FONT_STATUS,
        status_color,
        (status_color[0], status_color[1], status_color[2], 70),
        glow_radius=2,
    )

    draw.text((w - 130, 44), f"{fps:.0f} FPS", font=FONT_SMALL, fill=white)

    # Status dot
    dot_x = w - 46
    dot_y = 82
    dot_color = neon_green if locked else neon_red
    draw.ellipse((dot_x - 10, dot_y - 10, dot_x + 10, dot_y + 10), fill=dot_color)

    # Big translucent alert when NOT LOCKED IN
    if not locked:
        alert_text = "!"
        bbox = draw.textbbox((0, 0), alert_text, font=FONT_ALERT)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        ax = (w - tw) // 2
        ay = (h - th) // 2 - 40
        draw_glow_text(
            draw,
            (ax, ay),
            alert_text,
            FONT_ALERT,
            (255, 77, 109, 90),
            (255, 77, 109, 28),
            glow_radius=4,
        )

    # Left panel
    panel_x1, panel_y1, panel_x2, panel_y2 = 18, 130, SIDE_PANEL_WIDTH, h - 24
    draw.rounded_rectangle((panel_x1, panel_y1, panel_x2, panel_y2), radius=28, fill=soft_panel, outline=border, width=2)

    draw.text((40, 152), "Current Mode", font=FONT_TITLE, fill=white)

    mode_fill = BG_CHIP_DARK
    mode_text_fill = neon_blue
    if internal_mode == "WRITING":
        mode_text_fill = neon_purple
    elif internal_mode == "READING":
        mode_text_fill = amber
    elif internal_mode in ("OFF_TASK", "PHONE"):
        mode_text_fill = neon_red
    elif internal_mode == "FACING":
        mode_text_fill = neon_blue

    chip_y = 192
    chip_w, _ = draw_chip(
        draw,
        40,
        chip_y,
        internal_mode,
        mode_fill,
        text_fill=mode_text_fill,
        outline=(255, 255, 255, 24),
    )

    if phone_active:
        draw_chip(
            draw,
            40 + chip_w + 10,
            chip_y,
            f"PHONE {phone_conf:.2f}",
            BG_CHIP_DARK,
            text_fill=neon_red,
            outline=(255, 255, 255, 24),
        )

    draw.text((40, 252), "Signals", font=FONT_TITLE, fill=white)
    signal_y = 290
    for reason_text, color in reasons[:5]:
        darker = BG_CHIP_DARK
        readable_color = color
        if color == GREEN:
            readable_color = dark_green
        elif color == RED:
            readable_color = dark_red
        elif color == YELLOW:
            readable_color = amber
        elif color == BLUE:
            readable_color = neon_blue
        elif color == PURPLE:
            readable_color = neon_purple

        _, h_chip = draw_chip(
            draw,
            40,
            signal_y,
            reason_text,
            darker,
            text_fill=readable_color,
            outline=(255, 255, 255, 20),
        )
        signal_y += h_chip + 10

    draw.text((40, signal_y + 18), "Confidence", font=FONT_TITLE, fill=white)
    bar_y = signal_y + 58
    progress = (vote_score + VOTE_CAP) / (2 * VOTE_CAP)
    bar_left = (34, 227, 142)
    bar_right = (88, 166, 255) if locked else (255, 77, 109)
    draw_gradient_bar(draw, 40, bar_y, 260, bar_y + 20, progress, bar_left, bar_right)

    draw.text((40, bar_y + 34), f"vote score {vote_score:+d}/{VOTE_CAP}", font=FONT_DEBUG, fill=muted)

    # Phone box
    if phone_active and phone_box is not None:
        x1, y1, x2, y2 = phone_box
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, outline=neon_red, width=4)
        draw_chip(
            draw,
            x1,
            max(20, y1 - 38),
            "PHONE DETECTED",
            BG_CHIP_DARK,
            text_fill=neon_red,
            outline=(255, 255, 255, 24),
        )

    # Bottom debug panel
    dbg_x1, dbg_y1, dbg_x2, dbg_y2 = SIDE_PANEL_WIDTH + 22, h - 152, w - 18, h - 20
    draw.rounded_rectangle((dbg_x1, dbg_y1, dbg_x2, dbg_y2), radius=26, fill=soft_panel, outline=border, width=2)

    draw.text((dbg_x1 + 22, dbg_y1 + 18), "Live Debug", font=FONT_TITLE, fill=white)

    yy = dbg_y1 + 56
    for line in debug_lines:
        draw.text((dbg_x1 + 22, yy), line, font=FONT_DEBUG, fill=muted)
        yy += 24

    final = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)


# ---------------------------------
# MAIN
# ---------------------------------


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    phone_model = YOLO("yolo11n.pt")

    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ) as pose:

        facing_threshold = run_calibration(cap, pose)

        vert_ema = EMA(VERT_EMA_ALPHA)
        horiz_ema = EMA(HORIZ_EMA_ALPHA)
        motion_roller = RollingMean(MOTION_SMOOTH_WINDOW)
        motion_votes = deque(maxlen=WRITING_VOTE_WINDOW)

        state_machine = AttentionStateMachine()

        prev_left_sh = None
        prev_right_sh = None

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
            h, w, _ = frame.shape
            now = time.time()

            fps_buf.append(1.0 / max(now - last_time, 1e-6))
            last_time = now
            fps = sum(fps_buf) / len(fps_buf)

            # Phone detection every frame
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

            # Pose processing
            small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            sx = w / PROCESS_WIDTH
            sy = h / PROCESS_HEIGHT

            raw_state = "OFF_TASK"
            norm_vert = 0.0
            norm_horiz = 0.0
            smooth_motion = 0.0
            writing_votes = 0

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                pw, ph = PROCESS_WIDTH, PROCESS_HEIGHT

                nose = landmark_px(lm, mp_pose.PoseLandmark.NOSE.value, pw, ph)
                left_sh = landmark_px(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, pw, ph)
                right_sh = landmark_px(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, pw, ph)

                sh_mid = ((left_sh[0] + right_sh[0]) // 2, (left_sh[1] + right_sh[1]) // 2)
                sh_w = max(dist(left_sh, right_sh), 1)

                norm_vert = vert_ema.update((nose[1] - sh_mid[1]) / sh_w)
                norm_horiz = horiz_ema.update((nose[0] - sh_mid[0]) / sh_w)

                if prev_left_sh is not None and prev_right_sh is not None:
                    raw_motion = (dist(left_sh, prev_left_sh) + dist(right_sh, prev_right_sh)) / (2 * sh_w)
                else:
                    raw_motion = 0.0

                prev_left_sh = left_sh
                prev_right_sh = right_sh

                smooth_motion = motion_roller.update(raw_motion)
                motion_votes.append(WRITING_MOTION_LOW <= raw_motion <= WRITING_MOTION_HIGH)
                writing_votes = sum(motion_votes)

                raw_state = classify_raw(
                    norm_vert,
                    norm_horiz,
                    smooth_motion,
                    writing_votes,
                    facing_threshold,
                )

                mp_drawing.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(210, 210, 220), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(120, 170, 255), thickness=2),
                )

                nose_d = (int(nose[0] * sx), int(nose[1] * sy))
                sh_mid_d = (int(sh_mid[0] * sx), int(sh_mid[1] * sy))
                cv2.circle(frame, nose_d, 6, (80, 230, 230), -1)
                cv2.circle(frame, sh_mid_d, 6, (255, 200, 80), -1)
                cv2.line(frame, nose_d, sh_mid_d, (80, 230, 230), 2)

            else:
                prev_left_sh = None
                prev_right_sh = None
                raw_state = "OFF_TASK"

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
                    reasons.append(("Writing Motion", PURPLE))
                else:
                    reasons.append(("Off Task", RED))

                if abs(norm_horiz) > STRONG_TURN_HORIZ:
                    reasons.append(("Turned Away", RED))
                elif abs(norm_horiz) < CENTERED_HORIZ_MAX:
                    reasons.append(("Centered", GREEN))

                if WRITING_DOWN_MIN <= norm_vert <= WRITING_DOWN_MAX:
                    reasons.append(("Looking Down", YELLOW))
                elif norm_vert < facing_threshold:
                    reasons.append(("Facing Screen", GREEN))

            debug_lines = [
                f"raw: {raw_state:<8}  final: {status:<14}  mode: {internal_mode}",
                f"vert: {norm_vert:+.3f}  horiz: {norm_horiz:+.3f}  face_thr: {facing_threshold:+.3f}",
                f"motion: {smooth_motion:.4f}  writing_votes: {writing_votes}/{WRITING_VOTE_WINDOW}",
                f"phone_active: {phone_active}  phone_conf: {last_phone_conf:.2f}",
            ]

            frame = render_ui(
                frame=frame,
                status=status,
                internal_mode=internal_mode,
                vote_score=state_machine.vote_score,
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