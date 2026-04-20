"""
ui_app.py — Locked In UI overlay.

Design language: Arc Browser inspired — glassmorphism, soft gradients,
refined sans-serif typography, generous spacing, subtle depth.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE
# ══════════════════════════════════════════════════════════════════════════════
WHITE      = (248, 250, 252, 255)
SOFT       = (200, 207, 220, 255)
DIM        = (148, 156, 172, 255)
DIM_2      = (110, 118, 135, 255)

GREEN      = (74,  222, 128, 255)
GREEN_GLOW = (74,  222, 128, 80)
RED        = (248, 113, 113, 255)
RED_GLOW   = (248, 113, 113, 80)
AMBER      = (251, 191, 36,  255)
CYAN       = (103, 232, 249, 255)
VIOLET     = (167, 139, 250, 255)

GLASS_MAIN = (14,  16,  24,  215)
GLASS_DEEP = (9,   11,  18,  225)
GLASS_SOFT = (22,  25,  36,  195)

STROKE_HI  = (255, 255, 255, 38)
STROKE_LO  = (255, 255, 255, 14)


# ══════════════════════════════════════════════════════════════════════════════
# FONTS
# ══════════════════════════════════════════════════════════════════════════════
def _font(size: int, weight: str = "regular") -> ImageFont.ImageFont:
    is_bold = weight in ("bold", "black", "medium")
    bold_paths = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    reg_paths = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in (bold_paths if is_bold else reg_paths):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _mono(size: int, bold: bool = True) -> ImageFont.ImageFont:
    paths = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return _font(size, "bold")


F_DISPLAY   = _font(64, "black")
F_HERO      = _font(44, "bold")
F_TITLE     = _font(28, "bold")
F_SUBTITLE  = _font(20, "medium")
F_BODY      = _font(17, "medium")
F_CAPTION   = _font(14, "medium")
F_TINY      = _font(12, "medium")

M_DISPLAY   = _mono(56, bold=True)
M_HERO      = _mono(36, bold=True)
M_LARGE     = _mono(24, bold=True)
M_MED       = _mono(18, bold=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _tw(draw, text, font) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def _th(draw, text, font) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[3] - bb[1]


def _rounded(draw, xy, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _glass_panel(overlay, draw, xy, radius=18, fill=GLASS_MAIN,
                 stroke=STROKE_HI, highlight=True):
    x1, y1, x2, y2 = xy
    _rounded(draw, xy, radius=radius, fill=fill, outline=stroke, width=1)
    if highlight:
        draw.line([(x1 + radius, y1 + 1), (x2 - radius, y1 + 1)],
                  fill=(255, 255, 255, 22), width=1)


def _soft_glow(overlay, cx, cy, radius, color, intensity=0.6):
    size = radius * 2
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    for i in range(8, 0, -1):
        r = int(radius * i / 8)
        a = int(intensity * 30 * (i / 8) ** 2)
        gd.ellipse(
            (radius - r, radius - r, radius + r, radius + r),
            fill=(*color[:3], a),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=radius / 8))
    overlay.paste(glow, (cx - radius, cy - radius), glow)


def _gradient_bar(draw, x1, y1, x2, y2, value, color_a, color_b,
                  bg=(30, 34, 48, 180), radius=4):
    _rounded(draw, (x1, y1, x2, y2), radius=radius, fill=bg)
    bar_w = int((x2 - x1) * max(0, min(1, value)))
    if bar_w < 2:
        return
    grad = Image.new("RGBA", (bar_w, y2 - y1))
    for i in range(bar_w):
        t = i / max(bar_w - 1, 1)
        r = int(color_a[0] * (1 - t) + color_b[0] * t)
        g = int(color_a[1] * (1 - t) + color_b[1] * t)
        b = int(color_a[2] * (1 - t) + color_b[2] * t)
        for yy in range(y2 - y1):
            grad.putpixel((i, yy), (r, g, b, 255))
    mask = Image.new("L", grad.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, bar_w, y2 - y1), radius=radius, fill=255
    )
    draw._image.paste(grad, (x1, y1), mask)


def _draw_attention_arc(overlay, cx, cy, radius, value, color, now,
                        thickness=6):
    size = radius * 2 + thickness * 4
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rd   = ImageDraw.Draw(img)
    ox, oy = size // 2, size // 2

    rd.ellipse(
        (ox - radius, oy - radius, ox + radius, oy + radius),
        outline=(*color[:3], 22), width=thickness,
    )

    start_deg = -90
    sweep_deg = value * 360
    if sweep_deg > 0:
        rd.arc(
            (ox - radius, oy - radius, ox + radius, oy + radius),
            start=start_deg, end=start_deg + sweep_deg,
            fill=color, width=thickness,
        )

    tip_rad = math.radians(start_deg + sweep_deg)
    tx = ox + int(radius * math.cos(tip_rad))
    ty = oy + int(radius * math.sin(tip_rad))
    pulse_r = 5 + int(2 * math.sin(now * 3.5))
    halo = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    for i in range(6, 0, -1):
        a = int(40 * (i / 6) ** 2)
        hd.ellipse((20 - i * 2, 20 - i * 2, 20 + i * 2, 20 + i * 2),
                   fill=(*color[:3], a))
    halo = halo.filter(ImageFilter.GaussianBlur(3))
    img.paste(halo, (tx - 20, ty - 20), halo)
    rd.ellipse((tx - pulse_r, ty - pulse_r, tx + pulse_r, ty + pulse_r),
               fill=(255, 255, 255, 255))

    overlay.paste(img, (cx - ox, cy - oy), img)


def _fmt_time(seconds):
    s = int(max(0, seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# BUTTONS
# ══════════════════════════════════════════════════════════════════════════════
def _button(draw, x1, y1, x2, y2, label, color, primary=False, font=F_BODY):
    if primary:
        _rounded(draw, (x1, y1, x2, y2), radius=(y2 - y1) // 2,
                 fill=(*color[:3], 235))
        draw.line([(x1 + 10, y1 + 2), (x2 - 10, y1 + 2)],
                  fill=(255, 255, 255, 60), width=1)
        text_col = (15, 18, 28, 255)
    else:
        _rounded(draw, (x1, y1, x2, y2), radius=(y2 - y1) // 2,
                 fill=GLASS_SOFT, outline=(*color[:3], 180), width=1)
        text_col = color

    tw = _tw(draw, label, font)
    th = _th(draw, label, font)
    draw.text(
        (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 - th) // 2 - 2),
        label, font=font, fill=text_col,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAYS
# ══════════════════════════════════════════════════════════════════════════════

def _render_idle_overlay(overlay, draw, w, h):
    draw.rectangle((0, 0, w, h), fill=(0, 0, 0, 180))

    cw, ch = 560, 340
    x1, y1 = (w - cw) // 2, (h - ch) // 2
    x2, y2 = x1 + cw, y1 + ch
    _glass_panel(overlay, draw, (x1, y1, x2, y2), radius=24, fill=GLASS_DEEP)
    _soft_glow(overlay, (x1 + x2) // 2, y1 + 60, 180, CYAN, intensity=0.5)

    draw.text((x1 + 36, y1 + 38), "Locked In", font=F_HERO, fill=WHITE)
    draw.text((x1 + 36, y1 + 90), "Attention Monitor",
              font=F_SUBTITLE, fill=DIM)

    draw.line([(x1 + 36, y1 + 134), (x2 - 36, y1 + 134)],
              fill=STROKE_LO, width=1)

    draw.text((x1 + 36, y1 + 150), "Start a focus session to begin",
              font=F_BODY, fill=SOFT)
    draw.text((x1 + 36, y1 + 180),
              "Your webcam analyses gaze and posture locally.",
              font=F_CAPTION, fill=DIM_2)

    draw.text((x1 + 36, y1 + 230), "SHORTCUTS", font=F_TINY, fill=DIM_2)
    draw.text((x1 + 36, y1 + 250),
              "Space  pause/resume       E  end       Q  quit",
              font=F_CAPTION, fill=DIM)

    cx = (x1 + x2) // 2
    bx1, bx2 = cx - 100, cx + 100
    by1, by2 = y2 - 64, y2 - 24
    _button(draw, bx1, by1, bx2, by2, "Start Session", GREEN,
            primary=True, font=F_SUBTITLE)
    return [{"id": "start", "rect": (bx1, by1, bx2, by2)}]


def _render_pause_overlay(overlay, draw, w, h):
    draw.rectangle((0, 0, w, h), fill=(0, 0, 0, 140))
    cx, cy = w // 2, h // 2

    cw, ch = 360, 200
    x1, y1 = cx - cw // 2, cy - ch // 2
    x2, y2 = x1 + cw, y1 + ch
    _glass_panel(overlay, draw, (x1, y1, x2, y2), radius=20, fill=GLASS_DEEP)
    _soft_glow(overlay, cx, y1 + 50, 120, AMBER, intensity=0.5)

    hero_text = "Paused"
    tw = _tw(draw, hero_text, F_HERO)
    draw.text((cx - tw // 2, y1 + 32), hero_text, font=F_HERO, fill=AMBER)

    caption = "Session time is not counting"
    cw_ = _tw(draw, caption, F_CAPTION)
    draw.text((cx - cw_ // 2, y1 + 94), caption, font=F_CAPTION, fill=DIM)

    bx1, bx2 = cx - 90, cx + 90
    by1, by2 = y2 - 56, y2 - 20
    _button(draw, bx1, by1, bx2, by2, "Resume", GREEN, primary=True)
    return [{"id": "resume", "rect": (bx1, by1, bx2, by2)}]


def _render_end_overlay(overlay, draw, w, h, report):
    draw.rectangle((0, 0, w, h), fill=(0, 0, 0, 215))

    cw, ch = 780, 560
    x1, y1 = (w - cw) // 2, (h - ch) // 2
    x2, y2 = x1 + cw, y1 + ch
    _glass_panel(overlay, draw, (x1, y1, x2, y2), radius=28, fill=GLASS_DEEP)

    score = report.score
    if   score >= 80: accent, accent_b = GREEN, CYAN
    elif score >= 60: accent, accent_b = CYAN,  VIOLET
    elif score >= 40: accent, accent_b = AMBER, RED
    else:             accent, accent_b = RED,   RED

    _soft_glow(overlay, x1 + 180, y1 + 170, 190, accent, intensity=0.7)

    draw.text((x1 + 36, y1 + 32), "Session Complete", font=F_TITLE, fill=WHITE)
    draw.text((x1 + 36, y1 + 70), "Your Locked In Score",
              font=F_CAPTION, fill=DIM)

    score_str = f"{score:03d}"
    draw.text((x1 + 36, y1 + 100), score_str, font=M_DISPLAY, fill=accent)

    gx1, gy1 = x1 + 36, y1 + 200
    gx2, gy2 = gx1 + 104, gy1 + 104
    _rounded(draw, (gx1, gy1, gx2, gy2), radius=18,
             fill=GLASS_SOFT, outline=(*accent[:3], 180), width=2)
    grade_w = _tw(draw, report.grade, F_DISPLAY)
    grade_h = _th(draw, report.grade, F_DISPLAY)
    draw.text(
        (gx1 + (104 - grade_w) // 2, gy1 + (104 - grade_h) // 2 - 8),
        report.grade, font=F_DISPLAY, fill=accent,
    )
    draw.text((gx1 + 24, gy2 + 10), "GRADE", font=F_TINY, fill=DIM_2)

    mx1, my1 = x1 + 260, y1 + 104
    mx2, my2 = x2 - 36, my1 + 210
    _glass_panel(overlay, draw, (mx1, my1, mx2, my2),
                 radius=16, fill=GLASS_SOFT, highlight=False)

    metrics = [
        ("Duration",     _fmt_time(report.duration_sec)),
        ("Focused",      _fmt_time(report.focused_sec)),
        ("Best Streak",  _fmt_time(report.longest_streak_sec)),
        ("Efficiency",   f"{report.efficiency_pct:.0f}%"),
        ("Distractions", str(report.distraction_count)),
        ("Phone Events", str(report.phone_count)),
    ]
    col_w = (mx2 - mx1 - 32) // 2
    for i, (label, value) in enumerate(metrics):
        col = i % 2
        row = i // 2
        cx_ = mx1 + 16 + col * col_w
        cy_ = my1 + 18 + row * 62
        draw.text((cx_, cy_), label, font=F_CAPTION, fill=DIM)
        draw.text((cx_, cy_ + 22), value, font=M_LARGE, fill=WHITE)

    bd = report.breakdown
    by = y1 + 360
    draw.text((x1 + 36, by), "Score Breakdown", font=F_SUBTITLE, fill=WHITE)
    draw.text((x1 + 36, by + 26), "How your score was calculated",
              font=F_CAPTION, fill=DIM_2)

    components = [
        ("Efficiency",   "60%", bd.get("efficiency", 0),  accent, accent_b),
        ("Streak",       "25%", bd.get("streak", 0),      CYAN,   VIOLET),
        ("Consistency",  "15%", bd.get("consistency", 0), VIOLET, CYAN),
    ]
    cy_ = by + 64
    for label, weight, val, ca, cb in components:
        draw.text((x1 + 36, cy_), label, font=F_BODY, fill=WHITE)
        draw.text((x1 + 160, cy_ + 2), weight, font=F_CAPTION, fill=DIM_2)

        bar_x1 = x1 + 240
        bar_x2 = x2 - 100
        _gradient_bar(draw, bar_x1, cy_ + 4, bar_x2, cy_ + 20,
                      val / 100.0, ca, cb)
        val_str = f"{val:.0f}"
        vw = _tw(draw, val_str, M_MED)
        draw.text((x2 - 50 - vw, cy_ + 1), val_str, font=M_MED, fill=WHITE)
        cy_ += 32

    btn_y1 = y2 - 60
    btn_y2 = y2 - 20
    btns = []

    qx1, qx2 = x1 + 36, x1 + 36 + 130
    _button(draw, qx1, btn_y1, qx2, btn_y2, "Quit", RED, primary=False)
    btns.append({"id": "quit", "rect": (qx1, btn_y1, qx2, btn_y2)})

    nx2 = x2 - 36
    nx1 = nx2 - 180
    _button(draw, nx1, btn_y1, nx2, btn_y2, "New Session", accent,
            primary=True)
    btns.append({"id": "new", "rect": (nx1, btn_y1, nx2, btn_y2)})

    return btns


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render_app_ui(
    frame,
    status: str,
    reasons: list[str],
    attention_score: int = 50,
    session_elapsed: float = 0.0,
    locked_in_seconds: float = 0.0,
    current_streak: float = 0.0,
    longest_streak: float = 0.0,
    phone_active: bool = False,
    phone_box=None,
    session_state: str = "RUNNING",
    end_report=None,
):
    now   = time.time()
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base  = Image.fromarray(rgb).convert("RGBA")

    vignette = Image.new("RGBA", base.size, (0, 0, 0, 55))
    base     = Image.alpha_composite(base, vignette)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    w, h    = base.size
    locked  = status == "LOCKED IN"
    accent  = GREEN if locked else RED

    show_hud = session_state in ("RUNNING", "PAUSED")
    buttons: list[dict] = []

    if show_hud:
        # ═════════════════════════════════════════════════════════════════════
        # TOP BAR
        # ═════════════════════════════════════════════════════════════════════
        tb_x1, tb_y1 = 24, 20
        tb_x2, tb_y2 = w - 24, 86
        _glass_panel(overlay, draw, (tb_x1, tb_y1, tb_x2, tb_y2), radius=18)

        draw.text((tb_x1 + 24, tb_y1 + 14), "Locked In",
                  font=F_TITLE, fill=WHITE)
        draw.text((tb_x1 + 24, tb_y1 + 44), "Attention Monitor",
                  font=F_CAPTION, fill=DIM_2)

        status_text = "Locked In" if locked else "Locked Out"
        pill_tw = _tw(draw, status_text, F_SUBTITLE)
        pill_w  = pill_tw + 72
        pill_h  = 40
        pill_x1 = (w - pill_w) // 2
        pill_y1 = tb_y1 + (66 - pill_h) // 2
        pill_x2 = pill_x1 + pill_w
        pill_y2 = pill_y1 + pill_h

        _soft_glow(overlay, (pill_x1 + pill_x2) // 2,
                   (pill_y1 + pill_y2) // 2, 60, accent, intensity=0.6)
        _rounded(draw, (pill_x1, pill_y1, pill_x2, pill_y2),
                 radius=pill_h // 2, fill=GLASS_SOFT,
                 outline=(*accent[:3], 200), width=1)
        dot_r = 6
        dot_cx = pill_x1 + 22
        dot_cy = (pill_y1 + pill_y2) // 2
        if locked:
            pr = dot_r + int(2 * math.sin(now * 3))
            draw.ellipse((dot_cx - pr, dot_cy - pr, dot_cx + pr, dot_cy + pr),
                         fill=(*accent[:3], 90))
        draw.ellipse((dot_cx - dot_r, dot_cy - dot_r,
                      dot_cx + dot_r, dot_cy + dot_r), fill=accent)
        draw.text((pill_x1 + 42, pill_y1 + 8),
                  status_text, font=F_SUBTITLE, fill=accent)

        clock_str = time.strftime("%H:%M")
        ctw = _tw(draw, clock_str, M_LARGE)
        draw.text((tb_x2 - ctw - 24, tb_y1 + 14),
                  clock_str, font=M_LARGE, fill=WHITE)
        draw.text((tb_x2 - ctw - 24, tb_y1 + 44),
                  "Local time", font=F_CAPTION, fill=DIM_2)

        # ═════════════════════════════════════════════════════════════════════
        # LEFT PANEL — attention
        # ═════════════════════════════════════════════════════════════════════
        lp_x1, lp_y1 = 24, 104
        lp_x2, lp_y2 = 300, 420
        _glass_panel(overlay, draw, (lp_x1, lp_y1, lp_x2, lp_y2),
                     radius=18, fill=GLASS_MAIN)

        draw.text((lp_x1 + 20, lp_y1 + 18), "Attention", font=F_CAPTION, fill=DIM)
        draw.text((lp_x1 + 20, lp_y1 + 36), "Real-time focus signal",
                  font=F_TINY, fill=DIM_2)

        ring_cx = (lp_x1 + lp_x2) // 2
        ring_cy = lp_y1 + 160
        ring_r  = 74
        _soft_glow(overlay, ring_cx, ring_cy, 120, accent, intensity=0.5)
        _draw_attention_arc(overlay, ring_cx, ring_cy, ring_r,
                            attention_score / 100, accent, now, thickness=8)

        score_str = f"{attention_score:03d}"
        sw = _tw(draw, score_str, M_HERO)
        sh = _th(draw, score_str, M_HERO)
        draw.text((ring_cx - sw // 2, ring_cy - sh // 2 - 8),
                  score_str, font=M_HERO, fill=WHITE)
        sublabel = "SCORE"
        sbw = _tw(draw, sublabel, F_TINY)
        draw.text((ring_cx - sbw // 2, ring_cy + 20),
                  sublabel, font=F_TINY, fill=DIM)

        bar_y = lp_y1 + 258
        draw.text((lp_x1 + 20, bar_y), "SIGNAL LEVEL", font=F_TINY, fill=DIM_2)
        _gradient_bar(draw, lp_x1 + 20, bar_y + 20,
                      lp_x2 - 20, bar_y + 32,
                      attention_score / 100, accent,
                      CYAN if locked else RED)

        chip_y = lp_y1 + 300
        chip_text = ("Phone Override" if phone_active
                     else ("Active" if locked else "Standby"))
        chip_col  = (RED if phone_active else
                     (GREEN if locked else DIM))
        _rounded(draw,
                 (lp_x1 + 20, chip_y, lp_x2 - 20, chip_y + 28),
                 radius=14, fill=GLASS_SOFT,
                 outline=(*chip_col[:3], 160), width=1)
        ctw3 = _tw(draw, chip_text, F_CAPTION)
        draw.text(
            ((lp_x1 + lp_x2 - ctw3) // 2, chip_y + 7),
            chip_text, font=F_CAPTION, fill=chip_col,
        )

        # ═════════════════════════════════════════════════════════════════════
        # RIGHT PANEL — session
        # ═════════════════════════════════════════════════════════════════════
        rp_x1, rp_y1 = w - 300, 104
        rp_x2, rp_y2 = w - 24, 420
        _glass_panel(overlay, draw, (rp_x1, rp_y1, rp_x2, rp_y2),
                     radius=18, fill=GLASS_MAIN)

        draw.text((rp_x1 + 20, rp_y1 + 18), "Session",
                  font=F_CAPTION, fill=DIM)
        draw.text((rp_x1 + 20, rp_y1 + 36), "Running totals",
                  font=F_TINY, fill=DIM_2)

        metrics = [
            ("Elapsed",  _fmt_time(session_elapsed),     WHITE),
            ("Focused",  _fmt_time(locked_in_seconds),   WHITE),
            ("Streak",   _fmt_time(current_streak),      accent),
            ("Best",     _fmt_time(longest_streak),      WHITE),
        ]

        my = rp_y1 + 72
        for label, value, col in metrics:
            draw.text((rp_x1 + 20, my + 6), label, font=F_CAPTION, fill=DIM)
            vw = _tw(draw, value, M_LARGE)
            draw.text((rp_x2 - vw - 20, my),
                      value, font=M_LARGE, fill=col)
            draw.line([(rp_x1 + 20, my + 38), (rp_x2 - 20, my + 38)],
                      fill=STROKE_LO, width=1)
            my += 48

        eff = int(100 * locked_in_seconds / max(session_elapsed, 1))
        eff_str = f"{eff}%"
        eff_col = (GREEN if eff >= 60
                   else (AMBER if eff >= 30 else RED))

        eff_box_y1 = my + 4
        eff_box_y2 = rp_y2 - 16
        _rounded(draw,
                 (rp_x1 + 16, eff_box_y1, rp_x2 - 16, eff_box_y2),
                 radius=14, fill=GLASS_SOFT)
        draw.text((rp_x1 + 28, eff_box_y1 + 14), "Efficiency",
                  font=F_CAPTION, fill=DIM)
        ew = _tw(draw, eff_str, M_HERO)
        draw.text((rp_x2 - ew - 28, eff_box_y1 + 8),
                  eff_str, font=M_HERO, fill=eff_col)

        # ═════════════════════════════════════════════════════════════════════
        # BOTTOM PANEL — signals
        # ═════════════════════════════════════════════════════════════════════
        bp_x1, bp_y1 = 24, h - 200
        bp_x2, bp_y2 = w - 24, h - 24
        _glass_panel(overlay, draw, (bp_x1, bp_y1, bp_x2, bp_y2),
                     radius=18, fill=GLASS_MAIN)

        draw.text((bp_x1 + 24, bp_y1 + 20), "Attention Signals",
                  font=F_TITLE, fill=WHITE)
        count_str = f"{len(reasons or [])} active"
        ctw4 = _tw(draw, count_str, F_CAPTION)
        draw.text((bp_x2 - ctw4 - 30, bp_y1 + 30),
                  count_str, font=F_CAPTION, fill=DIM)

        draw.line([(bp_x1 + 24, bp_y1 + 66), (bp_x2 - 24, bp_y1 + 66)],
                  fill=STROKE_LO, width=1)

        shown = (reasons or ["No signals detected"])[:4]
        rx = bp_x1 + 24
        ry = bp_y1 + 82
        col_w = (bp_x2 - bp_x1 - 48) // 2

        for i, reason in enumerate(shown):
            col_x = rx + (col_w + 24) * (i // 2)
            row_y = ry + (i % 2) * 44

            _rounded(draw,
                     (col_x, row_y, col_x + 32, row_y + 32),
                     radius=10, fill=(*accent[:3], 200))
            idx_txt = f"{i + 1}"
            itw = _tw(draw, idx_txt, F_SUBTITLE)
            draw.text(
                (col_x + (32 - itw) // 2, row_y + 4),
                idx_txt, font=F_SUBTITLE, fill=(15, 18, 28, 255),
            )
            draw.text(
                (col_x + 46, row_y + 6),
                reason, font=F_BODY, fill=WHITE,
            )

        # ═════════════════════════════════════════════════════════════════════
        # CONTROL BUTTONS
        # ═════════════════════════════════════════════════════════════════════
        btn_y1 = 104
        btn_y2 = btn_y1 + 40
        cx = w // 2
        if session_state == "RUNNING":
            px1, px2 = cx - 150, cx - 16
            ex1, ex2 = cx + 16,  cx + 150
            _button(draw, px1, btn_y1, px2, btn_y2, "Pause", AMBER)
            _button(draw, ex1, btn_y1, ex2, btn_y2, "End", RED)
            buttons.append({"id": "pause", "rect": (px1, btn_y1, px2, btn_y2)})
            buttons.append({"id": "end",   "rect": (ex1, btn_y1, ex2, btn_y2)})

        # Warning mark
        if not locked and session_state == "RUNNING":
            pulsed_alpha = int(30 + 18 * math.sin(now * 2.2))
            mark = "!"
            big_font = _font(160, "black")
            bb  = draw.textbbox((0, 0), mark, font=big_font)
            tw2 = bb[2] - bb[0]
            th2 = bb[3] - bb[1]
            draw.text(
                ((w - tw2) // 2, (h - th2) // 2 - 40),
                mark, font=big_font,
                fill=(248, 113, 113, pulsed_alpha),
            )

        # Phone box
        if phone_active and phone_box is not None and session_state == "RUNNING":
            px1, py1, px2, py2 = phone_box
            pad = 12
            blink = (*RED[:3], int(180 + 70 * math.sin(now * 5)))
            _rounded(draw,
                     (px1 - pad, py1 - pad, px2 + pad, py2 + pad),
                     radius=12, outline=blink, width=3)

            tag_x1 = max(24, px1 - pad)
            tag_y1 = max(24, py1 - pad - 36)
            tag_x2 = tag_x1 + 176
            tag_y2 = tag_y1 + 30
            _rounded(draw, (tag_x1, tag_y1, tag_x2, tag_y2),
                     radius=8, fill=(40, 10, 10, 240),
                     outline=RED, width=1)
            draw.text((tag_x1 + 12, tag_y1 + 7),
                      "Phone Detected", font=F_CAPTION, fill=RED)

    # ══════════════════════════════════════════════════════════════════════════
    # OVERLAYS
    # ══════════════════════════════════════════════════════════════════════════
    if session_state == "IDLE":
        buttons = _render_idle_overlay(overlay, draw, w, h)
    elif session_state == "PAUSED":
        buttons = _render_pause_overlay(overlay, draw, w, h)
    elif session_state == "ENDED" and end_report is not None:
        buttons = _render_end_overlay(overlay, draw, w, h, end_report)

    final = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR), buttons