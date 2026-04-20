"""
ui_app.py — Production UI overlay for Locked In.

Design language: military/aerospace HUD — deep blacks, tight grid,
monospaced readouts, glowing accent lines, animated attention meter.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── palette ───────────────────────────────────────────────────────────────────
WHITE      = (245, 248, 252, 255)
SOFT       = (185, 193, 205, 255)
DIM        = (130, 138, 152, 255)
GHOST      = (255, 255, 255, 22)

GREEN      = (32,  220, 120, 255)
GREEN_DIM  = (32,  220, 120, 60)
RED        = (240, 60,  60,  255)
RED_DIM    = (240, 60,  60,  60)
AMBER      = (255, 185, 40,  255)
CYAN       = (40,  210, 230, 255)
CYAN_DIM   = (40,  210, 230, 40)

BG_MAIN    = (4,   5,   9,   235)
BG_PANEL   = (8,   9,   14,  230)
BG_PANEL_2 = (11,  13,  19,  225)
BG_PANEL_3 = (14,  16,  24,  218)

FRAME_HI   = (255, 255, 255, 55)
FRAME_LO   = (255, 255, 255, 18)
SCANLINE   = (0,   0,   0,   32)


# ── font loader ───────────────────────────────────────────────────────────────
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    mono_candidates = [
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/System/Library/Fonts/Courier New.ttf",
        "/Library/Fonts/Courier New Bold.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    sans_bold = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    sans_reg = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    candidates = mono_candidates + (sans_bold if bold else sans_reg)
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# Pre-load font sizes
F_HUGE    = _font(96, bold=True)
F_LARGE   = _font(52, bold=True)
F_MED     = _font(28, bold=True)
F_BODY    = _font(22, bold=True)
F_SMALL   = _font(18, bold=True)
F_TINY    = _font(15, bold=True)
F_LABEL   = _font(14, bold=True)


# ── drawing primitives ────────────────────────────────────────────────────────

def _tw(draw: ImageDraw.Draw, text: str, font) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def _th(draw: ImageDraw.Draw, text: str, font) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[3] - bb[1]


def _panel(draw: ImageDraw.Draw, xy, fill, outline=FRAME_HI, width: int = 1):
    draw.rectangle(xy, fill=fill, outline=outline, width=width)


def _corner_marks(draw: ImageDraw.Draw, x1, y1, x2, y2, size: int = 16, color=FRAME_HI, width: int = 2):
    """Draw corner-bracket marks on a rectangle."""
    lines = [
        # TL
        (x1, y1,        x1 + size, y1),
        (x1, y1,        x1,        y1 + size),
        # TR
        (x2 - size, y1, x2,        y1),
        (x2,        y1, x2,        y1 + size),
        # BL
        (x1, y2 - size, x1,        y2),
        (x1, y2,        x1 + size, y2),
        # BR
        (x2 - size, y2, x2,        y2),
        (x2, y2 - size, x2,        y2),
    ]
    for ax, ay, bx, by in lines:
        draw.line([(ax, ay), (bx, by)], fill=color, width=width)


def _h_rule(draw: ImageDraw.Draw, x1: int, x2: int, y: int, color=FRAME_LO):
    draw.line([(x1, y), (x2, y)], fill=color, width=1)


def _scanlines(overlay: Image.Image, h: int, w: int, step: int = 3):
    """Burn very faint horizontal scanlines into the overlay."""
    draw = ImageDraw.Draw(overlay)
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=SCANLINE, width=1)


# ── bar meter ─────────────────────────────────────────────────────────────────

def _bar_meter(
    draw: ImageDraw.Draw,
    x1: int, y1: int, x2: int, y2: int,
    value: float,          # 0.0–1.0
    color_on,
    color_off=None,
    segments: int = 20,
    gap: int = 2,
):
    color_off = color_off or BG_PANEL_3
    bar_w = x2 - x1
    seg_w = (bar_w - gap * (segments - 1)) / segments
    filled = int(value * segments)

    for i in range(segments):
        sx = x1 + int(i * (seg_w + gap))
        ex = sx + int(seg_w)
        fill = color_on if i < filled else color_off
        draw.rectangle((sx, y1, ex, y2), fill=fill)


# ── radial attention ring ─────────────────────────────────────────────────────

def _draw_attention_ring(
    overlay: Image.Image,
    cx: int, cy: int,
    radius: int,
    score: float,          # 0.0–1.0
    accent_rgba,
    now: float,
):
    """
    Draw an arc-based attention ring (no external libraries).
    Uses a small temp RGBA image composed onto the overlay.
    """
    size   = radius * 2 + 20
    ring_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rd       = ImageDraw.Draw(ring_img)

    ox, oy = size // 2, size // 2
    r_outer = radius
    r_inner = radius - 8

    # Background ring
    rd.ellipse((ox - r_outer, oy - r_outer, ox + r_outer, oy + r_outer),
               outline=(*accent_rgba[:3], 22), width=8)

    # Filled arc — simulate with many short lines on a circle
    arc_deg    = score * 340        # max 340° so there's always a gap at top
    start_deg  = -100               # start from top-ish
    steps      = max(2, int(arc_deg / 2))

    for i in range(steps):
        angle_deg  = start_deg + (arc_deg * i / max(steps - 1, 1))
        angle_rad  = math.radians(angle_deg)
        alpha      = max(80, min(255, int(60 + 195 * score)))
        ax = ox + int(r_inner * math.cos(angle_rad))
        ay = oy + int(r_inner * math.sin(angle_rad))
        bx = ox + int(r_outer * math.cos(angle_rad))
        by = oy + int(r_outer * math.sin(angle_rad))
        rd.line([(ax, ay), (bx, by)], fill=(*accent_rgba[:3], alpha), width=2)

    # Pulsing dot at arc tip
    tip_angle = math.radians(start_deg + arc_deg)
    tx = ox + int((r_inner + 4) * math.cos(tip_angle))
    ty = oy + int((r_inner + 4) * math.sin(tip_angle))
    pulse_alpha = int(180 + 70 * math.sin(now * 4))
    rd.ellipse((tx - 4, ty - 4, tx + 4, ty + 4),
               fill=(*accent_rgba[:3], pulse_alpha))

    # Compose
    px = cx - ox
    py = cy - oy
    overlay.paste(ring_img, (px, py), ring_img)


# ── outer frame ───────────────────────────────────────────────────────────────

def _draw_frame(draw: ImageDraw.Draw, w: int, h: int):
    m = 12
    draw.rectangle((m, m, w - m, h - m), outline=FRAME_HI, width=1)
    # tick marks
    tick = 20
    for x in [m, w - m]:
        for y in [m, h - m]:
            draw.line([(x - tick if x > m else x, y), (x + tick if x < m else x, y)],
                      fill=FRAME_HI, width=2)
            draw.line([(x, y - tick if y > m else y), (x, y + tick if y < m else y)],
                      fill=FRAME_HI, width=2)

    # side tick clusters
    for offset in [-60, 0, 60]:
        mx = w // 2 + offset
        draw.line([(mx - 3, m), (mx + 3, m)], fill=FRAME_LO, width=1)
        draw.line([(mx - 3, h - m), (mx + 3, h - m)], fill=FRAME_LO, width=1)


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── main render ───────────────────────────────────────────────────────────────

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
):
    now    = time.time()
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base   = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw   = ImageDraw.Draw(overlay)

    w, h   = base.size
    locked = status == "LOCKED IN"
    accent = GREEN if locked else RED
    accent_dim = GREEN_DIM if locked else RED_DIM

    # ── base vignette ─────────────────────────────────────────────────────────
    draw.rectangle((0, 0, w, h), fill=(0, 0, 0, 70))
    _scanlines(overlay, h, w, step=4)
    _draw_frame(draw, w, h)

    # ═════════════════════════════════════════════════════════════════════════
    # TOP BAR — branding + status
    # ═════════════════════════════════════════════════════════════════════════
    tb_x1, tb_y1, tb_x2, tb_y2 = 20, 20, w - 20, 110
    _panel(draw, (tb_x1, tb_y1, tb_x2, tb_y2), BG_MAIN, outline=FRAME_HI)
    _corner_marks(draw, tb_x1, tb_y1, tb_x2, tb_y2, size=14)

    # Left: logo — stacked tightly, room for both words on one line width
    draw.text((42, 32), "LOCKED", font=F_MED, fill=WHITE)
    draw.text((42, 64), "IN", font=F_MED, fill=accent)
    draw.text((42, 92), "ATTENTION MONITOR", font=F_LABEL, fill=DIM)

    # Separator
    draw.line([(240, 32), (240, 100)], fill=FRAME_LO, width=1)

    # Center: status label
    status_text = "● LOCKED IN" if locked else "○ LOCKED OUT"
    stw = _tw(draw, status_text, F_LARGE)
    draw.text(((w - stw) // 2, 40), status_text, font=F_LARGE, fill=accent)

    # Right separator
    draw.line([(w - 200, 32), (w - 200, 100)], fill=FRAME_LO, width=1)

    # Right: clock
    clock_str = time.strftime("%H:%M:%S")
    ctw = _tw(draw, clock_str, F_MED)
    draw.text((tb_x2 - ctw - 18, 38), clock_str, font=F_MED, fill=WHITE)
    draw.text((tb_x2 - ctw - 18, 74), "LOCAL TIME", font=F_LABEL, fill=DIM)

    # ═════════════════════════════════════════════════════════════════════════
    # LEFT PANEL — attention ring + score
    # ═════════════════════════════════════════════════════════════════════════
    lp_x1, lp_y1 = 20, 128
    lp_x2, lp_y2 = 280, 430
    _panel(draw, (lp_x1, lp_y1, lp_x2, lp_y2), BG_PANEL)
    _corner_marks(draw, lp_x1, lp_y1, lp_x2, lp_y2, size=10)

    # Header
    draw.text((lp_x1 + 14, lp_y1 + 14), "ATTENTION", font=F_LABEL, fill=DIM)
    _h_rule(draw, lp_x1 + 10, lp_x2 - 10, lp_y1 + 38)

    # Ring
    ring_cx = (lp_x1 + lp_x2) // 2
    ring_cy = lp_y1 + 130
    ring_r  = 70
    _draw_attention_ring(overlay, ring_cx, ring_cy, ring_r, attention_score / 100, accent, now)

    # Score — vertically centered inside ring
    score_str = f"{attention_score:03d}"
    sw = _tw(draw, score_str, F_LARGE)
    sh = _th(draw, score_str, F_LARGE)
    draw.text((ring_cx - sw // 2, ring_cy - sh // 2 - 8), score_str, font=F_LARGE, fill=accent)
    label = "SCORE"
    lw2 = _tw(draw, label, F_TINY)
    draw.text((ring_cx - lw2 // 2, ring_cy + 22), label, font=F_TINY, fill=DIM)

    # Separator
    _h_rule(draw, lp_x1 + 10, lp_x2 - 10, lp_y1 + 218, color=FRAME_LO)

    # Bar meter
    draw.text((lp_x1 + 14, lp_y1 + 230), "SIGNAL LEVEL", font=F_LABEL, fill=DIM)
    _bar_meter(draw, lp_x1 + 14, lp_y1 + 254, lp_x2 - 14, lp_y1 + 270,
               attention_score / 100, accent, segments=18, gap=2)

    # State chip
    chip_y = lp_y1 + 288
    chip_text = "▶ PHONE OVERRIDE" if phone_active else ("▶ ACTIVE" if locked else "▷ STANDBY")
    chip_col = RED if phone_active else (GREEN if locked else SOFT)
    draw.rectangle((lp_x1 + 12, chip_y, lp_x2 - 12, chip_y + 28), fill=BG_PANEL_3, outline=chip_col, width=1)
    ctw2 = _tw(draw, chip_text, F_SMALL)
    draw.text((lp_x1 + (lp_x2 - lp_x1 - ctw2) // 2, chip_y + 5), chip_text, font=F_SMALL, fill=chip_col)

    # ═════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — session metrics + efficiency (all inside)
    # ═════════════════════════════════════════════════════════════════════════
    rp_x1, rp_y1 = w - 280, 128
    rp_x2, rp_y2 = w - 20, 430
    _panel(draw, (rp_x1, rp_y1, rp_x2, rp_y2), BG_PANEL)
    _corner_marks(draw, rp_x1, rp_y1, rp_x2, rp_y2, size=10)

    # Header
    draw.text((rp_x1 + 14, rp_y1 + 14), "SESSION METRICS", font=F_LABEL, fill=DIM)
    _h_rule(draw, rp_x1 + 10, rp_x2 - 10, rp_y1 + 38)

    # Metric rows
    metrics = [
        ("ELAPSED",  _fmt_time(session_elapsed), WHITE),
        ("FOCUSED",  _fmt_time(locked_in_seconds), WHITE),
        ("STREAK",   _fmt_time(current_streak), accent),
        ("BEST",     _fmt_time(longest_streak), WHITE),
    ]

    my = rp_y1 + 52
    for label, value, col in metrics:
        draw.text((rp_x1 + 16, my + 4), label, font=F_SMALL, fill=DIM)
        vw = _tw(draw, value, F_BODY)
        draw.text((rp_x2 - vw - 16, my), value, font=F_BODY, fill=col)
        _h_rule(draw, rp_x1 + 10, rp_x2 - 10, my + 34)
        my += 44

    # Efficiency block — inside the panel
    eff = int(100 * locked_in_seconds / max(session_elapsed, 1))
    eff_str = f"{eff}%"
    eff_col = GREEN if eff >= 60 else (AMBER if eff >= 30 else RED)

    eff_y = my + 6
    draw.text((rp_x1 + 16, eff_y + 10), "EFFICIENCY", font=F_SMALL, fill=DIM)
    ew = _tw(draw, eff_str, F_LARGE)
    draw.text((rp_x2 - ew - 16, eff_y), eff_str, font=F_LARGE, fill=eff_col)

    # ═════════════════════════════════════════════════════════════════════════
    # BOTTOM PANEL — attention signals
    # ═════════════════════════════════════════════════════════════════════════
    bp_x1, bp_y1 = 20, h - 200
    bp_x2, bp_y2 = w - 20, h - 20
    _panel(draw, (bp_x1, bp_y1, bp_x2, bp_y2), BG_MAIN)
    _corner_marks(draw, bp_x1, bp_y1, bp_x2, bp_y2, size=12)

    # Header with count badge
    draw.text((bp_x1 + 18, bp_y1 + 18), "ATTENTION SIGNALS", font=F_MED, fill=WHITE)
    count_str = f"{len(reasons or [])} ACTIVE"
    ctw3 = _tw(draw, count_str, F_SMALL)
    draw.text((bp_x2 - ctw3 - 30, bp_y1 + 22), count_str, font=F_SMALL, fill=DIM)

    _h_rule(draw, bp_x1 + 10, bp_x2 - 10, bp_y1 + 54, color=FRAME_HI)

    shown = (reasons or ["No signals"])[:4]
    rx = bp_x1 + 18
    ry = bp_y1 + 70
    col_w = (bp_x2 - bp_x1 - 36) // 2

    for i, reason in enumerate(shown):
        col_x = rx + (col_w + 16) * (i // 2)
        row_y = ry + (i % 2) * 46

        # Index badge — brighter, larger
        idx_txt = f"{i + 1:02d}"
        draw.rectangle((col_x, row_y, col_x + 34, row_y + 32),
                        fill=accent, outline=None)
        itw = _tw(draw, idx_txt, F_SMALL)
        draw.text((col_x + (34 - itw) // 2, row_y + 7), idx_txt, font=F_SMALL, fill=(0, 0, 0, 255))

        draw.text((col_x + 46, row_y + 5), reason.upper(), font=F_BODY, fill=WHITE)

    # Accent strip (right edge)
    strip_w = 6
    draw.rectangle(
        (bp_x2 - strip_w - 4, bp_y1 + 10, bp_x2 - 4, bp_y2 - 10),
        fill=accent
    )

    # ═════════════════════════════════════════════════════════════════════════
    # LOCKED OUT — ghost warning mark
    # ═════════════════════════════════════════════════════════════════════════
    if not locked:
        pulsed_alpha = int(35 + 20 * math.sin(now * 2.2))
        mark = "!"
        bb  = draw.textbbox((0, 0), mark, font=F_HUGE)
        tw2 = bb[2] - bb[0]
        th2 = bb[3] - bb[1]
        draw.text(
            ((w - tw2) // 2, (h - th2) // 2 - 30),
            mark, font=F_HUGE,
            fill=(240, 60, 60, pulsed_alpha),
        )

    # ═════════════════════════════════════════════════════════════════════════
    # PHONE BOX overlay
    # ═════════════════════════════════════════════════════════════════════════
    if phone_active and phone_box is not None:
        x1, y1, x2, y2 = phone_box
        pad = 10
        # Animated border — pulse
        blink_col = (*RED[:3], int(180 + 70 * math.sin(now * 6)))
        draw.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), outline=blink_col, width=3)
        _corner_marks(draw, x1 - pad, y1 - pad, x2 + pad, y2 + pad, size=12, color=RED, width=2)

        # Tag
        tag_x1 = max(22, x1 - pad)
        tag_y1 = max(22, y1 - pad - 34)
        tag_x2 = tag_x1 + 168
        tag_y2 = tag_y1 + 28
        draw.rectangle((tag_x1, tag_y1, tag_x2, tag_y2), fill=(24, 6, 6, 245), outline=RED, width=1)
        draw.text((tag_x1 + 10, tag_y1 + 6), "PHONE DETECTED", font=F_SMALL, fill=RED)

    # ── composite ─────────────────────────────────────────────────────────────
    final = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)