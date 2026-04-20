"""
ui_debug.py — Debug overlay for Locked In.

Same HUD aesthetic as ui_app.py; adds a dense diagnostics pane
and raw signal readouts.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── palette (matches ui_app.py) ───────────────────────────────────────────────
WHITE      = (245, 248, 252, 255)
SOFT       = (185, 193, 205, 255)
DIM        = (130, 138, 152, 255)

GREEN      = (32,  220, 120, 255)
RED        = (240, 60,  60,  255)
AMBER      = (255, 185, 40,  255)
CYAN       = (40,  210, 230, 255)
PURPLE     = (160, 110, 255, 255)
YELLOW     = (255, 200, 50,  255)

BG_MAIN    = (4,   5,   9,   235)
BG_PANEL   = (8,   9,   14,  230)
BG_PANEL_2 = (11,  13,  19,  225)
BG_PANEL_3 = (14,  16,  24,  218)
FRAME_HI   = (255, 255, 255, 55)
FRAME_LO   = (255, 255, 255, 18)
SCANLINE   = (0,   0,   0,   30)


def _font(size: int, mono: bool = True) -> ImageFont.ImageFont:
    mono_candidates = [
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/Library/Fonts/Courier New Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    sans = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in (mono_candidates if mono else sans):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


F_STATUS = _font(46)
F_MED    = _font(24)
F_BODY   = _font(20)
F_SMALL  = _font(17)
F_TINY   = _font(14)
F_HUGE   = _font(110, mono=False)


# ── shared helpers ────────────────────────────────────────────────────────────

def _tw(draw, text, font) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def _panel(draw, xy, fill, outline=FRAME_HI, width=1):
    draw.rectangle(xy, fill=fill, outline=outline, width=width)


def _corner_marks(draw, x1, y1, x2, y2, size=12, color=FRAME_HI, width=2):
    lines = [
        (x1, y1,        x1 + size, y1),
        (x1, y1,        x1,        y1 + size),
        (x2 - size, y1, x2,        y1),
        (x2,        y1, x2,        y1 + size),
        (x1, y2 - size, x1,        y2),
        (x1, y2,        x1 + size, y2),
        (x2 - size, y2, x2,        y2),
        (x2, y2 - size, x2,        y2),
    ]
    for ax, ay, bx, by in lines:
        draw.line([(ax, ay), (bx, by)], fill=color, width=width)


def _scanlines(overlay, h, w):
    draw = ImageDraw.Draw(overlay)
    for y in range(0, h, 4):
        draw.line([(0, y), (w, y)], fill=SCANLINE, width=1)


def _h_rule(draw, x1, x2, y, color=FRAME_LO):
    draw.line([(x1, y), (x2, y)], fill=color, width=1)


def _chip(draw, x, y, text, color, font=None):
    font = font or F_SMALL
    bb = draw.textbbox((0, 0), text, font=font)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    pad_x, pad_y = 10, 5
    w = tw + pad_x * 2
    h = th + pad_y * 2
    draw.rectangle((x, y, x + w, y + h), fill=BG_PANEL_3, outline=color, width=1)
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=color)
    return w, h


_MODE_COLOR = {
    "FACING":   CYAN,
    "READING":  YELLOW,
    "WRITING":  PURPLE,
    "OFF_TASK": RED,
    "PHONE":    RED,
}


# ── main render ───────────────────────────────────────────────────────────────

def render_debug_ui(
    frame,
    status: str,
    internal_mode: str,
    reasons: list[str],
    debug_lines: list[str],
    attention_score: int = 50,
    phone_active: bool = False,
    phone_box=None,
):
    now     = time.time()
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base    = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    w, h    = base.size
    locked  = status == "LOCKED IN"
    accent  = GREEN if locked else RED

    draw.rectangle((0, 0, w, h), fill=(0, 0, 0, 70))
    _scanlines(overlay, h, w)

    # Outer frame
    m = 12
    draw.rectangle((m, m, w - m, h - m), outline=FRAME_HI, width=1)

    # ═════════════════════════════════════════════════════════════════════════
    # TOP-RIGHT STATUS CARD
    # ═════════════════════════════════════════════════════════════════════════
    tc_w = 400
    tc_x1, tc_y1 = w - tc_w - 20, 20
    tc_x2, tc_y2 = w - 20, 112
    _panel(draw, (tc_x1, tc_y1, tc_x2, tc_y2), BG_MAIN)
    _corner_marks(draw, tc_x1, tc_y1, tc_x2, tc_y2, size=12)

    stw = _tw(draw, status, F_STATUS)
    draw.text((tc_x2 - stw - 16, tc_y1 + 14), status, font=F_STATUS, fill=accent)
    draw.text((tc_x1 + 14, tc_y1 + 14), "DBG", font=F_TINY, fill=DIM)
    draw.text((tc_x1 + 14, tc_y1 + 32), f"SCR {attention_score:03d}", font=F_SMALL, fill=SOFT)
    clock = time.strftime("%H:%M:%S")
    draw.text((tc_x1 + 14, tc_y1 + 58), clock, font=F_SMALL, fill=DIM)

    # ═════════════════════════════════════════════════════════════════════════
    # LEFT PANEL — mode + reasons + flags
    # ═════════════════════════════════════════════════════════════════════════
    lp_x1, lp_y1, lp_x2, lp_y2 = 20, 20, 320, h - 20
    _panel(draw, (lp_x1, lp_y1, lp_x2, lp_y2), BG_PANEL_2)
    _corner_marks(draw, lp_x1, lp_y1, lp_x2, lp_y2)

    draw.text((lp_x1 + 14, lp_y1 + 14), "CLASSIFIER", font=F_TINY, fill=DIM)
    _h_rule(draw, lp_x1 + 8, lp_x2 - 8, lp_y1 + 36)

    # Mode chip
    mode_col = _MODE_COLOR.get(internal_mode, WHITE)
    _, mh = _chip(draw, lp_x1 + 14, lp_y1 + 46, f"▶ {internal_mode}", mode_col, font=F_MED)

    yy = lp_y1 + 46 + mh + 18
    draw.text((lp_x1 + 14, yy), "REASONS", font=F_TINY, fill=DIM)
    _h_rule(draw, lp_x1 + 8, lp_x2 - 8, yy + 18)
    yy += 26

    for r in (reasons or ["None"])[:5]:
        r_col = RED if any(k in r.upper() for k in ("PHONE", "LOOKING", "TALKING")) else (GREEN if locked else SOFT)
        _, rh = _chip(draw, lp_x1 + 14, yy, r.upper(), r_col)
        yy += rh + 8

    yy += 12
    draw.text((lp_x1 + 14, yy), "FLAGS", font=F_TINY, fill=DIM)
    _h_rule(draw, lp_x1 + 8, lp_x2 - 8, yy + 18)
    yy += 26

    flags = [
        ("STATUS",  status,                       accent),
        ("MODE",    internal_mode,                mode_col),
        ("PHONE",   "OVERRIDE" if phone_active else "CLEAR",
                    RED if phone_active else GREEN),
    ]
    for label, value, col in flags:
        draw.text((lp_x1 + 14, yy), f"{label:<10}", font=F_SMALL, fill=DIM)
        draw.text((lp_x1 + 14 + 90, yy), value, font=F_SMALL, fill=col)
        yy += 24

    # ═════════════════════════════════════════════════════════════════════════
    # BOTTOM DIAGNOSTICS PANEL
    # ═════════════════════════════════════════════════════════════════════════
    dp_x1, dp_y1 = 340, h - 210
    dp_x2, dp_y2 = w - 20, h - 20
    _panel(draw, (dp_x1, dp_y1, dp_x2, dp_y2), BG_PANEL_3)
    _corner_marks(draw, dp_x1, dp_y1, dp_x2, dp_y2)

    draw.text((dp_x1 + 16, dp_y1 + 12), "DIAGNOSTICS", font=F_TINY, fill=DIM)
    _h_rule(draw, dp_x1 + 8, dp_x2 - 8, dp_y1 + 34)

    dy = dp_y1 + 44
    for line in debug_lines:
        # Split "key: value" for two-tone colouring
        if ":" in line:
            key, _, val = line.partition(":")
            kw = _tw(draw, key + ":", F_SMALL)
            draw.text((dp_x1 + 16, dy), key + ":", font=F_SMALL, fill=DIM)
            draw.text((dp_x1 + 16 + kw + 4, dy), val.strip(), font=F_SMALL, fill=WHITE)
        else:
            draw.text((dp_x1 + 16, dy), line, font=F_SMALL, fill=SOFT)
        dy += 22

    # ═════════════════════════════════════════════════════════════════════════
    # LOCKED-OUT ghost mark
    # ═════════════════════════════════════════════════════════════════════════
    if not locked:
        alpha = int(30 + 18 * math.sin(now * 2.1))
        mark = "!"
        bb   = draw.textbbox((0, 0), mark, font=F_HUGE)
        tw   = bb[2] - bb[0]
        th   = bb[3] - bb[1]
        draw.text(((w - tw) // 2, (h - th) // 2 - 40), mark, font=F_HUGE,
                  fill=(240, 60, 60, alpha))

    # ═════════════════════════════════════════════════════════════════════════
    # PHONE BOX
    # ═════════════════════════════════════════════════════════════════════════
    if phone_active and phone_box is not None:
        x1, y1, x2, y2 = phone_box
        pad = 10
        blink = (*RED[:3], int(175 + 75 * math.sin(now * 6)))
        draw.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), outline=blink, width=3)
        _corner_marks(draw, x1 - pad, y1 - pad, x2 + pad, y2 + pad, size=12, color=RED)

        tag_x1 = max(22, x1 - pad)
        tag_y1 = max(22, y1 - pad - 32)
        draw.rectangle((tag_x1, tag_y1, tag_x1 + 164, tag_y1 + 26),
                        fill=(24, 6, 6, 245), outline=RED, width=1)
        draw.text((tag_x1 + 8, tag_y1 + 5), "PHONE DETECTED", font=F_SMALL, fill=RED)

    final = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)