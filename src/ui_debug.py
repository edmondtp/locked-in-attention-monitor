from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

WHITE = (245, 247, 250, 255)
MUTED = (205, 212, 220, 255)

GREEN = (39, 220, 124, 255)
RED = (255, 71, 87, 255)
YELLOW = (255, 193, 7, 255)
BLUE = (86, 156, 214, 255)
PURPLE = (163, 112, 255, 255)

BG_PANEL = (16, 18, 24, 224)
BG_PANEL_SOFT = (20, 23, 30, 205)
BG_CHIP_DARK = (36, 41, 54, 240)
BORDER = (255, 255, 255, 40)

SIDE_PANEL_WIDTH = 320


def get_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_STATUS = get_font(52)
FONT_TITLE = get_font(24)
FONT_BODY = get_font(19)
FONT_SMALL = get_font(18)
FONT_ALERT = get_font(220)


def draw_chip(draw, x, y, text, color):
    bbox = draw.textbbox((0, 0), text, font=FONT_BODY)
    w = bbox[2] + 26
    h = bbox[3] + 14
    draw.rounded_rectangle((x, y, x + w, y + h), radius=15, fill=BG_CHIP_DARK, outline=(255, 255, 255, 22))
    draw.text((x + 13, y + 6), text, font=FONT_BODY, fill=color)
    return w, h


def render_debug_ui(frame, status, internal_mode, reasons, debug_lines, phone_active=False, phone_box=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = base.size
    locked = status == "LOCKED IN"
    accent = GREEN if locked else RED

    draw.rounded_rectangle((18, 18, w - 18, 108), radius=28, fill=BG_PANEL, outline=BORDER, width=2)
    draw.text((40, 36), status, font=FONT_STATUS, fill=accent)

    if not locked:
        alert = "!"
        bbox = draw.textbbox((0, 0), alert, font=FONT_ALERT)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, (h - th) // 2 - 50), alert, font=FONT_ALERT, fill=(255, 71, 87, 70))

    draw.rounded_rectangle((18, 130, SIDE_PANEL_WIDTH, h - 24), radius=24, fill=BG_PANEL_SOFT, outline=BORDER, width=2)
    draw.text((38, 150), "Mode", font=FONT_TITLE, fill=WHITE)

    color_map = {
        "FACING": BLUE,
        "READING": YELLOW,
        "WRITING": PURPLE,
        "OFF_TASK": RED,
        "PHONE": RED,
    }

    chip_y = 190
    draw_chip(draw, 38, chip_y, internal_mode, color_map.get(internal_mode, WHITE))

    draw.text((38, 245), "Signals", font=FONT_TITLE, fill=WHITE)
    y = 285
    for r in reasons[:5]:
        _, hh = draw_chip(draw, 38, y, r, WHITE if locked else RED if "away" in r.lower() or "talk" in r.lower() or "phone" in r.lower() else BLUE)
        y += hh + 10

    dbg_x1 = SIDE_PANEL_WIDTH + 24
    dbg_y1 = h - 170
    draw.rounded_rectangle((dbg_x1, dbg_y1, w - 18, h - 20), radius=24, fill=BG_PANEL_SOFT, outline=BORDER, width=2)
    draw.text((dbg_x1 + 18, dbg_y1 + 16), "Debug", font=FONT_TITLE, fill=WHITE)

    yy = dbg_y1 + 50
    for line in debug_lines:
        draw.text((dbg_x1 + 18, yy), line, font=FONT_SMALL, fill=MUTED)
        yy += 24

    if phone_active and phone_box is not None:
        x1, y1, x2, y2 = phone_box
        draw.rounded_rectangle((x1, y1, x2, y2), radius=16, outline=RED, width=4)

    final = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)