# src/ui.py
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Neutral palette (less neon)
WHITE = (240, 242, 245, 255)
TEXT_PRIMARY = (220, 225, 232, 255)
TEXT_SECONDARY = (160, 168, 180, 255)

GREEN = (80, 200, 120, 255)
RED = (220, 90, 90, 255)
BLUE = (120, 160, 220, 255)
YELLOW = (220, 180, 80, 255)
PURPLE = (160, 120, 220, 255)

BG_MAIN = (18, 20, 26, 255)
BG_PANEL = (26, 30, 38, 230)
BG_CHIP = (40, 46, 58, 255)
BORDER = (255, 255, 255, 25)

SIDE_PANEL_WIDTH = 320

def get_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except:
        return ImageFont.load_default()

FONT_BIG = get_font(44)
FONT_TITLE = get_font(22)
FONT_BODY = get_font(18)
FONT_SMALL = get_font(15)

def draw_box(draw, x1, y1, x2, y2):
    draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=BG_PANEL, outline=BORDER)

def draw_chip(draw, x, y, text, color):
    bbox = draw.textbbox((0,0), text, font=FONT_BODY)
    w = bbox[2] + 26
    h = bbox[3] + 12
    draw.rounded_rectangle((x, y, x+w, y+h), radius=14, fill=BG_CHIP)
    draw.text((x+13, y+6), text, font=FONT_BODY, fill=color)
    return w, h

def render_ui(
    frame,
    status,
    internal_mode,
    vote_score,
    vote_cap,
    fps,
    phone_active,
    phone_conf,
    phone_box,
    reasons,
    debug_lines,
):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    w, h = base.size

    # Top bar
    draw_box(draw, 20, 20, w-20, 95)

    status_color = GREEN if status == "LOCKED IN" else RED
    draw.text((40, 35), status, font=FONT_BIG, fill=status_color)

    draw.text((w-140, 45), f"{fps:.0f} FPS", font=FONT_BODY, fill=TEXT_SECONDARY)

    # Left panel
    draw_box(draw, 20, 110, SIDE_PANEL_WIDTH, h-20)

    draw.text((40, 130), "Mode", font=FONT_TITLE, fill=TEXT_PRIMARY)

    color_map = {
        "FACING": BLUE,
        "READING": YELLOW,
        "WRITING": PURPLE,
        "OFF_TASK": RED,
        "PHONE": RED
    }

    draw_chip(draw, 40, 165, internal_mode, color_map.get(internal_mode, TEXT_PRIMARY))

    if phone_active:
        draw_chip(draw, 40, 210, "PHONE DETECTED", RED)

    draw.text((40, 260), "Signals", font=FONT_TITLE, fill=TEXT_PRIMARY)

    y = 300
    for reason, color in reasons[:5]:
        _, h_chip = draw_chip(draw, 40, y, reason, color)
        y += h_chip + 10

    # Confidence bar
    draw.text((40, y+10), "Confidence", font=FONT_TITLE, fill=TEXT_PRIMARY)

    bar_x1 = 40
    bar_y1 = y + 50
    bar_x2 = 280
    bar_y2 = bar_y1 + 16

    progress = (vote_score + vote_cap) / (2 * vote_cap)
    fill_x = int(bar_x1 + (bar_x2-bar_x1) * progress)

    draw.rounded_rectangle((bar_x1, bar_y1, bar_x2, bar_y2), radius=8, fill=(50,55,70))
    draw.rounded_rectangle((bar_x1, bar_y1, fill_x, bar_y2), radius=8,
                           fill=GREEN if status=="LOCKED IN" else RED)

    draw.text((40, bar_y2+10), f"score {vote_score:+d}", font=FONT_SMALL, fill=TEXT_SECONDARY)

    # Debug panel (bottom right)
    dbg_x1 = SIDE_PANEL_WIDTH + 40
    dbg_y1 = h - 140

    draw_box(draw, dbg_x1, dbg_y1, w-20, h-20)

    draw.text((dbg_x1+20, dbg_y1+15), "Debug", font=FONT_TITLE, fill=TEXT_PRIMARY)

    yy = dbg_y1 + 45
    for line in debug_lines:
        draw.text((dbg_x1+20, yy), line, font=FONT_SMALL, fill=TEXT_SECONDARY)
        yy += 18

    # Phone box overlay
    if phone_active and phone_box:
        x1,y1,x2,y2 = phone_box
        draw.rounded_rectangle((x1,y1,x2,y2), radius=14, outline=RED, width=3)

    final = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)