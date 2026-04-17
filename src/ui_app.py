from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

WHITE = (242, 244, 247, 255)
TEXT = (220, 224, 230, 255)
TEXT_MUTED = (165, 172, 182, 255)

GREEN = (74, 199, 122, 255)
RED = (220, 90, 90, 255)
AMBER = (220, 180, 80, 255)

BG_PANEL = (20, 24, 30, 230)
BG_SOFT = (28, 34, 42, 220)
BORDER = (255, 255, 255, 22)


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


FONT_BIG = get_font(54)
FONT_MED = get_font(24)
FONT_BODY = get_font(20)
FONT_SMALL = get_font(17)
FONT_ALERT = get_font(220)


def render_app_ui(frame, status, reasons, phone_active=False, phone_box=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = base.size
    locked = status == "LOCKED IN"
    accent = GREEN if locked else RED

    # Top banner
    draw.rounded_rectangle((24, 24, w - 24, 110), radius=28, fill=BG_PANEL, outline=BORDER, width=2)
    draw.text((44, 36), status, font=FONT_BIG, fill=accent)

    # Center alert if not locked in
    if not locked:
        alert = "!"
        bbox = draw.textbbox((0, 0), alert, font=FONT_ALERT)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, (h - th) // 2 - 50), alert, font=FONT_ALERT, fill=(220, 90, 90, 70))

    # Bottom reason card
    card_h = 120
    draw.rounded_rectangle((30, h - card_h - 24, w - 30, h - 24), radius=24, fill=BG_SOFT, outline=BORDER, width=2)

    title = "Status" if locked else "Why you're not locked in"
    draw.text((52, h - card_h), title, font=FONT_MED, fill=WHITE)

    y = h - card_h + 38
    for reason in reasons[:3]:
        draw.text((52, y), f"• {reason}", font=FONT_BODY, fill=TEXT)
        y += 28

    if phone_active and phone_box is not None:
        x1, y1, x2, y2 = phone_box
        draw.rounded_rectangle((x1, y1, x2, y2), radius=16, outline=RED, width=4)

    final = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(final), cv2.COLOR_RGB2BGR)