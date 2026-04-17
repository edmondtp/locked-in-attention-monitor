from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# --- Colors & Styles ---
def get_rgba(hex_c, alpha=255):
    hex_c = hex_c.lstrip('#')
    return tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)

COLORS = {
    "NEON_GREEN": get_rgba("#22E38E"),
    "SOFT_GREEN": get_rgba("#22E38E", 60),
    "NEON_RED": get_rgba("#FF4D6D"),
    "SOFT_RED": get_rgba("#FF4D6D", 70),
    "NEON_BLUE": get_rgba("#58A6FF"),
    "BG_DARK": (12, 14, 20, 220),
    "BORDER": (255, 255, 255, 30)
}

def load_fonts():
    try:
        # Using specific Mac/Windows paths for high-quality fonts
        f_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        return {
            "status": ImageFont.truetype(f_path, 52),
            "title": ImageFont.truetype(f_path, 24),
            "chip": ImageFont.truetype(f_path, 18),
            "debug": ImageFont.truetype(f_path, 16)
        }
    except:
        return {k: ImageFont.load_default() for k in ["status", "title", "chip", "debug"]}

FONTS = load_fonts()

def draw_hud(frame, status, mode, vote_score, phone_active, reasons, debug_lines):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = pil_img.size

    # 1. Header Banner
    locked = status == "LOCKED IN"
    main_color = COLORS["NEON_GREEN"] if locked else COLORS["NEON_RED"]
    draw.rounded_rectangle([20, 20, w-20, 100], radius=25, fill=COLORS["BG_DARK"], outline=COLORS["BORDER"], width=2)
    draw.rounded_rectangle([20, 20, w-20, 100], radius=25, fill=(main_color[0], main_color[1], main_color[2], 50))
    
    # Status Text + Glow
    txt_pos = (50, 32)
    for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
        draw.text((txt_pos[0]+dx, txt_pos[1]+dy), status, font=FONTS["status"], fill=(main_color[0], main_color[1], main_color[2], 80))
    draw.text(txt_pos, status, font=FONTS["status"], fill=main_color)

    # 2. Side Panel
    draw.rounded_rectangle([20, 120, 280, h-20], radius=20, fill=COLORS["BG_DARK"], outline=COLORS["BORDER"])
    draw.text((40, 140), "SYSTEM SIGNALS", font=FONTS["title"], fill=(255, 255, 255, 200))

    y_off = 180
    for text, color in reasons:
        draw.rounded_rectangle([40, y_off, 260, y_off+32], radius=10, fill=(30, 35, 45, 255))
        draw.text((55, y_off+7), text, font=FONTS["chip"], fill=color)
        y_off += 42

    # 3. Confidence Bar
    bar_y = h - 60
    progress = (vote_score + 16) / 32
    draw.rounded_rectangle([40, bar_y, 260, bar_y+15], radius=5, fill=(50, 50, 60))
    bar_w = int(220 * progress)
    draw.rounded_rectangle([40, bar_y, 40+bar_w, bar_y+15], radius=5, fill=main_color)

    # 4. Debug Panel (Bottom Right)
    draw.rounded_rectangle([w-350, h-160, w-20, h-20], radius=20, fill=(10, 10, 15, 200), outline=COLORS["BORDER"])
    for i, line in enumerate(debug_lines):
        draw.text((w-330, h-145 + (i*22)), line, font=FONTS["debug"], fill=(180, 180, 180))

    return cv2.cvtColor(np.array(Image.alpha_composite(pil_img, overlay)), cv2.COLOR_RGBA2BGR)