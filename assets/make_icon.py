# SPDX-License-Identifier: Apache-2.0
"""Generate Sluice's icon + social-preview image (Pillow only).

The mark: a pool of experts (dim squares) metered through a central GATE into a
small GPU cache (bright squares in a chip) — the Sluice concept. Palette nods to
vLLM's teal, paired with indigo; amber marks the router-selected/resident
experts. Original artwork — not the vLLM logo.

    python assets/make_icon.py
"""

from PIL import Image, ImageDraw, ImageFont

TEAL = (13, 184, 171)
INDIGO = (67, 56, 202)
AMBER = (245, 169, 38)
WHITE = (255, 255, 255)


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _vgrad(w, h, c0, c1):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        d.line([(0, y), (w, y)], fill=_lerp(c0, c1, y / (h - 1)))
    return img


def _round(img, radius):
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, img.size[0] - 1, img.size[1] - 1], radius=radius, fill=255
    )
    img = img.convert("RGBA")
    img.putalpha(mask)
    return img


def _wa(color, alpha):
    return (color[0], color[1], color[2], alpha)


def draw_glyph(draw, cx, cy, scale=1.0):
    """Draw the pool -> gate -> GPU-cache mark centered at (cx, cy)."""
    s = scale
    lanes = [cy - int(150 * s), cy, cy + int(150 * s)]
    lane_h = int(14 * s)
    x_pool = cx - int(300 * s)
    x_gate = cx - int(40 * s)
    x_chip0 = cx + int(90 * s)
    x_chip1 = cx + int(300 * s)
    sq = int(46 * s)

    # flow lanes
    for y in lanes:
        draw.rounded_rectangle(
            [x_pool, y - lane_h // 2, x_chip1, y + lane_h // 2],
            radius=lane_h // 2,
            fill=_wa(WHITE, 55),
        )

    # expert pool (dim squares queued on the left)
    for col in range(2):
        for y in lanes:
            x = x_pool - int(2 * s) + col * (sq + int(14 * s))
            draw.rounded_rectangle(
                [x, y - sq // 2, x + sq, y + sq // 2],
                radius=int(10 * s),
                fill=_wa(WHITE, 105),
            )

    # the gate: a vertical bar with three bright slots where the lanes cross
    gw = int(26 * s)
    draw.rounded_rectangle(
        [x_gate - gw // 2, cy - int(232 * s), x_gate + gw // 2, cy + int(232 * s)],
        radius=int(12 * s),
        fill=_wa(WHITE, 235),
    )
    slot = int(30 * s)
    for y in lanes:
        draw.rounded_rectangle(
            [x_gate - slot // 2, y - slot // 2, x_gate + slot // 2, y + slot // 2],
            radius=int(7 * s),
            fill=_wa(AMBER, 255),
        )

    # arrows from gate into the chip
    for y in lanes:
        ax = x_gate + int(34 * s)
        draw.polygon(
            [
                (ax, y - int(16 * s)),
                (ax + int(26 * s), y),
                (ax, y + int(16 * s)),
            ],
            fill=_wa(WHITE, 180),
        )

    # GPU cache: a chip outline holding the resident (amber) experts
    chip = [x_chip0, cy - int(210 * s), x_chip1, cy + int(210 * s)]
    draw.rounded_rectangle(chip, radius=int(28 * s), outline=_wa(WHITE, 240),
                           width=max(1, int(11 * s)))
    # chip pins
    pin = int(30 * s)
    for y in lanes:
        draw.rounded_rectangle(
            [x_chip1 + int(2 * s), y - int(5 * s), x_chip1 + pin, y + int(5 * s)],
            radius=int(5 * s), fill=_wa(WHITE, 200),
        )
    # resident experts
    rx = (x_chip0 + x_chip1) // 2
    for y in lanes:
        draw.rounded_rectangle(
            [rx - sq // 2, y - sq // 2, rx + sq // 2, y + sq // 2],
            radius=int(10 * s), fill=_wa(AMBER, 255),
        )


def _font(size):
    for path in (
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_icon(path, size=1024):
    img = _round(_vgrad(size, size, TEAL, INDIGO), radius=int(size * 0.22))
    draw = ImageDraw.Draw(img)
    draw_glyph(draw, size // 2, size // 2, scale=size / 1024)
    img.save(path)
    print("wrote", path)


def make_social(path, w=1280, h=640):
    img = _vgrad(w, h, TEAL, INDIGO).convert("RGBA")
    draw = ImageDraw.Draw(img)
    draw_glyph(draw, int(w * 0.27), h // 2, scale=0.62)
    tx = int(w * 0.50)
    draw.text((tx, h // 2 - 130), "Sluice", font=_font(150), fill=WHITE)
    draw.text((tx, h // 2 + 40), "Routing-aware MoE expert", font=_font(46),
              fill=_wa(WHITE, 230))
    draw.text((tx, h // 2 + 100), "offloading for vLLM", font=_font(46),
              fill=_wa(WHITE, 230))
    img.convert("RGB").save(path)
    print("wrote", path)


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    make_icon(os.path.join(here, "sluice-icon.png"))
    make_social(os.path.join(here, "sluice-social.png"))
