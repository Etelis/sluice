# SPDX-License-Identifier: Apache-2.0
"""Generate Sluice's icon + social-preview image (Pillow only).

The mark: three chevrons — a stream — with the last one amber, the
router-selected expert arriving in the GPU cache. Palette nods to vLLM's teal,
paired with indigo. Original artwork, not the vLLM logo.

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
    """Three chevrons (a stream); the last is amber — the selected expert."""
    s = scale
    span = int(150 * s)
    gap = int(168 * s)
    half_h = int(150 * s)
    w = max(6, int(76 * s))
    x0 = cx - int(250 * s)
    for i, (color, alpha) in enumerate(
        [(WHITE, 105), (WHITE, 185), (AMBER, 255)]
    ):
        x = x0 + i * gap
        draw.line(
            [(x, cy - half_h), (x + span, cy), (x, cy + half_h)],
            fill=_wa(color, alpha),
            width=w,
            joint="curve",
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
    draw_glyph(ImageDraw.Draw(img), size // 2, size // 2, scale=size / 1024)
    img.save(path)
    print("wrote", path)


def make_social(path, w=1280, h=640):
    img = _vgrad(w, h, TEAL, INDIGO).convert("RGBA")
    draw = ImageDraw.Draw(img)
    draw_glyph(draw, int(w * 0.24), h // 2, scale=0.74)
    tx = int(w * 0.45)
    draw.text((tx, h // 2 - 132), "Sluice", font=_font(150), fill=WHITE)
    draw.text((tx, h // 2 + 42), "Routing-aware MoE expert", font=_font(46),
              fill=_wa(WHITE, 235))
    draw.text((tx, h // 2 + 102), "offloading for vLLM", font=_font(46),
              fill=_wa(WHITE, 235))
    img.convert("RGB").save(path)
    print("wrote", path)


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    make_icon(os.path.join(here, "sluice-icon.png"))
    make_social(os.path.join(here, "sluice-social.png"))
