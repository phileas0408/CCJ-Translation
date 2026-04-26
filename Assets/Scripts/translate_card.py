#!/usr/bin/env python3
"""
translate_cards.py — Batch card translator

Usage:
    python translate_cards.py <input_folder> <output_folder>

    Processes every PNG/JPG in <input_folder>, overlays the template,
    translates the Japanese title via Claude Vision, and saves to <output_folder>.

Requirements:
    pip install pillow anthropic

Environment:
    ANTHROPIC_API_KEY  – your Anthropic API key (required for translation)
    TEMPLATE_PATH      – path to template PNG  (default: item_obtain_template.png)
    FONT_PATH          – path to TCG2SAB.ttf   (default: TCG2SAB.ttf)
"""

import base64
import json
import os
import sys
from pathlib import Path

# Load .env from the project root (two levels up from Assets/Scripts/)
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
TEMPLATE_PATH = Path(os.environ.get("TEMPLATE_PATH", "item_obtain_template.png"))
FONT_PATH     = Path(os.environ.get("FONT_PATH",     "TCG2SAB.ttf"))

FONT_SIZE_MAX = 42
FONT_SIZE_MIN = 16
BAND_TOP      = 16
BAND_BOTTOM   = 79
PADDING_X     = 40
SHADOW_OFFSET = 4

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ── Translation ───────────────────────────────────────────────────────────────
def get_anthropic_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("Please install the anthropic library:  pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Run:  export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    return anthropic.Anthropic(api_key=api_key)


def translate_with_claude(client, card_path: Path) -> str:
    with open(card_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    # Detect media type
    ext = card_path.suffix.lower()
    media_type = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Translate ALL Japanese text visible in this image to English. "
                        "Return ONLY a raw JSON object (no markdown, no backticks) like: "
                        '{"title": "...", "label": "..."} '
                        "using whatever keys best describe each piece of text. "
                        "Keep translations concise and accurate."
                    ),
                },
            ],
        }],
    )

    raw = response.content[0].text.strip()
    try:
        data = json.loads(raw)
        title = data.get("title") or next(iter(data.values()), raw)
    except json.JSONDecodeError:
        title = raw

    return str(title)


# ── Image processing ──────────────────────────────────────────────────────────
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(str(FONT_PATH), size)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def fit_font(draw: ImageDraw.Draw, text: str, max_width: int) -> ImageFont.FreeTypeFont:
    for size in range(FONT_SIZE_MAX, FONT_SIZE_MIN - 1, -1):
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
    return font


def draw_layered_text(draw: ImageDraw.Draw, text: str,
                      x: int, y: int, font: ImageFont.FreeTypeFont) -> None:
    bw = 4

    # Shadow (stroked, offset)
    for dx in range(-bw, bw + 1):
        for dy in range(-bw, bw + 1):
            if dx * dx + dy * dy >= (bw - 2) ** 2:
                draw.text((x + dx + SHADOW_OFFSET, y + dy + SHADOW_OFFSET),
                          text, font=font, fill="#000000")
    draw.text((x + SHADOW_OFFSET, y + SHADOW_OFFSET), text, font=font, fill="#000000")

    # Black outer stroke
    for dx in range(-bw, bw + 1):
        for dy in range(-bw, bw + 1):
            if dx * dx + dy * dy >= (bw - 2) ** 2:
                draw.text((x + dx, y + dy), text, font=font, fill="black")

    # Yellow inner stroke
    yw = 2
    for dx in range(-yw, yw + 1):
        for dy in range(-yw, yw + 1):
            if dx * dx + dy * dy >= (yw - 1) ** 2:
                draw.text((x + dx, y + dy), text, font=font, fill="#FFE600")

    # White fill
    draw.text((x, y), text, font=font, fill="white")


_template_cache: dict[tuple[int, int], Image.Image] = {}

def process_card(card_path: Path, output_path: Path, template: Image.Image, title: str) -> None:
    card = Image.open(card_path).convert("RGBA")

    if card.size not in _template_cache:
        _template_cache[card.size] = (
            template if template.size == card.size
            else template.resize(card.size, Image.LANCZOS)
        )
    tmpl = _template_cache[card.size]

    composite = Image.alpha_composite(card, tmpl)
    draw = ImageDraw.Draw(composite)
    W, _H = composite.size
    max_w = W - PADDING_X * 2

    font = fit_font(draw, title, max_w)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    band_center = (BAND_TOP + BAND_BOTTOM) // 2
    tx = (W - tw) // 2 - bbox[0] - SHADOW_OFFSET // 2
    ty = band_center - th // 2 - bbox[1] - SHADOW_OFFSET // 2

    draw_layered_text(draw, title, tx, ty, font)
    composite.save(output_path, "PNG")


# ── Batch runner ──────────────────────────────────────────────────────────────
def batch_process(input_dir: Path, output_dir: Path) -> None:
    # Validate inputs
    if not input_dir.is_dir():
        sys.exit(f"Input folder not found: {input_dir}")
    if not TEMPLATE_PATH.exists():
        sys.exit(f"Template not found: {TEMPLATE_PATH}")
    if not FONT_PATH.exists():
        print(f"Warning: font not found at {FONT_PATH}, will use system default.")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        sys.exit(f"No supported images found in {input_dir}")

    print(f"Found {len(images)} image(s) in '{input_dir}'")
    print(f"Output folder: '{output_dir}'\n")

    # Load template once
    template = Image.open(TEMPLATE_PATH).convert("RGBA")

    # Init Claude client once
    client = get_anthropic_client()

    ok, failed = 0, []

    for i, card_path in enumerate(images, 1):
        out_path = output_dir / (card_path.stem + card_path.suffix)
        print(f"[{i}/{len(images)}] {card_path.name}")

        if out_path.exists():
            print(f"  → Skipped (already exists)")
            ok += 1
            continue

        try:
            print(f"  → Translating...")
            title = translate_with_claude(client, card_path)
            print(f"  → Title: {title}")

            process_card(card_path, out_path, template, title)
            print(f"  → Saved: {out_path.name}")
            ok += 1

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append((card_path.name, str(e)))


    # Summary
    print(f"\n{'─'*50}")
    print(f"Done: {ok}/{len(images)} succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name, err in failed:
            print(f"  • {name}: {err}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit("Usage: python translate_cards.py <input_folder> <output_folder>")

    batch_process(Path(sys.argv[1]), Path(sys.argv[2]))