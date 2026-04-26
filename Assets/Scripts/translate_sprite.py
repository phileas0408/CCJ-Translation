#!/usr/bin/env python3
"""
translate_sprite.py — Batch sprite text translator

Usage:
    python translate_sprite.py <input_folder> <output_folder>

    Detects Japanese text in the lower region of each PNG/JPG, translates it
    via Claude Vision, erases the original, and redraws with the English text.
    Verifies placement matches the original region using IoU.

Requirements:
    pip install pillow anthropic numpy

Environment:
    ANTHROPIC_API_KEY  – your Anthropic API key (required)
    FONT_PATH          – path to TCG2SAB.ttf (default: TCG2SAB.ttf)
"""

import base64
import json
import os
import re
import sys
from pathlib import Path

# Load .env from the project root (two levels up from Assets/Scripts/)
_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
def _resolve(p: str, default: str) -> Path:
    path = Path(os.environ.get(p, default))
    return path if path.is_absolute() else _project_root / path

FONT_PATH = _resolve("FONT_PATH", "Assets/Fonts/TCG2SAB.ttf")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
IOU_THRESHOLD  = 0.5


# ── Client ────────────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def image_to_base64(path: Path) -> tuple[str, str]:
    with open(path, "rb") as f:
        data = f.read()
    ext = path.suffix.lower().lstrip(".")
    media_type = "image/png" if ext == "png" else f"image/{ext}"
    return base64.standard_b64encode(data).decode("utf-8"), media_type


def detect_text_bbox(image_path: Path, search_from_row: float = 0.7) -> tuple | None:
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)
    h = arr.shape[0]
    start_row = int(h * search_from_row)

    alpha = arr[start_row:, :, 3]
    rows = np.where(alpha.max(axis=1) > 50)[0]
    if len(rows) == 0:
        return None

    y_min = int(rows.min()) + start_row
    y_max = int(rows.max()) + start_row
    cols = np.where(arr[y_min:y_max + 1, :, 3].max(axis=0) > 50)[0]
    if len(cols) == 0:
        return None

    return (int(cols.min()), y_min, int(cols.max()), y_max)


def compute_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Translation ───────────────────────────────────────────────────────────────
def translate_with_claude(client, image_path: Path) -> tuple[str, str]:
    b64, media_type = image_to_base64(image_path)
    prompt = (
        "This game item image has a short Japanese label at the bottom. "
        "Translate it to English. "
        'Respond ONLY with valid JSON: {"original": "...", "translation": "..."}'
    )
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    data = json.loads(raw)
    return data["original"], data["translation"]


# ── Image processing ──────────────────────────────────────────────────────────
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(str(FONT_PATH), size)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _draw_line(draw: ImageDraw.Draw, text: str, cx: int, cy: int, font_size: int, bw: int) -> None:
    """Fit text to bw, then draw it centered at (cx, cy) with stroke."""
    while font_size > 4:
        font = _load_font(font_size)
        tb = draw.textbbox((0, 0), text, font=font)
        if (tb[2] - tb[0]) <= bw:
            break
        font_size -= 1
    font = _load_font(font_size)
    tb = draw.textbbox((0, 0), text, font=font)
    px = cx - (tb[2] - tb[0]) // 2
    py = cy - (tb[3] - tb[1]) // 2
    stroke = max(1, font_size // 8)
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx != 0 or dy != 0:
                draw.text((px + dx, py + dy), text, font=font, fill=(0, 0, 0, 255))
    draw.text((px, py), text, font=font, fill=(255, 255, 255, 255))


def _best_wrap(draw: ImageDraw.Draw, text: str, font_size: int, bw: int) -> tuple[str, str] | None:
    """Split text at the word boundary closest to the midpoint that fits both halves in bw."""
    words = text.split()
    if len(words) < 2:
        return None
    best = None
    mid = len(text) // 2
    for i in range(1, len(words)):
        line1 = " ".join(words[:i])
        line2 = " ".join(words[i:])
        font = _load_font(font_size)
        w1 = draw.textbbox((0, 0), line1, font=font)[2]
        w2 = draw.textbbox((0, 0), line2, font=font)[2]
        if w1 <= bw and w2 <= bw:
            split_pos = len(line1)
            if best is None or abs(split_pos - mid) < abs(len(best[0]) - mid):
                best = (line1, line2)
    return best


def draw_translated_text(img: Image.Image, text: str, box: tuple) -> Image.Image:
    x1, y1, x2, y2 = box
    pad = 3
    arr = np.array(img)
    arr[max(0, y1 - pad):y2 + pad + 1, max(0, x1 - pad):x2 + pad + 1] = [0, 0, 0, 0]
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    bw, bh = x2 - x1, y2 - y1
    cx = x1 + bw // 2

    # Split "Main text (parenthetical)" into two lines
    paren_match = re.search(r'\s*(\(.*\))\s*$', text)
    if paren_match:
        main_text  = text[:paren_match.start()].strip()
        paren_text = paren_match.group(1)
        half = bh // 2
        _draw_line(draw, main_text,  cx, y1 + half // 2,       bh,   bw)
        _draw_line(draw, paren_text, cx, y1 + half + half // 2, half, bw)
    else:
        # Check if single-line fits at a reasonable size; wrap if not
        fit_size = bh
        while fit_size > 4:
            font = _load_font(fit_size)
            if draw.textbbox((0, 0), text, font=font)[2] <= bw:
                break
            fit_size -= 1

        wrap = _best_wrap(draw, text, bh // 2, bw) if fit_size < bh // 2 else None
        if wrap:
            line1, line2 = wrap
            half = bh // 2
            _draw_line(draw, line1, cx, y1 + half // 2,       half, bw)
            _draw_line(draw, line2, cx, y1 + half + half // 2, half, bw)
        else:
            _draw_line(draw, text, cx, y1 + bh // 2, bh, bw)

    return img


# ── Batch runner ──────────────────────────────────────────────────────────────
def batch_process(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.is_dir():
        sys.exit(f"Input folder not found: {input_dir}")
    if not FONT_PATH.exists():
        print(f"Warning: font not found at {FONT_PATH}, will use system default.")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        sys.exit(f"No supported images found in {input_dir}")

    print(f"Found {len(images)} image(s) in '{input_dir}'")
    print(f"Output folder: '{output_dir}'\n")

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
            orig_box = detect_text_bbox(card_path)
            if orig_box is None:
                print("  ⚠ Could not detect text region, skipping.")
                failed.append((card_path.name, "no text region detected"))
                continue
            print(f"  → Detected bbox: {orig_box[0]},{orig_box[1]} – {orig_box[2]},{orig_box[3]}")

            print(f"  → Translating...")
            original, translation = translate_with_claude(client, card_path)
            print(f"  → {original!r} → {translation!r}")

            img = Image.open(card_path).convert("RGBA")
            result = draw_translated_text(img, translation, orig_box)
            result.save(str(out_path), "PNG")

            trans_box = detect_text_bbox(out_path)
            if trans_box:
                iou = compute_iou(orig_box, trans_box)
                status = "✓ PASS" if iou >= IOU_THRESHOLD else "✗ low IoU — check manually"
                print(f"  → IoU: {iou:.3f}  {status}")

            print(f"  → Saved: {out_path.name}")
            ok += 1

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append((card_path.name, str(e)))

    print(f"\n{'─' * 50}")
    print(f"Done: {ok}/{len(images)} succeeded.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name, err in failed:
            print(f"  • {name}: {err}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit("Usage: python translate_sprite.py <input_folder> <output_folder>")

    batch_process(Path(sys.argv[1]), Path(sys.argv[2]))
