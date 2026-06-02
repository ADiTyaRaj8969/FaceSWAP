"""
One-time migration: upload local Location/<gender>/<n. name>/<image>
into Supabase (Storage bucket + locations table), preserving the folder number
as sort_order and using the clean label as the name.

Prereqs:
  1. Run supabase_schema.sql once in the Supabase SQL editor.
  2. Set SUPABASE_URL + SUPABASE_SERVICE_KEY in .env.

Run:  python scripts/seed_supabase_locations.py
Re-runnable (idempotent): re-uploading just overwrites the image + row.
"""
import os
import re
import io
import sys

# project root on path so `core` imports work when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from PIL import Image, ImageOps          # noqa: E402
from core import supabase_store          # noqa: E402

LOCATIONS_DIR = "Location"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def clean_label(name: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", name).strip()


def folder_number(name: str) -> int:
    m = re.match(r"^\s*(\d+)", name)
    return int(m.group(1)) if m else 9999


def to_jpeg_bytes(path: str) -> bytes:
    pil = Image.open(path)
    pil = ImageOps.exif_transpose(pil) or pil      # honour phone EXIF rotation
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def main():
    if not supabase_store.is_enabled():
        print("✗ Supabase is not configured.")
        print("  Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env, then retry.")
        sys.exit(1)

    print(f"→ Supabase: {supabase_store.SUPABASE_URL}  bucket={supabase_store.BUCKET}\n")

    uploaded = empty = errors = 0
    for gender in ("Male", "Female"):
        gdir = os.path.join(LOCATIONS_DIR, gender)
        if not os.path.isdir(gdir):
            continue
        print(f"== {gender} ==")
        for folder in sorted(os.listdir(gdir), key=folder_number):
            fpath = os.path.join(gdir, folder)
            if not os.path.isdir(fpath):
                continue
            label = clean_label(folder)
            order = folder_number(folder)
            imgs = [f for f in sorted(os.listdir(fpath)) if f.lower().endswith(IMG_EXTS)]
            if not imgs:
                print(f"   · {order:>2}. {label:<28} — no photo (skipped)")
                empty += 1
                continue
            try:
                data = to_jpeg_bytes(os.path.join(fpath, imgs[0]))
                supabase_store.upsert_location(gender, label, data, sort_order=order)
                print(f"   ✓ {order:>2}. {label:<28} ← {imgs[0]}")
                uploaded += 1
            except Exception as e:
                print(f"   ✗ {order:>2}. {label:<28} ERROR: {e}")
                errors += 1
        print()

    print(f"Done — {uploaded} uploaded, {empty} empty, {errors} errors.")


if __name__ == "__main__":
    main()
