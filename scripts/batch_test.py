"""
Batch QA harness — swap many SOURCE faces onto every gender + location, save the
results, build a contact-sheet grid per location, and write a metrics report.

Put source faces in:
    tests/faces/male/*.jpg     tests/faces/female/*.jpg
(or pass --faces <dir>). If empty, falls back to external/HF_weights/input/.

Run on the GPU machine:
    python scripts/batch_test.py                 # face-only, all locations (fast)
    python scripts/batch_test.py --hair          # include HairFastGAN (slow!)
    python scripts/batch_test.py --gender Male --limit 5
    python scripts/batch_test.py --locations "ICT Department,Library"

Outputs:
    tests/output/<gender>/<location>/<source>.jpg
    tests/output/<gender>/<location>/_grid.jpg     (contact sheet)
    tests/output/report.csv                        (alignment/blend/ΔE/naturalness)
"""
import os, sys, csv, glob, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

import cv2
import numpy as np
from utils.image_io import resize_keep_aspect
from core.detector import detect_faces, _get_insightface
from core.swapper import swap_face_insightface
from core.super_res import restore_faces, _device
from core.head_swap import (full_head_swap, swap_hair, match_skin_to_source,
                            transfer_glasses)
from core.hair_transfer import transfer_hair
from core.blender import laplacian_blend
from core.segmentor import segment_hair_neck_skin
from core.quality_checker import compute_quality_score, _naturalness_score
from core import supabase_store

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR = os.path.join(ROOT, "tests", "faces")
OUT_DIR   = os.path.join(ROOT, "tests", "output")
LOC_DIR   = os.path.join(ROOT, "Location")
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG")


def list_sources(gender, faces_dir):
    d = os.path.join(faces_dir, gender.lower())
    files = [f for f in sorted(glob.glob(os.path.join(d, "*"))) if f.endswith(EXTS)]
    if not files:  # fallback sample set
        files = sorted(glob.glob(os.path.join(ROOT, "external", "HF_weights", "input", "*.png")))
    return files


def list_locations(gender):
    """Locations with a usable target image (Supabase if enabled, else local)."""
    if supabase_store.is_enabled():
        return [(l["folder"], None) for l in supabase_store.list_locations(gender) if l["has_image"]]
    out = []
    gdir = os.path.join(LOC_DIR, gender)
    if os.path.isdir(gdir):
        for name in sorted(os.listdir(gdir)):
            folder = os.path.join(gdir, name)
            if os.path.isdir(folder):
                imgs = [f for f in sorted(os.listdir(folder)) if f.endswith(EXTS)]
                if imgs:
                    out.append((name, os.path.join(folder, imgs[0])))
    return out


def load_target(gender, folder, local_path):
    if local_path:
        return cv2.imread(local_path)
    data = supabase_store.get_image_bytes(gender, folder)
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def run_swap(src, tgt, with_hair):
    fs, ft = detect_faces(src), detect_faces(tgt)
    if not fs or not ft:
        return None, {}
    base = tgt
    hs = full_head_swap(src, tgt)
    if hs is not None:
        base = hs
    sw = swap_face_insightface(src, base)
    sw = restore_faces(sw)
    sw = match_skin_to_source(sw, src, fs[0], ft[0], strength=0.75)
    if with_hair:
        try:
            hf = transfer_hair(face_bgr=sw, shape_bgr=src, color_bgr=src)
            if hf is not None:
                pad = int(max(hf.shape[:2]) * 0.4)
                hfp = cv2.copyMakeBorder(hf, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
                sw = swap_hair(sw, hfp, sw, include_face=False)
        except Exception as e:
            print("   hair err:", e)
    try:
        sw = transfer_glasses(sw, src)
        fm = segment_hair_neck_skin(sw).get("face_mask")
        if fm is not None and fm.max() > 0:
            sw = laplacian_blend(sw, tgt, fm, levels=4)
    except Exception:
        pass
    q = compute_quality_score(sw, tgt, None, None)
    return sw, q


def grid(images, cols=5, cell=220):
    if not images:
        return None
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 240, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        thumb = resize_keep_aspect(im, cell - 8)
        h, w = thumb.shape[:2]
        y, x = r * cell + (cell - h) // 2, c * cell + (cell - w) // 2
        canvas[y:y+h, x:x+w] = thumb
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default=FACES_DIR)
    ap.add_argument("--gender", choices=["Male", "Female", "both"], default="both")
    ap.add_argument("--locations", default="", help="comma-separated location labels (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="max sources per gender (0 = all)")
    ap.add_argument("--hair", action="store_true", help="include HairFastGAN (slow)")
    args = ap.parse_args()

    if _get_insightface() is None:
        print("InsightFace not available — aborting."); return 1
    print(f"device: {_device()}  hair: {args.hair}")

    genders = ["Male", "Female"] if args.gender == "both" else [args.gender]
    want_locs = {s.strip().lower() for s in args.locations.split(",") if s.strip()}
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [("gender", "location", "source", "alignment", "blend", "delta_e", "naturalness", "seconds")]
    total = 0

    for gender in genders:
        sources = list_sources(gender, args.faces)
        if args.limit:
            sources = sources[:args.limit]
        locs = list_locations(gender)
        if want_locs:
            from core.detector import _get_cascade  # noqa
            def clean(n): import re; return re.sub(r"^\s*\d+\.\s*", "", n).strip().lower()
            locs = [(f, p) for (f, p) in locs if clean(f) in want_locs]
        print(f"\n== {gender}: {len(sources)} sources x {len(locs)} locations ==")

        for folder, lpath in locs:
            tgt = load_target(gender, folder, lpath)
            if tgt is None:
                print(f"   [skip] {folder}: no target"); continue
            tgt = resize_keep_aspect(tgt, 1536 if _device() == "cuda" else 1024)
            odir = os.path.join(OUT_DIR, gender, folder.replace("/", "_"))
            os.makedirs(odir, exist_ok=True)
            results = []
            for sp in sources:
                src = resize_keep_aspect(cv2.imread(sp), 1536 if _device() == "cuda" else 1024)
                name = os.path.splitext(os.path.basename(sp))[0]
                t = time.time()
                sw, q = run_swap(src, tgt, args.hair)
                dt = time.time() - t
                if sw is None:
                    print(f"   [fail] {folder} <- {name}"); continue
                cv2.imwrite(os.path.join(odir, f"{name}.jpg"), sw)
                results.append(sw)
                rows.append((gender, folder, name, q.get("alignment"), q.get("blend"),
                             q.get("delta_e"), q.get("naturalness"), round(dt, 1)))
                total += 1
                print(f"   ok {folder} <- {name}  ({dt:.1f}s)  align={q.get('alignment')} nat={q.get('naturalness')}")
            g = grid(results)
            if g is not None:
                cv2.imwrite(os.path.join(odir, "_grid.jpg"), g)

    with open(os.path.join(OUT_DIR, "report.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"\nDone — {total} swaps. Report: tests/output/report.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
