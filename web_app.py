"""
Flask web server for the Face Swap Deepfake application.
Run: python web_app.py
Then open http://localhost:5000
"""
import os
# Required before importing mediapipe/insightface on some platforms to avoid
# protobuf C-extension symbol errors.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import io
import re
import base64
import traceback
import mimetypes

# python:3.10-slim has an incomplete MIME database — JS/CSS would be served as
# application/octet-stream and the browser would refuse to execute them.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")

import cv2
import cv2.data
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image, ImageOps

from core.detector import detect_faces, _get_insightface
from core.swapper import swap_face_insightface
from core.skin_tone import analyze_skin_tone
from core.super_res import restore_faces, upscale_image
from core.head_swap import (swap_hair, match_skin_to_source, transfer_glasses,
                            full_head_swap)
from core.hair_transfer import transfer_hair
from core.blender import laplacian_blend
from core.quality_checker import compute_quality_score
from utils.image_io import resize_keep_aspect

REACT_BUILD   = os.path.join("static", "react")
LOCATIONS_DIR = "Location"            # Location/Male/<loc>/img  +  Location/Female/<loc>/img
VALID_GENDERS = {"Male", "Female"}
_IMG_EXTS     = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _clean_location_label(name: str) -> str:
    """Strip a leading numeric prefix like '1. ' for display."""
    return re.sub(r"^\s*\d+\.\s*", "", name).strip()


def _list_locations(gender: str) -> list:
    """
    Return [{folder, label}] for every location sub-folder under the gender,
    sorted by the numeric prefix. Folders are listed even if they have no image
    yet (the picker shows them; the swap call validates the image exists).
    """
    if gender not in VALID_GENDERS:
        return []
    gender_dir = os.path.join(LOCATIONS_DIR, gender)
    if not os.path.isdir(gender_dir):
        return []

    def sort_key(n):
        m = re.match(r"^\s*(\d+)", n)
        return (int(m.group(1)) if m else 9999, n.lower())

    out = []
    for name in sorted(os.listdir(gender_dir), key=sort_key):
        if os.path.isdir(os.path.join(gender_dir, name)):
            out.append({"folder": name, "label": _clean_location_label(name),
                        "has_image": _find_location_image(gender, name) is not None})
    return out


def _find_location_image(gender: str, location: str):
    """
    Resolve the target image path inside Location/<gender>/<location>/.
    Returns the first image file found, or None. Guards against path traversal.
    """
    if gender not in VALID_GENDERS:
        return None
    # Reject anything that could escape the locations dir.
    if not location or ".." in location or "/" in location or "\\" in location:
        return None
    folder = os.path.join(LOCATIONS_DIR, gender, location)
    if not os.path.isdir(folder):
        return None
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(_IMG_EXTS) and os.path.isfile(os.path.join(folder, f)):
            return os.path.join(folder, f)
    return None

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
_debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"] if _debug_mode else "*")

# Nothing is written to disk — uploads are decoded in memory and the result is
# returned inline as a data-URI for client-side download.


# -- helpers -------------------------------------------------------------------

def _decode_image(data_or_file) -> np.ndarray | None:
    """
    Accept a Flask FileStorage or a base64 data-URI string and return a BGR image.
    Honours EXIF orientation so phone photos (stored rotated with an orientation
    tag) aren't processed sideways — cv2.imdecode ignores EXIF, PIL applies it.
    """
    if isinstance(data_or_file, str):
        # base64 data URI: "data:image/jpeg;base64,<data>"
        if "," in data_or_file:
            data_or_file = data_or_file.split(",", 1)[1]
        raw = base64.b64decode(data_or_file)
    else:
        raw = data_or_file.read()

    try:
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil) or pil  # auto-rotate; fallback if None
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback: raw decode (no EXIF) if PIL can't read it.
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_image(img: np.ndarray, fmt: str = "JPEG", quality: int = 88) -> str:
    """Return a base64 data-URI for a BGR numpy image."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _safe_detect(img):
    faces = detect_faces(img)
    return faces


def _enhance_input(img: np.ndarray) -> np.ndarray:
    """
    Enhance an uploaded image before it enters the swap pipeline, so low-res
    photos don't lose detail. Small images are upscaled (RealESRGAN/Lanczos) to a
    workable size, then GFPGAN restores facial detail. Large images are just
    face-restored. The result feeds detection + swap, and the final output is
    enhanced again — detail is preserved at both ends.
    """
    try:
        h, w = img.shape[:2]
        # Only enhance genuinely low-res uploads. Already-decent photos are left
        # alone so they don't get GFPGAN-restored on input AND output (stacking
        # GFPGAN makes skin look plastic/unnatural).
        if max(h, w) < 800:
            img = upscale_image(img, scale=2)        # bring small uploads up
            img = resize_keep_aspect(img, 1280)      # but cap the working size
            img = restore_faces(img)                 # restore detail in the upscale
        return img
    except Exception as e:
        print(f"[swap] input enhance skipped: {e}")
        return img


# -- routes --------------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    """Serve the React SPA — assets by exact path, everything else → index.html."""
    react_dir   = os.path.abspath(REACT_BUILD)
    react_index = os.path.join(react_dir, "index.html")

    if not os.path.exists(react_index):
        return (
            "<h2 style='font-family:sans-serif;padding:2rem;color:#D4DE95;"
            "background:#0f1209;min-height:100vh'>"
            "React build not found.<br><br>"
            "<code style='font-size:0.9rem'>"
            "cd frontend &amp;&amp; npm install &amp;&amp; npm run build"
            "</code></h2>",
            503,
        )

    # Serve the asset if it exists (JS, CSS, images, etc.)
    if path:
        asset = os.path.join(react_dir, path)
        if os.path.isfile(asset):
            return send_from_directory(react_dir, path)

    # SPA fallback — let React Router handle the route
    return send_from_directory(react_dir, "index.html")


@app.route("/api/locations", methods=["GET"])
def api_locations():
    """List the curated destination locations for a gender (Male|Female)."""
    gender = (request.args.get("gender") or "").strip().capitalize()
    if gender not in VALID_GENDERS:
        return jsonify({"ok": False, "error": "gender must be Male or Female"}), 400
    return jsonify({"ok": True, "gender": gender, "locations": _list_locations(gender)})


@app.route("/api/location-image", methods=["GET"])
def api_location_image():
    """Serve the curated target image for a gender + location (for the preview)."""
    gender   = (request.args.get("gender") or "").strip().capitalize()
    location = (request.args.get("location") or "").strip()
    path = _find_location_image(gender, location)
    if path is None:
        return jsonify({"ok": False, "error": "Image not available yet"}), 404
    return send_file(os.path.abspath(path))


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Quick face-detection check. Returns count + thumbnail with boxes drawn."""
    try:
        if "image" in request.files:
            img = _decode_image(request.files["image"])
        else:
            data = request.get_json(force=True)
            img = _decode_image(data["image"])

        if img is None:
            return jsonify({"ok": False, "error": "Could not decode image"}), 400

        img = resize_keep_aspect(img, 800)
        faces = _safe_detect(img)

        # Draw boxes on thumbnail
        preview = img.copy()
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 80), 2)

        thumbnail = resize_keep_aspect(preview, 400)
        return jsonify({
            "ok": True,
            "faces": len(faces),
            "thumbnail": _encode_image(thumbnail),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/swap", methods=["POST"])
def api_swap():
    """
    Full face-swap pipeline: InsightFace swap → GFPGAN face restoration →
    RealESRGAN 4K upscale (for download).
    Accepts multipart/form-data:
      - source_file  (file)  OR  source_b64 (string)  - source face
      - target_file  (file)                            - target face
    Returns JSON with result_image (base64), quality metrics, delta_e.
    """
    try:
        # -- decode source ----------------------------------------------------
        if "source_file" in request.files and request.files["source_file"].filename:
            source = _decode_image(request.files["source_file"])
        elif request.form.get("source_b64"):
            source = _decode_image(request.form["source_b64"])
        else:
            return jsonify({"ok": False, "error": "No source image provided"}), 400

        # -- resolve target ---------------------------------------------------
        # The user no longer uploads a target. They pick a gender + location and
        # we use the corresponding curated image from Location/<gender>/<location>/.
        # (A direct target upload is still accepted as an admin override.)
        if "target_file" in request.files and request.files["target_file"].filename:
            target = _decode_image(request.files["target_file"])
        elif request.form.get("target_b64"):
            target = _decode_image(request.form["target_b64"])
        else:
            gender   = (request.form.get("gender") or "").strip().capitalize()
            location = (request.form.get("location") or "").strip()
            if gender not in VALID_GENDERS or not location:
                return jsonify({"ok": False, "error":
                    "Please choose a gender (Male/Female) and a location."}), 400
            tgt_path = _find_location_image(gender, location)
            if tgt_path is None:
                return jsonify({"ok": False, "error":
                    f"No photo is available yet for {gender} · "
                    f"{_clean_location_label(location)}. Please pick another location."}), 404
            with open(tgt_path, "rb") as _tf:
                target = _decode_image(_tf)

        if source is None or target is None:
            return jsonify({"ok": False, "error": "Could not decode one or both images"}), 400

        user_name = (request.form.get("name") or "").strip()

        # -- resize -----------------------------------------------------------
        # 1024 working resolution is plenty: InsightFace swaps at 128px and
        # GFPGAN restores on 512px face crops, so a larger canvas only wastes
        # time. The 4x RealESRGAN upscale at the end takes this to ~4K.
        source = resize_keep_aspect(source, 1024)
        target = resize_keep_aspect(target, 1024)

        # -- enhance inputs (upscale small + GFPGAN restore) so detail isn't
        #    lost through the pipeline; the output is enhanced again at the end.
        source = _enhance_input(source)
        target = _enhance_input(target)

        # -- face detection ---------------------------------------------------
        faces_src = _safe_detect(source)
        faces_tgt = _safe_detect(target)
        if not faces_src:
            return jsonify({"ok": False, "error":
                "No face detected in source image. Tips: ensure good lighting, "
                "face the camera directly, remove heavy occlusions (mask/sunglasses), "
                "and use a photo where the face is at least 10% of the frame."}), 400
        if not faces_tgt:
            return jsonify({"ok": False, "error":
                "No face detected in target image. Tips: ensure good lighting, "
                "face the camera directly, and use a clear frontal portrait."}), 400

        # -- warn on very small detected faces (quality will be poor) ---------
        warnings = []
        x1, y1, x2, y2 = faces_src[0]
        src_face_px = min(x2 - x1, y2 - y1)
        if src_face_px < 80:
            warnings.append(
                "Source face is very small — swap quality may be reduced. "
                "Use a closer/higher-resolution photo for best results."
            )
        x1, y1, x2, y2 = faces_tgt[0]
        tgt_face_px = min(x2 - x1, y2 - y1)
        if tgt_face_px < 80:
            warnings.append(
                "Target face is very small — swap quality may be reduced. "
                "Use a closer/higher-resolution photo for best results."
            )

        # -- skin tone analysis (for the info chips only) ---------------------
        src_tone = analyze_skin_tone(source, faces_src[0])
        tgt_tone = analyze_skin_tone(target, faces_tgt[0])
        delta_e  = (
            (src_tone["L"] - tgt_tone["L"]) ** 2 +
            (src_tone["a"] - tgt_tone["a"]) ** 2 +
            (src_tone["b"] - tgt_tone["b"]) ** 2
        ) ** 0.5

        # 1. HEAD SWAP — transplant the SOURCE's whole head (face SHAPE + skin +
        #    hair + glasses) onto the target's body, so the source's face shape is
        #    preserved (InsightFace alone would impose the target's shape). Only
        #    the head region is touched, so the background stays intact. Falls
        #    back to the InsightFace face swap if the transplant can't run.
        swapped = None
        if request.form.get("head_swap", "1") in ("1", "true", "on"):
            try:
                swapped = full_head_swap(source, target)
            except Exception as e:
                print(f"[swap] head swap error: {e}")
        head_done = swapped is not None
        if not head_done:
            swapped = swap_face_insightface(source, target)

        # 2. GFPGAN face restoration — natural facial detail, removes any softness
        #    from the warp/swap. Runs before the preview is encoded.
        swapped = restore_faces(swapped)

        # 3. SOURCE complexion across face+neck (one consistent tone, no jaw
        #    seam; the head swap already carries the source skin, this also pulls
        #    the target's visible neck to match).
        swapped = match_skin_to_source(
            swapped, source, faces_src[0], faces_tgt[0], strength=0.75
        )

        # 4. If we only did a FACE swap (head transplant unavailable), add the
        #    source's hair + glasses separately — the head swap already has both.
        if not head_done:
            swap_hair_flag = request.form.get("swap_hair", "1") in ("1", "true", "on")
            full_head      = request.form.get("full_head", "0") in ("1", "true", "on")
            if swap_hair_flag or full_head:
                hf_portrait = None
                try:
                    hf_portrait = transfer_hair(face_bgr=swapped, shape_bgr=source,
                                                color_bgr=source)
                except Exception as e:
                    print(f"[swap] hair transfer error: {e}")
                if hf_portrait is not None:
                    pad = int(max(hf_portrait.shape[:2]) * 0.4)
                    hf_padded = cv2.copyMakeBorder(hf_portrait, pad, pad, pad, pad,
                                                   cv2.BORDER_CONSTANT, value=(127, 127, 127))
                    swapped = swap_hair(swapped, hf_padded, swapped, include_face=False)
                else:
                    swapped = swap_hair(swapped, source, target, include_face=full_head)
            if request.form.get("keep_glasses", "1") in ("1", "true", "on"):
                swapped = transfer_glasses(swapped, source)

        # 5. Laplacian pyramid blend over the face boundary — multi-scale so
        #    high-frequency hair/skin detail and low-frequency colour transitions
        #    are blended independently. This eliminates the hard edge that can
        #    remain after InsightFace's paste_back and the hair composite above.
        try:
            from core.segmentor import segment_hair_neck_skin
            face_mask = segment_hair_neck_skin(swapped).get("face_mask")
            if face_mask is not None and face_mask.max() > 0:
                swapped = laplacian_blend(swapped, target, face_mask, levels=4)
        except Exception as e:
            print(f"[swap] Laplacian blend skipped: {e}")

        # -- quality metrics --------------------------------------------------
        quality = compute_quality_score(swapped, target, None, None)

        # Real alignment: how closely the swapped face's 5 landmarks sit on the
        # target's (the swap keeps the target geometry, so a clean swap aligns
        # tightly). Normalised by inter-ocular distance => resolution-independent.
        try:
            _ifa = _get_insightface()
            _area = lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            sw = _ifa.get(swapped) if _ifa else []
            tg = _ifa.get(target) if _ifa else []
            if sw and tg:
                swf = max(sw, key=_area)
                sk = np.asarray(swf.kps, dtype=np.float32)
                tk = np.asarray(max(tg, key=_area).kps, dtype=np.float32)
                err = float(np.linalg.norm(sk - tk, axis=1).mean())
                iod = float(np.linalg.norm(tk[0] - tk[1])) or 1.0
                # Deviation as a fraction of inter-ocular distance. A clean swap
                # lands within a few % (re-detection + GFPGAN shift the features
                # slightly), so 0% => 100 and 15% => 0 gives an honest ~75-95.
                quality["alignment"] = round(max(0.0, 100.0 * (1 - (err / iod) / 0.15)), 1)

                # Naturalness on the FACE region only. Measuring the whole frame
                # unfairly penalises studio shots (white background + dark shirt
                # = lots of clipped pixels + hard edges) even when the face is
                # perfect, so crop to the face before scoring.
                from core.quality_checker import _naturalness_score
                x1, y1, x2, y2 = [int(v) for v in swf.bbox]
                pad = int((x2 - x1) * 0.25)
                fy1, fy2 = max(0, y1 - pad), min(swapped.shape[0], y2 + pad)
                fx1, fx2 = max(0, x1 - pad), min(swapped.shape[1], x2 + pad)
                face = swapped[fy1:fy2, fx1:fx2]
                if face.size > 0:
                    quality["naturalness"] = _naturalness_score(face)
        except Exception as e:
            print(f"[swap] alignment metric skipped: {e}")

        # -- 4K upscale for download (RealESRGAN x4, Lanczos fallback) --------
        hi_res = upscale_image(swapped, scale=4)

        # The 4K result is returned inline as a base64 data-URI so the user can
        # download it client-side (works on the HF Space too). We do NOT store it
        # server-side — nothing is written to disk (privacy + no disk growth).
        download_uri = _encode_image(hi_res, fmt="JPEG", quality=95)

        return jsonify({
            "ok": True,
            "result_image": _encode_image(swapped, fmt="JPEG", quality=92),
            "download_image": download_uri,
            "quality": quality,
            "delta_e": round(delta_e, 2),
            "src_tone": src_tone,
            "tgt_tone": tgt_tone,
            "warnings": warnings,
            "name": user_name,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


def _prewarm_models():
    """Load heavy ML models at startup so the first swap request is fast."""
    print("Pre-warming ML models (this takes ~30s on first run)...")
    try:
        from core.detector import _get_insightface
        from core.swapper import _load_swapper
        from core.super_res import _load_gfpgan, _load_realesrgan
        _get_insightface()
        _load_swapper()
        _load_gfpgan()
        _load_realesrgan()
        print("Models ready.")
    except Exception as e:
        print(f"Model pre-warm skipped: {e}")


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print(f"Starting Face Swap Web App on port {port}...")
    if not debug:
        _prewarm_models()
    app.run(debug=debug, use_reloader=False, host="0.0.0.0", port=port)
