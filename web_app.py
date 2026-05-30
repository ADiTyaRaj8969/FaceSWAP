"""
Flask web server for the Face Swap Deepfake application.
Run: python web_app.py
Then open http://localhost:5000
"""
import os
# Must be set before any protobuf-using package (mediapipe, insightface) is imported.
# Prevents 'SymbolDatabase has no attribute GetPrototype' with protobuf 3.20+.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import io
import base64
import traceback
import uuid
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
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image

from core.detector import detect_faces, _get_insightface
from core.swapper import swap_face_insightface
from core.skin_tone import analyze_skin_tone
from core.super_res import restore_faces, upscale_image
from core.head_swap import swap_hair, match_skin_to_source, transfer_glasses
from core.hair_transfer import transfer_hair
from core.quality_checker import compute_quality_score
from utils.image_io import save_image, resize_keep_aspect

REACT_BUILD = os.path.join("static", "react")

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
_debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"] if _debug_mode else "*")

UPLOAD_DIR = "uploads/temp"
OUTPUT_DIR = "outputs/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- helpers -------------------------------------------------------------------

def _decode_image(data_or_file) -> np.ndarray | None:
    """Accept either a Flask FileStorage or a base64 data-URI string."""
    if isinstance(data_or_file, str):
        # base64 data URI: "data:image/jpeg;base64,<data>"
        if "," in data_or_file:
            data_or_file = data_or_file.split(",", 1)[1]
        raw = base64.b64decode(data_or_file)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        raw = data_or_file.read()
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


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

        # -- decode target ----------------------------------------------------
        if "target_file" in request.files and request.files["target_file"].filename:
            target = _decode_image(request.files["target_file"])
        elif request.form.get("target_b64"):
            target = _decode_image(request.form["target_b64"])
        else:
            return jsonify({"ok": False, "error": "No target image provided"}), 400

        if source is None or target is None:
            return jsonify({"ok": False, "error": "Could not decode one or both images"}), 400

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
            return jsonify({"ok": False, "error": "No face detected in source image"}), 400
        if not faces_tgt:
            return jsonify({"ok": False, "error": "No face detected in target image"}), 400

        # -- skin tone analysis (for the info chips only) ---------------------
        src_tone = analyze_skin_tone(source, faces_src[0])
        tgt_tone = analyze_skin_tone(target, faces_tgt[0])
        delta_e  = (
            (src_tone["L"] - tgt_tone["L"]) ** 2 +
            (src_tone["a"] - tgt_tone["a"]) ** 2 +
            (src_tone["b"] - tgt_tone["b"]) ** 2
        ) ** 0.5

        # 1. Core face swap (InsightFace inswapper_128). paste_back already
        #    blends the face boundary, and the source skin tone is carried by
        #    the model — so we do NOT touch the hair/neck or re-transfer colour
        #    (that smeared the hairline and shifted tone in earlier versions).
        swapped = swap_face_insightface(source, target)

        # 2. GFPGAN face restoration — recreates the detail lost in the 128px
        #    swap. THIS is what removes the blur; it runs before the preview is
        #    encoded so the on-screen result is sharp, not just the download.
        swapped = restore_faces(swapped)

        # 3. Gently align the swapped face+neck to the TARGET complexion so the
        #    face blends with the body and there's no jaw seam. Matching toward
        #    the SOURCE instead looks pasted/washed-out when source and target
        #    skin tones differ a lot (e.g. a light source on a dark target), so
        #    we keep the target's complexion for a natural, consistent result.
        swapped = match_skin_to_source(
            swapped, target, faces_tgt[0], faces_tgt[0], strength=0.5
        )

        # 4. Optional: transplant the source's hair (opt-in). InsightFace only
        #    swaps the face, so this is what makes the hair change too.
        #    Preferred: HairFastGAN (StyleGAN, GPU server-side) for a high-quality
        #    hairstyle, pasted back into the scene. Fallback: the lightweight
        #    local warp-composite when the Space is unavailable.
        swap_hair_flag = request.form.get("swap_hair", "0") in ("1", "true", "on")
        full_head      = request.form.get("full_head", "0") in ("1", "true", "on")
        if swap_hair_flag or full_head:
            hf_portrait = None
            try:
                hf_portrait = transfer_hair(face_bgr=swapped, shape_bgr=source,
                                            color_bgr=source)
            except Exception as e:
                print(f"[swap] hair transfer error: {e}")
            if hf_portrait is not None:
                # HairFast returns a TIGHT FFHQ portrait (face fills the frame),
                # which RetinaFace can't detect — pad it so the paste-back can
                # find the face and warp the new hair onto the scene.
                pad = int(max(hf_portrait.shape[:2]) * 0.4)
                hf_padded = cv2.copyMakeBorder(hf_portrait, pad, pad, pad, pad,
                                               cv2.BORDER_CONSTANT, value=(127, 127, 127))
                swapped = swap_hair(swapped, hf_padded, swapped, include_face=False)
            else:
                # Space unavailable → lightweight source-hair composite.
                swapped = swap_hair(swapped, source, target, include_face=full_head)

        # 5. Carry the source's spectacles onto the swapped face (InsightFace
        #    doesn't transfer accessories). No-op when the source wears none.
        if request.form.get("keep_glasses", "1") in ("1", "true", "on"):
            swapped = transfer_glasses(swapped, source)

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
                sk = np.asarray(max(sw, key=_area).kps, dtype=np.float32)
                tk = np.asarray(max(tg, key=_area).kps, dtype=np.float32)
                err = float(np.linalg.norm(sk - tk, axis=1).mean())
                iod = float(np.linalg.norm(tk[0] - tk[1])) or 1.0
                # Deviation as a fraction of inter-ocular distance. A clean swap
                # lands within a few % (re-detection + GFPGAN shift the features
                # slightly), so 0% => 100 and 15% => 0 gives an honest ~75-95.
                quality["alignment"] = round(max(0.0, 100.0 * (1 - (err / iod) / 0.15)), 1)
        except Exception as e:
            print(f"[swap] alignment metric skipped: {e}")

        # -- 4K upscale for download (RealESRGAN x4, Lanczos fallback) --------
        hi_res = upscale_image(swapped, scale=4)

        # -- save 4K PNG output -----------------------------------------------
        out_name = f"swap_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        save_image(hi_res, out_path)

        return jsonify({
            "ok": True,
            "result_image": _encode_image(swapped, fmt="JPEG", quality=92),
            "quality": quality,
            "delta_e": round(delta_e, 2),
            "src_tone": src_tone,
            "tgt_tone": tgt_tone,
            "output_file": out_name,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/download/<filename>")
def api_download(filename):
    """Download a previously generated swap result."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, mimetype="image/png", as_attachment=True,
                     download_name="face_swap_result.png")


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
