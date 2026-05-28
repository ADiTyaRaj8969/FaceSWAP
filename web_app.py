"""
Flask web server for the Face Swap Deepfake application.
Run: python web_app.py
Then open http://localhost:5000
"""
import os
import io
import base64
import traceback
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

from core.detector import detect_faces
from core.landmarks import extract_landmarks_468
from core.segmentor import segment_hair_neck_skin
from core.swapper import swap_face_insightface
from core.blender import laplacian_blend, poisson_blend
from core.skin_tone import analyze_skin_tone, match_skin_tone
from core.neck_integrator import seamless_hair_to_neck_blend
from core.color_corrector import harmonize_colors
from core.quality_checker import compute_quality_score
from utils.image_io import save_image, resize_keep_aspect

REACT_BUILD = os.path.join("static", "react")

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

UPLOAD_DIR = "uploads/temp"
OUTPUT_DIR = "outputs/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- helpers -------------------------------------------------------------------

def _decode_image(data_or_file) -> np.ndarray:
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


# -- routes --------------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    """Serve the React build for all non-API routes."""
    react_index = os.path.join(REACT_BUILD, "index.html")
    if os.path.exists(react_index):
        # Serve static asset if it exists, otherwise fall back to index.html (SPA routing)
        asset = os.path.join(REACT_BUILD, path)
        if path and os.path.isfile(asset):
            return send_file(asset)
        return send_file(react_index)
    # Fallback to old Jinja templates during development before first build
    if path == "app":
        return render_template("app.html")
    return render_template("landing.html")


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
    Full face-swap pipeline.
    Accepts multipart/form-data:
      - source_file  (file)  OR  source_b64 (string)  - source face
      - target_file  (file)                            - target face
      - blend_strength, tone_match, hair_preserve, neck_blend  (0-100 integers)
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
        if "target_file" not in request.files or not request.files["target_file"].filename:
            return jsonify({"ok": False, "error": "No target image provided"}), 400
        target = _decode_image(request.files["target_file"])

        if source is None or target is None:
            return jsonify({"ok": False, "error": "Could not decode one or both images"}), 400

        # -- parameters -------------------------------------------------------
        blend_strength  = int(request.form.get("blend_strength",  85)) / 100.0
        tone_match      = int(request.form.get("tone_match",      90)) / 100.0
        hair_preserve   = int(request.form.get("hair_preserve",   80)) / 100.0
        neck_blend      = int(request.form.get("neck_blend",      75)) / 100.0

        # -- resize -----------------------------------------------------------
        source = resize_keep_aspect(source, 1024)
        target = resize_keep_aspect(target, 1024)

        # -- face detection ---------------------------------------------------
        faces_src = _safe_detect(source)
        faces_tgt = _safe_detect(target)
        if not faces_src:
            return jsonify({"ok": False, "error": "No face detected in source image"}), 400
        if not faces_tgt:
            return jsonify({"ok": False, "error": "No face detected in target image"}), 400

        # -- pipeline ---------------------------------------------------------
        src_lm    = extract_landmarks_468(source)
        tgt_lm    = extract_landmarks_468(target)
        src_masks = segment_hair_neck_skin(source)
        tgt_masks = segment_hair_neck_skin(target)

        src_tone  = analyze_skin_tone(source, faces_src[0])
        tgt_tone  = analyze_skin_tone(target, faces_tgt[0])
        delta_e   = (
            (src_tone["L"] - tgt_tone["L"]) ** 2 +
            (src_tone["a"] - tgt_tone["a"]) ** 2 +
            (src_tone["b"] - tgt_tone["b"]) ** 2
        ) ** 0.5

        swapped = swap_face_insightface(source, target)
        swapped = match_skin_tone(swapped, target, src_tone, tgt_tone,
                                  strength=tone_match)
        swapped = seamless_hair_to_neck_blend(
            source_img=swapped, target_img=target,
            src_masks=src_masks, tgt_masks=tgt_masks,
            src_landmarks=src_lm, tgt_landmarks=tgt_lm,
            hair_preserve=hair_preserve,
            neck_strength=neck_blend,
            blend_strength=blend_strength,
        )
        blend_mask = tgt_masks["face_mask"]
        swapped = laplacian_blend(target, swapped, blend_mask, levels=4)
        swapped = poisson_blend(swapped, target, blend_mask)
        swapped = harmonize_colors(swapped, target, tgt_masks)

        quality = compute_quality_score(swapped, target, src_lm, tgt_lm)

        # -- save output -------------------------------------------------------
        out_name = f"swap_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        save_image(swapped, out_path)

        return jsonify({
            "ok": True,
            "result_image": _encode_image(swapped, fmt="JPEG", quality=90),
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


if __name__ == "__main__":
    print("Starting Face Swap Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
