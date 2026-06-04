"""
Flask web server for the Face Swap Deepfake application.
Run: python web_app.py
Then open http://localhost:5000
"""
import os
# Load .env before anything reads os.environ (Supabase keys, admin passwords).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
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

from core.detector import detect_faces, _get_insightface, align_face_upright, pad_until_detectable
from core.swapper import swap_face_insightface
from core.skin_tone import analyze_skin_tone
from core.super_res import restore_faces, upscale_image
from core.head_swap import (swap_hair, match_skin_to_source, transfer_glasses,
                            full_head_swap)
from core.hair_transfer import transfer_hair
from core.blender import laplacian_blend
from core.quality_checker import compute_quality_score
from core import supabase_store
from utils.image_io import resize_keep_aspect

REACT_BUILD   = os.path.join("static", "react")
LOCATIONS_DIR = "Location"            # Location/Male/<loc>/img  +  Location/Female/<loc>/img
VALID_GENDERS = {"Male", "Female"}
_IMG_EXTS     = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Control Panel access is restricted to these Google accounts. Keep in sync with
# ALLOWED_ADMINS in frontend/src/pages/AdminPage.jsx. Comma-separated in env.
ALLOWED_ADMIN_EMAILS = set(filter(None, (
    e.strip().lower() for e in
    (os.environ.get("ALLOWED_ADMIN_EMAILS")
     or "adivid198986@gmail.com,nishith.kotak@gmail.com,cdparmar9824416484@gmail.com").split(",")
)))

# Primary admin(s) — full control incl. DELETE. Other allowlisted admins can
# add + edit (update/rename) but NOT delete. Keep in sync with the frontend.
PRIMARY_ADMIN_EMAILS = set(filter(None, (
    e.strip().lower() for e in
    (os.environ.get("PRIMARY_ADMIN_EMAILS") or "adivid198986@gmail.com").split(",")
)))

# Firebase Web API key — used to verify the owner's Google ID token server-side.
FIREBASE_API_KEY = os.environ.get(
    "FIREBASE_API_KEY", "AIzaSyCF9t3xPf_Se4qkqAHFRoW-YqG8LfoQIpo")

# Optional legacy passwords (break-glass). Empty by default — Google is primary.
ADMIN_PASSWORDS = set(filter(None, (
    os.environ.get("ADMIN_PASSWORD_1", ""),
    os.environ.get("ADMIN_PASSWORD_2", ""),
)))


def _verify_google_email(id_token: str):
    """
    Verify a Firebase Google ID token via Identity Toolkit and return the
    verified email (lowercased), or None. Rejects invalid/expired/unverified.
    """
    if not id_token or not FIREBASE_API_KEY:
        return None
    try:
        import requests
        r = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}",
            json={"idToken": id_token}, timeout=10)
        if r.status_code != 200:
            return None
        users = r.json().get("users") or []
        if not users:
            return None
        u = users[0]
        email = (u.get("email") or "").lower()
        if not email or not u.get("emailVerified", False):
            return None
        return email
    except Exception as e:
        print(f"[admin] token verify failed: {e}")
        return None


def _admin_identity(req):
    """
    Resolve the caller's admin identity from the request token.
    Returns (email, is_admin):
      - email: verified Google email (lowercased), or "" for legacy-password auth
      - is_admin: True if authorised at all
    Token is read from the X-Admin-Token header, form field, or JSON body.
    """
    tok = req.headers.get("X-Admin-Token") or ""
    if not tok and req.form:
        tok = req.form.get("admin_token") or ""
    if not tok:
        data = req.get_json(silent=True) or {}
        tok = data.get("admin_token") or ""
    if not tok:
        return (None, False)

    # Firebase ID tokens are JWTs: header.payload.signature (two dots, long).
    if tok.count(".") == 2 and len(tok) > 100:
        email = _verify_google_email(tok)
        if email and email in ALLOWED_ADMIN_EMAILS:
            return (email, True)
        return (None, False)

    # Legacy password fallback (break-glass) — treated as a primary admin.
    if bool(ADMIN_PASSWORDS) and tok in ADMIN_PASSWORDS:
        return ("", True)
    return (None, False)


def _is_primary(email) -> bool:
    """Primary admin (full control incl. delete). Legacy password == primary."""
    return email == "" or (email is not None and email in PRIMARY_ADMIN_EMAILS)


def _check_admin(req) -> bool:
    """Authorised at all? (add/edit allowed for any admin)."""
    return _admin_identity(req)[1]


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
        folder = os.path.join(gender_dir, name)
        if os.path.isdir(folder):
            msg = ""
            mp = os.path.join(folder, "message.txt")
            if os.path.isfile(mp):
                try:
                    with open(mp, encoding="utf-8") as f:
                        msg = f.read().strip()
                except OSError:
                    pass
            out.append({"folder": name, "label": _clean_location_label(name),
                        "has_image": _find_location_image(gender, name) is not None,
                        "message": msg})
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


def _next_location_number(gender: str) -> int:
    """Highest numeric prefix in the gender folder + 1 (for new locations)."""
    gender_dir = os.path.join(LOCATIONS_DIR, gender)
    n = 0
    if os.path.isdir(gender_dir):
        for name in os.listdir(gender_dir):
            m = re.match(r"^\s*(\d+)", name)
            if m:
                n = max(n, int(m.group(1)))
    return n + 1


def _resolve_or_create_location(gender: str, location_name: str, create: bool = True):
    """
    Find the folder for a location label inside a gender (matched by cleaned
    label, case-insensitive). Creates '<next-number>. <name>' if missing and
    create=True. Returns the folder name, or None.
    """
    gender_dir = os.path.join(LOCATIONS_DIR, gender)
    clean = _clean_location_label(location_name)
    if not clean:
        return None
    if os.path.isdir(gender_dir):
        for name in os.listdir(gender_dir):
            if os.path.isdir(os.path.join(gender_dir, name)) and \
               _clean_location_label(name).lower() == clean.lower():
                return name
    if not create:
        return None
    folder = f"{_next_location_number(gender)}. {clean}"
    os.makedirs(os.path.join(gender_dir, folder), exist_ok=True)
    return folder

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
    if faces:
        return faces
    # Close-up face filling the frame defeats the detector (0 faces). Pad a
    # replicated margin, detect there, and remap the boxes back to the original
    # image coords so the count is right and the upload isn't wrongly rejected.
    h, w = img.shape[:2]
    for frac in (0.3, 0.5, 0.8):
        p = int(max(h, w) * frac)
        padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE)
        pf = detect_faces(padded)
        if pf:
            return [(max(0, x1 - p), max(0, y1 - p),
                     min(w, x2 - p), min(h, y2 - p)) for (x1, y1, x2, y2) in pf]
    return []


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
    locs = supabase_store.list_locations(gender) if supabase_store.is_enabled() \
        else _list_locations(gender)
    resp = jsonify({"ok": True, "gender": gender, "locations": locs})
    resp.headers["Cache-Control"] = "no-store"   # always reflect latest uploads
    return resp


@app.route("/api/location-image", methods=["GET"])
def api_location_image():
    """Serve the curated target image for a gender + location (for the preview)."""
    gender   = (request.args.get("gender") or "").strip().capitalize()
    location = (request.args.get("location") or "").strip()

    if supabase_store.is_enabled():
        data = supabase_store.get_image_bytes(gender, location)
        if not data:
            return jsonify({"ok": False, "error": "Image not available yet"}), 404
        resp = send_file(io.BytesIO(data), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    path = _find_location_image(gender, location)
    if path is None:
        return jsonify({"ok": False, "error": "Image not available yet"}), 404
    resp = send_file(os.path.abspath(path))
    resp.headers["Cache-Control"] = "no-cache"   # always reflect latest upload
    return resp


@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():
    """Validate owner credentials server-side (frontend also checks)."""
    data = request.get_json(silent=True) or {}
    pw = (data.get("password") or "").strip()
    return jsonify({"ok": pw in ADMIN_PASSWORDS})


@app.route("/api/admin/location", methods=["POST"])
def api_admin_add_location():
    """
    Owner-only: add or update a location image.
    Form: admin_token, gender (Male|Female), location (display name), image (file).
    Creates the folder if new, then stores the (re-encoded) image. Becomes
    visible on the app page immediately.
    """
    if not _check_admin(request):
        return jsonify({"ok": False, "error": "Unauthorised — owner login required."}), 401

    gender = (request.form.get("gender") or "").strip().capitalize()
    if gender not in VALID_GENDERS:
        return jsonify({"ok": False, "error": "Gender must be Male or Female."}), 400

    location_name = (request.form.get("location") or "").strip()
    if not location_name:
        return jsonify({"ok": False, "error": "Location name is required."}), 400
    if ".." in location_name or "/" in location_name or "\\" in location_name:
        return jsonify({"ok": False, "error": "Invalid location name."}), 400

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"ok": False, "error": "An image file is required."}), 400

    img = _decode_image(file)
    if img is None:
        return jsonify({"ok": False, "error": "Could not read the image file."}), 400

    # Normalised JPEG bytes (strips EXIF, validates the upload).
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return jsonify({"ok": False, "error": "Could not encode the image."}), 500
    jpeg_bytes = buf.tobytes()

    # ── Supabase mode ─────────────────────────────────────────────────────────
    if supabase_store.is_enabled():
        try:
            res = supabase_store.upsert_location(gender, _clean_location_label(location_name),
                                                 jpeg_bytes)
            return jsonify({"ok": True, "gender": gender, **res})
        except Exception as e:
            return jsonify({"ok": False, "error": f"Supabase store failed: {e}"}), 502

    # ── Local filesystem fallback ─────────────────────────────────────────────
    folder = _resolve_or_create_location(gender, location_name, create=True)
    if folder is None:
        return jsonify({"ok": False, "error": "Could not create the location."}), 500
    folder_path = os.path.join(LOCATIONS_DIR, gender, folder)

    # Remove any existing image(s) so each location holds exactly one photo.
    for f in os.listdir(folder_path):
        if f.lower().endswith(_IMG_EXTS):
            try:
                os.remove(os.path.join(folder_path, f))
            except OSError:
                pass

    with open(os.path.join(folder_path, "image.jpg"), "wb") as out:
        out.write(jpeg_bytes)

    return jsonify({"ok": True, "gender": gender, "folder": folder,
                    "label": _clean_location_label(folder)})


@app.route("/api/admin/location/delete", methods=["POST"])
def api_admin_delete_location():
    """Primary-admin only: delete a whole location for a gender."""
    _email, _ok = _admin_identity(request)
    if not _ok:
        return jsonify({"ok": False, "error": "Unauthorised — owner login required."}), 401
    if not _is_primary(_email):
        return jsonify({"ok": False, "error":
            "Only the Primary Admin can delete locations."}), 403

    data     = request.get_json(silent=True) or request.form
    gender   = (data.get("gender") or "").strip().capitalize()
    location = (data.get("location") or "").strip()          # folder name
    if gender not in VALID_GENDERS or not location:
        return jsonify({"ok": False, "error": "gender and location are required."}), 400
    if ".." in location or "/" in location or "\\" in location:
        return jsonify({"ok": False, "error": "Invalid location."}), 400

    if supabase_store.is_enabled():
        try:
            supabase_store.delete_location(gender, location)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": f"Supabase delete failed: {e}"}), 502

    folder_path = os.path.join(LOCATIONS_DIR, gender, location)
    if not os.path.isdir(folder_path):
        return jsonify({"ok": False, "error": "Location not found."}), 404

    import shutil
    shutil.rmtree(folder_path, ignore_errors=True)
    return jsonify({"ok": True})


@app.route("/api/admin/location/rename", methods=["POST"])
def api_admin_rename_location():
    """Owner-only: rename a location (keeps its numeric prefix + its photo)."""
    if not _check_admin(request):
        return jsonify({"ok": False, "error": "Unauthorised — owner login required."}), 401

    data     = request.get_json(silent=True) or request.form
    gender   = (data.get("gender") or "").strip().capitalize()
    location = (data.get("location") or "").strip()          # current folder name
    new_name = (data.get("new_name") or "").strip()
    if gender not in VALID_GENDERS or not location or not new_name:
        return jsonify({"ok": False, "error": "gender, location and new_name are required."}), 400
    for v in (location, new_name):
        if ".." in v or "/" in v or "\\" in v:
            return jsonify({"ok": False, "error": "Invalid name."}), 400

    if supabase_store.is_enabled():
        try:
            res = supabase_store.rename_location(gender, location,
                                                _clean_location_label(new_name))
            return jsonify({"ok": True, **res})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 502

    gender_dir = os.path.join(LOCATIONS_DIR, gender)
    src = os.path.join(gender_dir, location)
    if not os.path.isdir(src):
        return jsonify({"ok": False, "error": "Location not found."}), 404

    # Keep the existing numeric prefix if there was one.
    m = re.match(r"^\s*(\d+)\.", location)
    prefix = f"{m.group(1)}. " if m else ""
    new_folder = f"{prefix}{_clean_location_label(new_name)}"
    dst = os.path.join(gender_dir, new_folder)
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.exists(dst):
            return jsonify({"ok": False, "error": "A location with that name already exists."}), 409
        os.rename(src, dst)
    return jsonify({"ok": True, "folder": new_folder, "label": _clean_location_label(new_folder)})


@app.route("/api/admin/location/message", methods=["POST"])
def api_admin_set_message():
    """
    Owner (any admin) sets/clears a location's custom result message. The message
    may contain {name} and {location} placeholders, filled in on the result.
    """
    if not _check_admin(request):
        return jsonify({"ok": False, "error": "Unauthorised - owner login required."}), 401

    data     = request.get_json(silent=True) or request.form
    gender   = (data.get("gender") or "").strip().capitalize()
    location = (data.get("location") or "").strip()      # folder/name
    message  = (data.get("message") or "").strip()
    if gender not in VALID_GENDERS or not location:
        return jsonify({"ok": False, "error": "gender and location are required."}), 400
    if ".." in location or "/" in location or "\\" in location:
        return jsonify({"ok": False, "error": "Invalid location."}), 400

    if supabase_store.is_enabled():
        try:
            supabase_store.set_message(gender, location, message)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 502

    # FS fallback: store message.txt alongside the image.
    folder_path = os.path.join(LOCATIONS_DIR, gender, location)
    if not os.path.isdir(folder_path):
        return jsonify({"ok": False, "error": "Location not found."}), 404
    msg_path = os.path.join(folder_path, "message.txt")
    try:
        if message:
            with open(msg_path, "w", encoding="utf-8") as f:
                f.write(message)
        elif os.path.exists(msg_path):
            os.remove(msg_path)
        return jsonify({"ok": True})
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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

            if supabase_store.is_enabled():
                data = supabase_store.get_image_bytes(gender, location)
                if not data:
                    return jsonify({"ok": False, "error":
                        f"No photo is available yet for {gender} · "
                        f"{_clean_location_label(location)}. Please pick another location."}), 404
                target = _decode_image(io.BytesIO(data))
            else:
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
        # Working resolution: 1536 on GPU (sharper composite + better landmarks;
        # the RTX-class card handles it easily) and 1024 on CPU to stay fast.
        # InsightFace still swaps at 128px and GFPGAN restores 512px crops; the
        # larger canvas mainly helps the final blend + the RealESRGAN 4x upscale.
        from core.super_res import _device as _sr_device
        _work_res = 1536 if _sr_device() == "cuda" else 1024
        source = resize_keep_aspect(source, _work_res)
        target = resize_keep_aspect(target, _work_res)

        # Close-up selfie? A face that fills the whole frame defeats the detector
        # (0 faces -> crude paste -> no real swap/hair/skin). Pad a margin so the
        # face is found. The source is only read for its landmarks, so the border
        # never appears in the output (which is drawn on the target canvas).
        source = pad_until_detectable(source)

        # De-roll a tilted source selfie so the eyes are level. Without this,
        # InsightFace misses faces rolled >~15deg and the swap silently falls
        # back to a crude paste. The swapper then aligns the upright source to
        # the target's pose as usual.
        source = align_face_upright(source)

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

        # Default pipeline = the clean, high-quality FACE swap (this is what
        # produced the good results). HEAD swap and HAIR swap are OPT-IN only
        # (?head_swap=1 / ?swap_hair=1) because, with the current models, they
        # degrade the clean face swap more often than they help.
        #
        # 1. (optional) HEAD SWAP — transplant the source head shape+hair.
        base = target
        if request.form.get("head_swap", "0") in ("1", "true", "on"):
            try:
                hs = full_head_swap(source, target)
                if hs is not None:
                    base = hs
            except Exception as e:
                print(f"[swap] head swap error: {e}")

        # 2. FACE SWAP — the main event. Source identity onto the main face only
        #    (background/poster faces are never swapped).
        swapped = swap_face_insightface(source, base)

        # 3. GFPGAN face restoration — recovers detail lost in the 128px swap.
        swapped = restore_faces(swapped)

        # 4. (skin-tone match moved to the very END of the pipeline — see below.
        #    Doing it here let the Laplacian blend in step 6 pull the face colour
        #    back toward the target, undoing the match. It must be the last word.)

        # 5. HAIR + NECK — transfer the SOURCE's hair onto the result. HairFastGAN
        #    generates the source's hairstyle on an aligned portrait; swap_hair now
        #    composites the FULL long hair back into the scene (the parse crop +
        #    region clamp were widened so long hair is no longer cut off). ON by
        #    default (set swap_hair=0 for a fast face-only swap).
        if request.form.get("swap_hair", "1") in ("1", "true", "on"):
            hf_portrait = None
            try:
                hf_portrait = transfer_hair(face_bgr=swapped, shape_bgr=source,
                                            color_bgr=source)
            except Exception as e:
                print(f"[swap] HairFastGAN error: {e}")
            try:
                if hf_portrait is not None:
                    pad = int(max(hf_portrait.shape[:2]) * 0.4)
                    hf_padded = cv2.copyMakeBorder(hf_portrait, pad, pad, pad, pad,
                                                   cv2.BORDER_CONSTANT, value=(127, 127, 127))
                    swapped = swap_hair(swapped, hf_padded, swapped, include_face=False)
                else:
                    swapped = swap_hair(swapped, source, target, include_face=False)
            except Exception as e:
                print(f"[swap] hair compose error: {e}")

        # 6. Glasses (no-op if the source isn't wearing any).
        if request.form.get("keep_glasses", "1") in ("1", "true", "on"):
            try:
                swapped = transfer_glasses(swapped, source)
            except Exception as e:
                print(f"[swap] glasses transfer error: {e}")

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

        # 7. SKIN TONE — the FINAL step, so nothing can override it. Drive ALL
        #    visible skin (face + neck + arms + hands) to the SOURCE complexion;
        #    clothes/background are excluded. Done AFTER the Laplacian blend so the
        #    blend can't pull the face colour back toward the target's tone.
        try:
            swapped = match_skin_to_source(
                swapped, source, faces_src[0], faces_tgt[0], strength=0.92
            )
        except Exception as e:
            print(f"[swap] skin tone match skipped: {e}")

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
