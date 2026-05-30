"""
High-quality hair transfer with HairFastGAN (StyleGAN-based).

InsightFace swaps only the face; this transfers a reference hairstyle. It prefers
a LOCAL GPU install (vendored under external/HairFastGAN, run via its MSVC-env
launcher) and falls back to the hosted HairFastGAN Gradio Space. Set
HAIRFAST_LOCAL=0 to force the Space, or HAIRFAST_SPACE to your own duplicated
(HF Pro GPU) Space; HF_TOKEN lifts the public Space's ZeroGPU quota.

Either backend returns an FFHQ-aligned 1024 portrait (face + new hair, no
background), so callers paste the head back into the original scene themselves.
"""
import os
import tempfile

import cv2
import numpy as np

# Some environments (conda) set SSL_CERT_FILE to a path that doesn't exist,
# which makes httpx/gradio_client crash. Repair it from certifi before use.
_cf = os.environ.get("SSL_CERT_FILE")
if not _cf or not os.path.exists(_cf):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass

HAIRFAST_SPACE = os.environ.get("HAIRFAST_SPACE", "AIRI-Institute/HairFastGAN")

# Local GPU HairFastGAN (vendored under external/HairFastGAN). When present with
# a CUDA GPU we run it via the MSVC-env launcher instead of the hosted Space —
# faster, private, no quota. Set HAIRFAST_LOCAL=0 to force the Space.
_HF_LOCAL_DIR = os.path.join("external", "HairFastGAN")
_HF_LOCAL_BAT = os.path.join(_HF_LOCAL_DIR, "run_hairfast.bat")

_client = None
_client_failed = False
_local_ok = None  # cached availability check


def _local_available() -> bool:
    global _local_ok
    if _local_ok is not None:
        return _local_ok
    _local_ok = False
    if os.environ.get("HAIRFAST_LOCAL", "1") not in ("1", "true", "on"):
        return False
    # Need the vendored repo with weights linked in (run scripts/setup_hairfast.py).
    if not os.path.isdir(os.path.join(_HF_LOCAL_DIR, "pretrained_models")):
        return False
    # Self-heal: drop the committed runner scripts into the repo if missing.
    if not os.path.isfile(_HF_LOCAL_BAT):
        src = os.path.join("scripts", "hairfast")
        try:
            import shutil
            for f in ("run_hairfast.py", "run_hairfast.bat"):
                s = os.path.join(src, f)
                if os.path.isfile(s):
                    shutil.copy(s, os.path.join(_HF_LOCAL_DIR, f))
        except Exception:
            pass
    if not os.path.isfile(_HF_LOCAL_BAT):
        return False
    try:
        import torch
        _local_ok = torch.cuda.is_available()
    except Exception:
        _local_ok = False
    if _local_ok:
        print("[hair_transfer] local GPU HairFastGAN available")
    return _local_ok


def _transfer_hair_local(face_bgr, shape_bgr, color_bgr):
    """Run the vendored HairFastGAN on the local GPU via its MSVC-env launcher."""
    import subprocess
    work = tempfile.mkdtemp(prefix="hairfast_")
    fp = os.path.join(work, "face.png")
    sp = os.path.join(work, "shape.png")
    cp = os.path.join(work, "color.png")
    op = os.path.join(work, "out.png")
    cv2.imwrite(fp, face_bgr)
    cv2.imwrite(sp, shape_bgr)
    cv2.imwrite(cp, color_bgr)
    try:
        # The .bat sets the VS build env + CUDA arch, cds into the repo, and runs
        # run_hairfast.py with absolute paths. ~18s warm; first ever call is slow
        # (one-time op compile + model downloads), hence the generous timeout.
        r = subprocess.run(
            ["cmd", "/c", os.path.abspath(_HF_LOCAL_BAT), fp, sp, cp, op],
            capture_output=True, text=True, timeout=900,
        )
        if os.path.exists(op):
            return cv2.imread(op)
        print(f"[hair_transfer] local run produced no output:\n{r.stdout[-500:]}\n{r.stderr[-500:]}")
        return None
    except subprocess.TimeoutExpired:
        print("[hair_transfer] local run timed out")
        return None
    except Exception as e:
        print(f"[hair_transfer] local run error: {e}")
        return None
    finally:
        for p in (fp, sp, cp, op):
            try:
                os.remove(p)
            except Exception:
                pass
        try:
            os.rmdir(work)
        except Exception:
            pass


def _get_client():
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        return None
    try:
        from gradio_client import Client
        # An HF token lifts the anonymous ZeroGPU quota that makes the public
        # Space error on every call. Required in practice; point HAIRFAST_SPACE
        # at your own duplicated (Pro GPU) Space for reliability.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        _client = Client(HAIRFAST_SPACE, token=token, verbose=False)
        print(f"[hair_transfer] connected to Space: {HAIRFAST_SPACE}"
              f"{' (authenticated)' if token else ' (anonymous — may hit GPU quota)'}")
        return _client
    except Exception as e:
        print(f"[hair_transfer] could not connect to {HAIRFAST_SPACE}: {e}")
        _client_failed = True
        return None


def _write_tmp(img_bgr: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img_bgr)
    return path


def transfer_hair(face_bgr: np.ndarray,
                  shape_bgr: np.ndarray,
                  color_bgr=None):   # np.ndarray | None
    """
    Return an FFHQ-aligned 1024 portrait: the `face` identity wearing the
    hairstyle (shape) and hair colour (color) of the references. Returns a BGR
    image, or None on any failure (caller should fall back gracefully).

    face_bgr : whose face/identity to keep (e.g. the InsightFace swap result)
    shape_bgr: hairstyle-shape reference (e.g. the source person)
    color_bgr: hair-colour reference (defaults to shape)

    Uses the local GPU HairFastGAN when available; otherwise the hosted Space.
    """
    if color_bgr is None:
        color_bgr = shape_bgr

    # Prefer the local GPU model (faster, private, no quota).
    if _local_available():
        res = _transfer_hair_local(face_bgr, shape_bgr, color_bgr)
        if res is not None:
            return res
        print("[hair_transfer] local run failed — falling back to Space")

    client = _get_client()
    if client is None:
        return None

    from gradio_client import handle_file
    paths = []
    try:
        fp = _write_tmp(face_bgr);  paths.append(fp)
        sp = _write_tmp(shape_bgr); paths.append(sp)
        cp = _write_tmp(color_bgr); paths.append(cp)

        # 1. Align each photo to FFHQ (the Space's preprocessing step).
        align = ["Face", "Shape", "Color"]
        fa = client.predict(img=handle_file(fp), align=align, api_name="/resize_inner")
        sa = client.predict(img=handle_file(sp), align=align, api_name="/resize_inner_1")
        ca = client.predict(img=handle_file(cp), align=align, api_name="/resize_inner_2")

        # 2. Swap the hair.
        out = client.predict(
            face=handle_file(fa),
            shape=handle_file(sa),
            color=handle_file(ca),
            blending="Article",
            poisson_iters=0,
            poisson_erosion=15,
            api_name="/swap_hair",
        )
        result_path = out[0] if isinstance(out, (list, tuple)) else out
        err = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else ""
        if err:
            print(f"[hair_transfer] Space reported: {err}")
        if not result_path or not os.path.exists(result_path):
            print("[hair_transfer] no result image returned")
            return None
        res = cv2.imread(result_path)
        return res
    except Exception as e:
        print(f"[hair_transfer] transfer failed: {e}")
        return None
    finally:
        for p in paths:
            try:
                os.remove(p)
            except Exception:
                pass
