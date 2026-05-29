"""
High-quality hair transfer via the HairFastGAN Gradio Space.

InsightFace swaps only the face; this calls a hosted HairFastGAN Space (StyleGAN-
based) to transfer a reference hairstyle, which runs on GPU server-side. Set the
HAIRFAST_SPACE env var to your own duplicated (HF Pro GPU) Space for speed and
privacy; it defaults to the public AIRI-Institute Space.

The Space returns an FFHQ-aligned 1024 portrait (face + new hair, no background),
so callers paste the head back into the original scene themselves.
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

_client = None
_client_failed = False


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
    """
    client = _get_client()
    if client is None:
        return None
    if color_bgr is None:
        color_bgr = shape_bgr

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
