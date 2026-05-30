"""
Hair / head transfer on top of an InsightFace face swap.

InsightFace only swaps the inner face — it keeps the target's hair. This module
transplants the SOURCE person's hair (and optionally forehead/ears) onto the
result so the whole head reads as the source.

Approach (no extra heavy models beyond facexlib, which GFPGAN already pulls in):
  1. Detect 5-point landmarks in source and target (InsightFace).
  2. Fit a similarity transform (rotation + uniform scale + translation) mapping
     the source face onto the target face position.
  3. Parse the source head with BiSeNet (facexlib) to get a hair (+head) mask.
  4. Warp the source image and that mask into the target frame with the same
     transform, feather the edge, and composite it over the face-swapped result.

This is pose-sensitive: it works best when source and target are roughly the
same orientation. It is exposed as an opt-in step, never the default.
"""
import cv2
import numpy as np

from .detector import _get_insightface

# CelebAMask-HQ / facexlib BiSeNet class ids (verified empirically):
#   1 skin  2 l_brow 3 r_brow 4 l_eye 5 r_eye 6 eye_g 7 l_ear 8 r_ear
#   9 ear_r 10 nose 11 mouth 12 u_lip 13 l_lip 14 neck 15 neck_l
#   16 cloth 17 hair 18 hat
_HAIR_CLASSES = {17}                           # hair only (hat 18 excluded — it
                                               # misfires on bright backgrounds)
_HEAD_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17}
# Face skin + ears + neck (NOT eyes/glasses/hair/cloth) — the visible skin whose
# tone must stay consistent so the swap doesn't look pasted at the jaw.
_SKIN_NECK_CLASSES = {1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15}
_GLASSES_CLASSES = {6}                         # eye_g (spectacles)

_parser = None
_parser_failed = False


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_parser():
    global _parser, _parser_failed
    if _parser is not None:
        return _parser
    if _parser_failed:
        return None
    try:
        from facexlib.parsing import init_parsing_model
        _parser = init_parsing_model(model_name="bisenet", device=_device())
        _parser.eval()
        print(f"[head_swap] BiSeNet parser loaded ({_device()})")
        return _parser
    except Exception as e:
        print(f"[head_swap] parser load failed: {e}")
        _parser_failed = True
        return None


def _parse_region_mask(image, bbox, classes,
                       up=0.9, down=0.4, side=0.5) -> np.ndarray:
    """
    Return a full-image float mask (0..1) of the requested parse classes for the
    head in bbox. The crop (bbox expanded by up/down/side fractions of its size)
    is parsed at 512px and mapped back to image coordinates.
    """
    import torch

    parser = _get_parser()
    h, w = image.shape[:2]
    full = np.zeros((h, w), np.float32)
    if parser is None:
        return full

    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    ex1 = max(0, x1 - int(bw * side))
    ey1 = max(0, y1 - int(bh * up))
    ex2 = min(w, x2 + int(bw * side))
    ey2 = min(h, y2 + int(bh * down))
    crop = image[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        return full

    inp = cv2.resize(crop, (512, 512))
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(_device())
    with torch.no_grad():
        seg = parser(t)[0].argmax(1).squeeze().cpu().numpy().astype(np.uint8)

    mask512 = np.isin(seg, list(classes)).astype(np.float32)
    mask_crop = cv2.resize(mask512, (ex2 - ex1, ey2 - ey1),
                           interpolation=cv2.INTER_LINEAR)
    full[ey1:ey2, ex1:ex2] = mask_crop
    return full


def swap_hair(
    swapped: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    include_face: bool = False,
    feather: float = 0.04,
) -> np.ndarray:
    """
    Transplant the source's hair onto the face-swapped result.

    swapped: the InsightFace face-swap output (source face on target).
    source/target: the original images (needed for landmarks + parsing).
    include_face: also transfer source forehead/skin/ears (fuller head swap);
                  leave False to move hair only and keep the crisp swapped face.

    Returns the composited image, or `swapped` unchanged if anything is missing.
    """
    app = _get_insightface()
    if app is None:
        return swapped

    try:
        src_faces = app.get(source)
        tgt_faces = app.get(target)
        if not src_faces or not tgt_faces:
            print("[head_swap] face(s) not detected — skipping hair swap")
            return swapped

        src_face = max(src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        tgt_face = max(tgt_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        src_kps = np.asarray(src_face.kps, dtype=np.float32)
        tgt_kps = np.asarray(tgt_face.kps, dtype=np.float32)

        # Similarity transform: source coords -> target coords.
        M, _ = cv2.estimateAffinePartial2D(src_kps, tgt_kps, method=cv2.LMEDS)
        if M is None:
            print("[head_swap] could not fit transform — skipping")
            return swapped

        classes = _HEAD_CLASSES if include_face else _HAIR_CLASSES
        src_mask = _parse_region_mask(source, src_face.bbox, classes,
                                      up=0.9, down=0.4, side=0.5)
        if src_mask.max() <= 0:
            print("[head_swap] empty source hair mask — skipping")
            return swapped

        # Keep only the largest connected region — drops stray patches the
        # parser sometimes marks on a noisy/low-contrast background.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (src_mask > 0.5).astype(np.uint8), connectivity=8
        )
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            src_mask = (labels == biggest).astype(np.float32)

        h, w = target.shape[:2]
        warped_src  = cv2.warpAffine(source, M, (w, h), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(src_mask, M, (w, h), flags=cv2.INTER_LINEAR)

        # Clamp the transferred hair to a plausible head region around the target
        # face, so a size/pose mismatch can't leave hair floating in the sky.
        # The box is generous downward and sideways so LONG hair (e.g. female
        # styles past the shoulders) is preserved — only far-above/far-aside
        # stray regions are cut.
        tx1, ty1, tx2, ty2 = [int(v) for v in tgt_face.bbox]
        tbw, tbh = tx2 - tx1, ty2 - ty1
        region = np.zeros((h, w), np.float32)
        rx1 = max(0, tx1 - int(tbw * 1.2));  ry1 = max(0, ty1 - int(tbh * 1.4))
        rx2 = min(w, tx2 + int(tbw * 1.2));  ry2 = min(h, ty2 + int(tbh * 3.0))
        region[ry1:ry2, rx1:rx2] = 1.0
        warped_mask *= region

        # Tighten the mask BEFORE feathering so the soft edge stays *inside* the
        # hair — otherwise the Gaussian expands it outward and pulls the source's
        # background in past the hairline (a bright halo above the head).
        warped_mask = (warped_mask > 0.5).astype(np.float32)
        erode_px = max(2, int(min(h, w) * 0.015))
        warped_mask = cv2.erode(warped_mask, np.ones((erode_px, erode_px), np.uint8))
        k = max(3, int(min(h, w) * feather) | 1)        # odd kernel
        warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)
        alpha = np.stack([warped_mask] * 3, axis=-1)

        result = (warped_src.astype(np.float32) * alpha +
                  swapped.astype(np.float32) * (1.0 - alpha))
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[head_swap] hair swap error: {e}")
        return swapped


def match_skin_to_source(
    swapped: np.ndarray,
    source: np.ndarray,
    src_bbox,
    tgt_bbox,
    strength: float = 0.85,
) -> np.ndarray:
    """
    Recolour the whole visible skin region — face AND neck — toward the source
    complexion so there is no tone seam at the jaw (the "pasted face" look).

    The shift is one uniform LAB offset (source central-face mean minus the
    swapped central-face mean) applied across a BiSeNet skin+neck mask, feathered
    at the edge. Because both the face and neck receive the same offset, they end
    up the same complexion; the mask stops at the collar (cloth is excluded), so
    the transition to clothing is hidden. Falls back to the face-only ellipse
    matcher when the parser/mask is unavailable.
    """
    from .skin_tone import _central_mean_lab, match_face_to_source_tone

    src_mean = _central_mean_lab(source, src_bbox)
    swp_mean = _central_mean_lab(swapped, tgt_bbox)
    if src_mean is None or swp_mean is None:
        return swapped

    # Skin + neck mask on the swapped image (target geometry). Expand the parse
    # box well below the chin so the neck is captured.
    mask = _parse_region_mask(swapped, tgt_bbox, _SKIN_NECK_CLASSES,
                              up=0.3, down=1.8, side=0.7)
    if mask.max() <= 0:
        # No parse — fall back to recolouring just the face oval.
        return match_face_to_source_tone(swapped, source, src_bbox, tgt_bbox,
                                         strength=strength)

    h, w = swapped.shape[:2]
    k = max(3, int(min(h, w) * 0.03) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.clip(mask, 0.0, 1.0)

    delta = (src_mean - swp_mean) * strength
    lab = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        lab[:, :, c] += delta[c] * mask
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _similarity_from_2pts(src, dst) -> np.ndarray:
    """2x3 similarity (rotation+uniform scale+translation) mapping src->dst eyes."""
    src = np.asarray(src, np.float32)
    dst = np.asarray(dst, np.float32)
    sv = src[1] - src[0]
    dv = dst[1] - dst[0]
    scale = float(np.linalg.norm(dv)) / (float(np.linalg.norm(sv)) + 1e-6)
    ang = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    R = np.array([[c, -s], [s, c]], np.float32)
    t = dst.mean(0) - R @ src.mean(0)
    return np.array([[R[0, 0], R[0, 1], t[0]],
                     [R[1, 0], R[1, 1], t[1]]], np.float32)


def transfer_glasses(swapped: np.ndarray, source: np.ndarray,
                     feather: float = 0.008) -> np.ndarray:
    """
    Carry the SOURCE's spectacles onto the swapped face (InsightFace swaps the
    face but not accessories). Segments them with BiSeNet (class 6 = eye_g) and:
      - aligns by the two EYE keypoints so the glasses sit exactly on the eyes;
      - composites frames solid but LENSES semi-transparent (opacity follows how
        dark/frame-like each pixel is), so the target's own eyes show through the
        lenses instead of the source's — far less "pasted".
    No-op when the source wears no glasses.
    """
    app = _get_insightface()
    if app is None:
        return swapped
    try:
        src_faces = app.get(source)
        tgt_faces = app.get(swapped)
        if not src_faces or not tgt_faces:
            return swapped
        area = lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        sf = max(src_faces, key=area)
        tf = max(tgt_faces, key=area)

        gmask = _parse_region_mask(source, sf.bbox, _GLASSES_CLASSES,
                                   up=0.5, down=0.5, side=0.5)
        if gmask.max() <= 0:
            return swapped  # source has no glasses

        # Lock the glasses to the eye line (kps[0]=left eye, kps[1]=right eye).
        M = _similarity_from_2pts(sf.kps[:2], tf.kps[:2])

        h, w = swapped.shape[:2]
        warped_src = cv2.warpAffine(source, M, (w, h), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(gmask, M, (w, h), flags=cv2.INTER_LINEAR)
        warped_mask = (warped_mask > 0.4).astype(np.float32)

        # Per-pixel opacity: dark/coloured frame pixels ~opaque, bright lens
        # pixels ~half — so the underlying (target) eyes read through the tint.
        gray = cv2.cvtColor(warped_src, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        opacity = 0.45 + 0.55 * np.clip(1.0 - gray, 0.0, 1.0)

        k = max(3, int(min(h, w) * feather) | 1)
        warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)
        alpha = np.clip(warped_mask * opacity, 0.0, 1.0)
        alpha3 = np.stack([alpha] * 3, axis=-1)

        out = (warped_src.astype(np.float32) * alpha3 +
               swapped.astype(np.float32) * (1.0 - alpha3))
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[head_swap] glasses transfer error: {e}")
        return swapped
