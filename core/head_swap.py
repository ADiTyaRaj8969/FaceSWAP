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
_FACE_SKIN_CLASSES = {1, 2, 3, 7, 8, 9, 10, 11, 12, 13}   # face skin/ears/nose/mouth
_FACE_SKIN_ONLY = {1, 7, 8, 9, 10}        # skin/ears/nose ONLY — recolour these;
                                          # brows/eyes/lips excluded (keep colour)
_NECK_ONLY_CLASSES = {14, 15}                              # neck region
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


def _soft_hair_matte(image, bbox, up=1.0, down=2.6, side=1.4) -> np.ndarray:
    """
    Soft 0..1 hair matte from BiSeNet's SOFTMAX probability for the hair class
    (class 17) — NOT the hard argmax mask. The probability fades gradually across
    the individual strands at the hair's outline, so compositing with it gives a
    wispy, real-hair edge instead of a cut-out mask boundary. Returns a full-image
    float map in the input image's coordinates.
    """
    import torch
    parser = _get_parser()
    h, w = image.shape[:2]
    full = np.zeros((h, w), np.float32)
    if parser is None:
        return full
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1

    def _matte(up_, down_, side_):
        ex1 = max(0, x1 - int(bw * side_)); ey1 = max(0, y1 - int(bh * up_))
        ex2 = min(w, x2 + int(bw * side_)); ey2 = min(h, y2 + int(bh * down_))
        crop = image[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            return None
        inp = cv2.resize(crop, (512, 512))
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(_device())
        with torch.no_grad():
            prob = torch.softmax(parser(t)[0], dim=1)[0, 17].cpu().numpy().astype(np.float32)
        out = np.zeros((h, w), np.float32)
        out[ey1:ey2, ex1:ex2] = cv2.resize(prob, (ex2 - ex1, ey2 - ey1))
        return out

    # The wide crop exists so long hair isn't cut off before parsing, but it
    # also shrinks the head inside the 512x512 parse input, and BiSeNet's hair
    # confidence falls with it. Measured on a 66x77 face: the wide crop peaks at
    # 0.341, leaving NOTHING above the 0.4 that swap_hair thresholds at, so the
    # hair mask came out empty and the transfer silently did nothing. So keep
    # the wide crop when it is confident, and tighten only when it is not.
    for factor in (1.0, 0.6, 0.38, 0.25):
        m = _matte(up * max(factor, 0.6), down * factor, side * factor)
        if m is None:
            continue
        full = m
        if m.max() >= 0.5:
            break
    return full


def swap_hair(
    swapped: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    include_face: bool = False,
    feather: float = 0.025,
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

        # SOURCE hair as a SOFT matte (real wispy strand edges) for the hair case;
        # a binary head mask for the full-head-swap case. Generous crop either way
        # so long hair isn't cut before parsing.
        if include_face:
            src_mask = _parse_region_mask(source, src_face.bbox, _HEAD_CLASSES,
                                          up=0.9, down=0.4, side=0.5)
            soft = False
        else:
            src_mask = _soft_hair_matte(source, src_face.bbox,
                                        up=1.0, down=2.6, side=1.4)
            soft = True
        if src_mask.max() <= 0:
            print("[head_swap] empty source hair mask — skipping")
            return swapped

        # Keep only the largest connected region (drop stray bg patches). For the
        # soft matte, keep the SOFT values inside that region — don't binarise, or
        # we'd lose the strand gradient that makes the edge look real.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (src_mask > (0.4 if soft else 0.5)).astype(np.uint8), connectivity=8)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep = (labels == biggest).astype(np.float32)
            src_mask = src_mask * keep if soft else keep

        h, w = target.shape[:2]
        warped_src  = cv2.warpAffine(source, M, (w, h), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(src_mask, M, (w, h), flags=cv2.INTER_LINEAR)

        # Clamp to a plausible head box around the target face (generous down/side
        # for long hair) so a pose/size mismatch can't leave hair in the sky.
        tx1, ty1, tx2, ty2 = [int(v) for v in tgt_face.bbox]
        tbw, tbh = tx2 - tx1, ty2 - ty1
        region = np.zeros((h, w), np.float32)
        rx1 = max(0, tx1 - int(tbw * 2.2));  ry1 = max(0, ty1 - int(tbh * 1.6))
        rx2 = min(w, tx2 + int(tbw * 2.2));  ry2 = min(h, ty2 + int(tbh * 5.0))
        region[ry1:ry2, rx1:rx2] = 1.0
        warped_mask *= region

        # KILL THE GREY HALO/HAZE: replace everything outside the hair with the
        # SCENE so the soft strand edge blends hair -> scene, not hair -> the
        # portrait's grey padding. The threshold matters: the soft matte has a
        # faint tail (low probability) ABOVE/around the hair; if we keep the
        # portrait's grey hair-edge pixels there, that tail composites as a grey
        # haze floating over the head. Using a higher threshold (0.3) makes that
        # tail scene-over-scene = invisible, so only the confident strands remain.
        thr = 0.3 if soft else 0.5
        hair_region = cv2.dilate((warped_mask > thr).astype(np.uint8),
                                 np.ones((3, 3), np.uint8))[..., None] > 0
        warped_src = np.where(hair_region, warped_src, swapped)

        if soft:
            # REAL-HAIR alpha: the softmax matte already fades across strands. Boost
            # the interior to fully opaque (no forehead show-through) while the
            # low-probability strand edge stays semi-transparent — a natural wispy
            # boundary. A tiny blur removes 512px parser blockiness.
            base_matte = np.clip(warped_mask, 0.0, 1.0)
            am = np.clip(base_matte * 1.6, 0.0, 1.0)
            am = cv2.GaussianBlur(am, (3, 3), 0)
            # SOFT strand edge: keep the matte's natural falloff, but bound it to a
            # modest margin beyond the confident hair so a far faint tail can't haze.
            # (No guided filter — it spread the haze and isn't in HF's headless
            # OpenCV, so local and HF stay identical.)
            mg = max(3, int(min(h, w) * 0.022))
            extent = cv2.dilate((base_matte > 0.4).astype(np.uint8),
                                np.ones((mg, mg), np.uint8)).astype(np.float32)
            extent = cv2.GaussianBlur(extent, (mg | 1, mg | 1), 0)
            am = am * np.clip(extent, 0.0, 1.0)
            # Kill the HairFastGAN portrait's GREY thin-crown tail by COLOUR, not by
            # cropping (which hardened the edge). In the soft edge only, attenuate
            # pixels that are desaturated mid-grey (≈ the portrait's grey padding) —
            # that removes the grey haze while leaving hair-coloured strands (the raw
            # -selfie path, dark or blond) fully soft. The website grey line is gone
            # AND the edge stays wispy.
            hsv = cv2.cvtColor(np.clip(warped_src, 0, 255).astype(np.uint8),
                               cv2.COLOR_BGR2HSV)
            grey = ((hsv[:, :, 1] < 32) & (hsv[:, :, 2] > 105) &
                    (hsv[:, :, 2] < 160)).astype(np.float32)
            # Only the VERY faint edge (am < 0.45) — so we trim the portrait's grey
            # padding tail but never cut into semi-solid hair (lit/highlighted or
            # grey-streaked real hair has a higher matte and must be kept).
            soft_zone = (am < 0.45).astype(np.float32)
            am = am * (1.0 - 0.85 * grey * soft_zone)
            warped_mask = np.clip(cv2.GaussianBlur(am, (3, 3), 0), 0.0, 1.0)
        else:
            # Head-swap path: erode to solid hair, then a small feather.
            binm = (warped_mask > 0.5).astype(np.float32)
            er_px = max(2, int(min(h, w) * 0.005))
            solid = cv2.erode(binm, np.ones((er_px, er_px), np.uint8))
            k = max(3, int(min(h, w) * feather) | 1)
            warped_mask = np.clip(cv2.GaussianBlur(solid, (k, k), 0), 0.0, 1.0)
        alpha = np.stack([warped_mask] * 3, axis=-1)

        # Match the source hair's exposure to the target scene before compositing.
        # Without this, hair from a brightly-lit source photo looks like a cut-out
        # when pasted onto a dimmer target scene.
        warped_src = _match_lighting(warped_src, swapped, warped_mask)

        result = (warped_src.astype(np.float32) * alpha +
                  swapped.astype(np.float32) * (1.0 - alpha)).astype(np.float32)

        # Cover the target's OLD hair that the new hair didn't reach (a dark bun or
        # temples peeking out beside the new hair) by recolouring those residual
        # pixels toward the new hair's colour — so the head reads as ONE hairstyle
        # instead of the new hair plus a patch of the target's original hair.
        if not include_face:
            try:
                old_hair = _parse_region_mask(swapped, tgt_face.bbox, _HAIR_CLASSES,
                                              up=1.0, down=0.5, side=0.7)
                residual = np.clip(old_hair - warped_mask, 0.0, 1.0)
                sel_new = warped_mask > 0.6
                sel_res = residual > 0.5
                if int(sel_new.sum()) > 200 and int(sel_res.sum()) > 50:
                    lab_r = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8),
                                         cv2.COLOR_BGR2LAB).astype(np.float32)
                    new_mean = np.array([lab_r[:, :, c][sel_new].mean() for c in range(3)], np.float32)
                    res_mean = np.array([lab_r[:, :, c][sel_res].mean() for c in range(3)], np.float32)
                    dlt = new_mean - res_mean
                    rk = max(3, int(min(h, w) * 0.02) | 1)
                    rm = cv2.GaussianBlur(residual, (rk, rk), 0)
                    for c in range(3):
                        lab_r[:, :, c] += dlt[c] * rm
                    result = cv2.cvtColor(np.clip(lab_r, 0, 255).astype(np.uint8),
                                          cv2.COLOR_LAB2BGR).astype(np.float32)
            except Exception:
                pass
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[head_swap] hair swap error: {e}")
        return swapped


def _body_skin_mask(img: np.ndarray, seed_mask: np.ndarray,
                    face_bbox=None) -> np.ndarray:
    """
    Find ALL visible skin in the image (arms, hands, … not just the head) that
    matches the person's complexion, so the tone shift can cover the whole body.

    The face+neck `seed_mask` tells us this person's actual skin chroma; we then
    select every pixel whose Cr/Cb (colour, brightness-independent) is close to
    that seed. Clothes (a green polo, black trousers) sit far away in chroma and
    are naturally excluded — so we recolour skin only, never the shirt. The match
    is confined to a column around the person (`face_bbox`) so warm-coloured
    background (wood, skin-toned walls/reflections) can't be tinted.
    """
    seed = seed_mask > 0.5
    if seed.sum() < 300:
        return np.zeros(img.shape[:2], np.float32)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y  = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    cr_m, cb_m = cr[seed].mean(), cb[seed].mean()
    # Tolerance: a little wider than the seed's own spread (skin in shadow/light
    # drifts), but clamped so it can't bleed into clothing.
    cr_t = float(np.clip(cr[seed].std() * 2.5, 12.0, 22.0))
    cb_t = float(np.clip(cb[seed].std() * 2.5, 12.0, 22.0))

    mask = ((np.abs(cr - cr_m) < cr_t) &
            (np.abs(cb - cb_m) < cb_t) &
            (y > 45)).astype(np.uint8)                # drop near-black pixels

    h, w = img.shape[:2]
    # Confine to the person's column: a generous band centred on the face (wide
    # enough for arms reaching out) from just above the head to the image bottom.
    if face_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        fw, fh = x2 - x1, y2 - y1
        cx = (x1 + x2) // 2
        region = np.zeros((h, w), np.uint8)
        rx1 = max(0, cx - int(fw * 2.6))
        rx2 = min(w, cx + int(fw * 2.6))
        # Start at mid-face: the parser already handles the face+neck precisely,
        # and arms/hands are always BELOW — so beginning here adds the body while
        # excluding head-height background reflections beside the face.
        ry1 = max(0, y1 + int(fh * 0.5))
        region[ry1:h, rx1:rx2] = 1
        mask = mask * region

    # Clean specks + close small gaps so arms/hands come through as solid regions.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    return mask.astype(np.float32)


def _skin_lab_mean(img, bbox, classes, up=0.3, down=0.6, side=0.45):
    """Mean LAB of the PARSED skin (given classes) for the face in bbox, or None.
    Uses BiSeNet skin pixels — so it ignores lips/eyes/hair/background and gives a
    clean complexion sample even when the source wears makeup."""
    m = _parse_region_mask(img, bbox, classes, up=up, down=down, side=side)
    sel = m > 0.5
    if int(sel.sum()) < 80:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.array([lab[:, :, c][sel].mean() for c in range(3)], np.float32)


def match_skin_to_source(
    swapped: np.ndarray,
    source: np.ndarray,
    src_bbox,
    tgt_bbox,
    strength: float = 0.85,
    whole_body: bool = True,
) -> np.ndarray:
    """
    Shift ALL visible skin — face, neck, arms, hands — toward the SOURCE
    complexion with ONE uniform LAB offset over a single feathered skin mask.
    Clothes are untouched.

    Deliberately simple: NO per-region deltas and NO luminance gate. Both of those
    moved different skin pixels by different amounts (a side-lit cheek or a beard
    shifting differently from the rest), which is exactly what produced the
    blotches/patches. A single uniform offset moves every skin pixel by the same
    amount, so it changes the overall tone to the source's and can never create an
    internal patch — only a smooth, feathered edge where skin meets clothes/hair.
    The offset is measured from clean PARSED skin (lips/eyes/makeup excluded) on
    both the source and the swapped face, so it's an accurate complexion delta.
    """
    src_skin = _skin_lab_mean(source, src_bbox, _FACE_SKIN_ONLY,
                              up=0.3, down=0.5, side=0.4)
    swp_skin = _skin_lab_mean(swapped, tgt_bbox, _FACE_SKIN_ONLY,
                              up=0.3, down=0.6, side=0.45)
    if src_skin is None or swp_skin is None:
        return swapped

    h, w = swapped.shape[:2]
    # One skin mask: face skin + nose + lips + ears + neck (no interior holes) plus
    # the arms/hands. No holes => a uniform offset stays seamless.
    mask = _parse_region_mask(swapped, tgt_bbox, _SKIN_NECK_CLASSES,
                              up=0.3, down=1.8, side=0.7)
    if whole_body:
        mask = np.maximum(mask, _body_skin_mask(swapped, mask, face_bbox=tgt_bbox))
    if mask.max() <= 0:
        return swapped

    delta = (src_skin - swp_skin) * strength          # one offset for ALL skin
    k = max(3, int(min(h, w) * 0.04) | 1)
    m = cv2.GaussianBlur(np.clip(mask, 0.0, 1.0), (k, k), 0)
    lab = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        lab[:, :, c] += delta[c] * m

    # Gentle EXTRA pass for NECK + BODY only — nudge the neck/arms/hands the rest
    # of the way to the source complexion. CRITICAL: restrict to strictly BELOW the
    # chin. (The old version did mask - face_parse, which left the LIPS and a ring
    # of cheek/jaw in the region — recolouring them produced the discoloured facial
    # patches.) Zeroing everything above the chin guarantees the face is untouched.
    if whole_body:
        try:
            ty2 = int(tgt_bbox[3])
        except Exception:
            ty2 = int(tgt_bbox.bbox[3])
        nb = mask.copy()
        nb[:max(0, ty2), :] = 0.0                  # exclude the entire face
        sel = nb > 0.5
        if int(sel.sum()) > 100:
            nb_mean = np.array([lab[:, :, c][sel].mean() for c in range(3)], np.float32)
            extra = (src_skin - nb_mean) * 0.45
            mb = cv2.GaussianBlur(nb, (k, k), 0)
            for c in range(3):
                lab[:, :, c] += extra[c] * mb
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


def harmonize_to_scene(result, target, face_bbox, grain=1.0):
    """
    Make the swapped HEAD look PHOTOGRAPHED WITH the scene, not pasted on it.

    The GAN-restored face and the HairFastGAN hair come out unnaturally SMOOTH,
    while the target is a real photo with sensor grain + a colour cast. That
    mismatch is exactly what reads as 'synthetic hair' and 'pasted head'. We:
      1. estimate the scene's luminance grain and add a matching amount over the
         head region — this de-smooths the GAN hair/skin so it matches the photo;
      2. nudge the head's overall colour cast toward the scene's (gentle), so the
         white-balance/tone of the head agrees with the room.
    Confined to a feathered head box so the body/background are untouched.
    """
    h, w = result.shape[:2]
    res = result.astype(np.float32)

    # Grain must land on the HEAD, not on a rectangle around it. A box sprays
    # noise over whatever shares its bounds — background, shelves, clothing —
    # which dirties the scene and leaves the box's own feathered edge visible.
    # So parse the actual head/hair/neck shape and clamp it to a generous head
    # box, which keeps stray parse specks elsewhere from picking up grain.
    x1, y1, x2, y2 = [int(v) for v in face_bbox]
    bw, bh = x2 - x1, y2 - y1
    box = np.zeros((h, w), np.float32)
    rx1 = max(0, x1 - int(bw * 1.25)); ry1 = max(0, y1 - int(bh * 1.6))
    rx2 = min(w, x2 + int(bw * 1.25)); ry2 = min(h, y2 + int(bh * 3.0))
    box[ry1:ry2, rx1:rx2] = 1.0

    mask = None
    try:
        # Same generous crop as the hair matte so long hair is parsed, not cut.
        # Class 18 (hat) is included HERE only: the parser labels part of the
        # hair as hat, and while that misfire matters when transferring hair
        # (_HEAD_CLASSES rightly drops it), leaving it out here punches an
        # ungrained hole in the top of the head.
        parsed = _parse_region_mask(result, face_bbox,
                                    _HEAD_CLASSES | _NECK_ONLY_CLASSES | {18},
                                    up=1.0, down=2.6, side=1.4)
        if parsed.max() > 0:
            mask = parsed * box
    except Exception as e:
        print(f"[head_swap] harmonize parse failed, falling back to head box: {e}")
    if mask is None or mask.max() <= 0:
        mask = box

    # Small feather: enough to fade at the hairline, not so wide it bleeds the
    # grain back out into the scene the way the old box-sized blur did.
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.5, min(h, w) * 0.006))
    m3 = mask[..., None]

    # (1) match the scene's grain (estimated from the target's high-freq luminance)
    try:
        g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grain_std = float(np.clip(np.std(g - cv2.GaussianBlur(g, (0, 0), 1.6)),
                                  2.5, 8.0)) * grain
        noise = np.random.randn(h, w).astype(np.float32) * grain_std
        res = res + noise[..., None] * m3
    except Exception as e:
        print(f"[head_swap] harmonize grain skipped: {e}")

    # (2) gentle colour-cast harmonisation: shift the head's mean toward the
    # scene's — sampled OUTSIDE the head mask (the surrounding scene), never from
    # inside it. `target` still shows the ORIGINAL target face (never overwritten
    # by the swap), so sampling the whole frame would pull the head's colour back
    # toward the target's own skin tone, undoing match_skin_to_source which runs
    # immediately before this step. LAB's L channel is darken-only, mirroring
    # _match_lighting's rule, so this can't wash hair/skin out to grey/silver
    # against a bright background.
    try:
        sel = mask > 0.4
        bg_sel = mask < 0.05
        if int(sel.sum()) > 200 and int(bg_sel.sum()) > 200:
            head_lab = cv2.cvtColor(np.clip(res, 0, 255).astype(np.uint8),
                                    cv2.COLOR_BGR2LAB).astype(np.float32)
            scene_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
            head_mean = head_lab[sel].reshape(-1, 3).mean(0)
            scene_mean = scene_lab[bg_sel].reshape(-1, 3).mean(0)
            cast = np.zeros(3, np.float32)
            cast[0] = min(0.0, scene_mean[0] - head_mean[0]) * 0.35     # L: darken-only
            cast[1:] = (scene_mean[1:] - head_mean[1:]) * 0.12          # a/b: gentle nudge
            out_lab = head_lab + cast[None, None, :] * m3
            res = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8),
                               cv2.COLOR_LAB2BGR).astype(np.float32)
    except Exception as e:
        print(f"[head_swap] harmonize colour-cast skipped: {e}")

    return np.clip(res, 0, 255).astype(np.uint8)


def _match_lighting(src_region, dst, mask):
    """Shift src_region's mean LAB toward dst inside mask (match scene exposure,
    keep most of the source's own colour). Returns a lit-corrected src_region."""
    m = mask > 0.5
    if m.sum() < 50:
        return src_region
    s = cv2.cvtColor(src_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    for c in range(3):
        sm, dm = s[:, :, c][m].mean(), d[:, :, c][m].mean()
        if c == 0:
            # L (brightness): ONLY allow DARKENING toward a dim scene — never
            # brighten dark hair toward the bright wall it now covers, which washes
            # it out to grey/silver (the artefact seen on bright backgrounds).
            out[:, :, c] = s[:, :, c] + min(0.0, dm - sm) * 0.35
        else:
            # a/b (colour): gentle nudge so the hair picks up the scene's cast.
            out[:, :, c] = s[:, :, c] + (dm - sm) * 0.12
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def full_head_swap(source: np.ndarray, target: np.ndarray,
                   feather: float = 0.012):
    """
    Transplant the SOURCE's whole head — face SHAPE + skin + hair (+ glasses) —
    onto the target's body/scene. Unlike InsightFace (which keeps the target's
    face shape), this uses the source's actual head, so a round/slim source stays
    round/slim. A similarity transform (eyes+nose+mouth) preserves the source's
    proportions; only the head region is touched, so the BACKGROUND is untouched.
    Best when source and target face roughly the same way. Returns the composited
    image, or None if it can't (caller falls back to the face swap).

    Sharpness is bounded by the SOURCE photo. The head is warped by whatever
    scale maps the source's eyes/nose/mouth onto the target's, so a source whose
    head is smaller than the target's gets upscaled and comes out visibly softer
    than the surrounding photograph — measured at roughly 1.43x upscale for a
    66x77 source face against a 102x116 target one. A close, sharp source photo
    is what makes this look right; no amount of blending recovers detail that
    was never captured.
    """
    app = _get_insightface()
    if app is None:
        return None
    try:
        src_faces = app.get(source)
        tgt_faces = app.get(target)
        if not src_faces or not tgt_faces:
            return None
        area = lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        sf = max(src_faces, key=area)
        tf = max(tgt_faces, key=area)

        # Similarity (no shear) keeps the source face SHAPE; aligns it to the
        # target's eyes/nose/mouth position, scale and rotation.
        M, _ = cv2.estimateAffinePartial2D(
            np.asarray(sf.kps, np.float32), np.asarray(tf.kps, np.float32),
            method=cv2.LMEDS)
        if M is None:
            return None

        h, w = target.shape[:2]
        head = _parse_region_mask(source, sf.bbox, _HEAD_CLASSES,
                                  up=1.0, down=0.45, side=0.6)
        if head.max() <= 0:
            return None
        # Keep the largest connected region (drop stray parse specks).
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (head > 0.5).astype(np.uint8), connectivity=8)
        if n > 1:
            head = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.float32)

        warped_src = cv2.warpAffine(source, M, (w, h), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(head, M, (w, h), flags=cv2.INTER_LINEAR)

        # Confine to a plausible head box around the target face (no stray bits
        # in the background); generous down/side for long hair.
        tx1, ty1, tx2, ty2 = [int(v) for v in tf.bbox]
        tbw, tbh = tx2 - tx1, ty2 - ty1
        region = np.zeros((h, w), np.float32)
        rx1 = max(0, tx1 - int(tbw * 1.1)); ry1 = max(0, ty1 - int(tbh * 1.6))
        rx2 = min(w, tx2 + int(tbw * 1.1)); ry2 = min(h, ty2 + int(tbh * 1.3))
        region[ry1:ry2, rx1:rx2] = 1.0
        warped_mask *= region

        warped_mask = (warped_mask > 0.5).astype(np.float32)
        er = max(2, int(min(h, w) * 0.012))
        warped_mask = cv2.erode(warped_mask, np.ones((er, er), np.uint8))

        # Match the transplanted head's exposure to the target scene so it doesn't
        # look like a cut-out from a differently-lit photo.
        warped_src = _match_lighting(warped_src, target, warped_mask)

        k = max(3, int(min(h, w) * feather) | 1)
        warped_mask = cv2.GaussianBlur(warped_mask, (k, k), 0)
        alpha = np.stack([np.clip(warped_mask, 0.0, 1.0)] * 3, axis=-1)

        out = (warped_src.astype(np.float32) * alpha +
               target.astype(np.float32) * (1.0 - alpha))
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"[head_swap] full head swap error: {e}")
        return None
