import cv2
import numpy as np

from core.detector import detect_faces
from core.landmarks import extract_landmarks_468
from core.segmentor import segment_hair_neck_skin
from core.swapper import swap_face_insightface
from core.blender import laplacian_blend, poisson_blend
from core.skin_tone import analyze_skin_tone, match_skin_tone
from core.neck_integrator import seamless_hair_to_neck_blend
from core.color_corrector import harmonize_colors
from core.quality_checker import compute_quality_score
from utils.image_io import resize_keep_aspect


def run_full_pipeline(
    source: np.ndarray,
    target: np.ndarray,
    blend_strength: float = 0.85,
    tone_match_strength: float = 0.9,
    hair_preserve: float = 0.8,
    neck_blend_strength: float = 0.75,
    enable_super_res: bool = False,
    progress_callback=None,
) -> dict:
    """
    End-to-end orchestration pipeline.
    Returns dict with 'result', 'quality', 'src_tone', 'tgt_tone', 'delta_e'.
    progress_callback(pct: int, msg: str) is called at each stage.
    """

    def _progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _progress(5, "Detecting faces...")
    faces_src = detect_faces(source)
    faces_tgt = detect_faces(target)
    if not faces_src or not faces_tgt:
        return {"error": "No face detected in source or target image."}

    _progress(10, "Extracting 468 landmarks...")
    src_lm = extract_landmarks_468(source)
    tgt_lm = extract_landmarks_468(target)

    _progress(20, "Segmenting hair, skin, neck regions...")
    src_masks = segment_hair_neck_skin(source)
    tgt_masks = segment_hair_neck_skin(target)

    _progress(30, "Analysing skin tones...")
    src_tone = analyze_skin_tone(source, faces_src[0])
    tgt_tone = analyze_skin_tone(target, faces_tgt[0])
    delta_e = ((src_tone["L"] - tgt_tone["L"]) ** 2 +
               (src_tone["a"] - tgt_tone["a"]) ** 2 +
               (src_tone["b"] - tgt_tone["b"]) ** 2) ** 0.5

    _progress(40, "Running deep face swap model...")
    swapped = swap_face_insightface(source, target)

    _progress(55, "Matching skin tones...")
    swapped = match_skin_tone(swapped, target, src_tone, tgt_tone,
                              strength=tone_match_strength)

    _progress(65, "Blending hair, face, neck seamlessly...")
    swapped = seamless_hair_to_neck_blend(
        source_img=swapped,
        target_img=target,
        src_masks=src_masks,
        tgt_masks=tgt_masks,
        src_landmarks=src_lm,
        tgt_landmarks=tgt_lm,
        hair_preserve=hair_preserve,
        neck_strength=neck_blend_strength,
        blend_strength=blend_strength,
    )

    _progress(75, "Applying Laplacian pyramid blending...")
    blend_mask = tgt_masks["face_mask"]
    swapped = laplacian_blend(target, swapped, blend_mask, levels=4)
    swapped = poisson_blend(swapped, target, blend_mask)

    _progress(85, "Harmonizing colors and lighting...")
    swapped = harmonize_colors(swapped, target, tgt_masks)

    if enable_super_res:
        _progress(92, "Enhancing resolution...")
        try:
            from core.super_res import enhance_resolution
            swapped = enhance_resolution(swapped)
        except Exception:
            pass

    _progress(97, "Computing quality metrics...")
    quality = compute_quality_score(swapped, target, src_lm, tgt_lm)

    _progress(100, "Done!")

    return {
        "result": swapped,
        "quality": quality,
        "src_tone": src_tone,
        "tgt_tone": tgt_tone,
        "delta_e": delta_e,
    }
