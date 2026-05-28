import numpy as np

from core.landmarks import extract_landmarks_468
from core.segmentor import segment_hair_neck_skin
from core.neck_integrator import seamless_hair_to_neck_blend
from core.blender import laplacian_blend, poisson_blend
from core.color_corrector import harmonize_colors
from core.quality_checker import compute_quality_score


def run_hair_neck_pipeline(
    swapped: np.ndarray,
    target: np.ndarray,
    hair_preserve: float = 0.8,
    neck_strength: float = 0.75,
    blend_strength: float = 0.85,
    blend_levels: int = 4,
) -> dict:
    """
    Focused hair-to-neck seamless blending pipeline.
    Expects a pre-swapped image and target; applies seamless hair-to-neck integration.
    Returns dict with 'result' and 'quality'.
    """
    src_lm = extract_landmarks_468(swapped)
    tgt_lm = extract_landmarks_468(target)

    src_masks = segment_hair_neck_skin(swapped)
    tgt_masks = segment_hair_neck_skin(target)

    result = seamless_hair_to_neck_blend(
        source_img=swapped,
        target_img=target,
        src_masks=src_masks,
        tgt_masks=tgt_masks,
        src_landmarks=src_lm,
        tgt_landmarks=tgt_lm,
        hair_preserve=hair_preserve,
        neck_strength=neck_strength,
        blend_strength=blend_strength,
    )

    blend_mask = tgt_masks["face_mask"]
    result = laplacian_blend(target, result, blend_mask, levels=blend_levels)
    result = poisson_blend(result, target, blend_mask)
    result = harmonize_colors(result, target, tgt_masks)

    quality = compute_quality_score(result, target, src_lm, tgt_lm)

    return {"result": result, "quality": quality}
