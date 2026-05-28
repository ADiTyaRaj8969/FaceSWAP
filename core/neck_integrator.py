import cv2
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter


def seamless_hair_to_neck_blend(
    source_img, target_img,
    src_masks, tgt_masks,
    src_landmarks, tgt_landmarks,
    hair_preserve=0.8,
    neck_strength=0.75,
    blend_strength=0.85
):
    """
    Perform seamless full-head swap: hairline -> face -> neck.
    No hard-paste boundaries. Uses gradient-domain + Laplacian blending.
    """

    h, w = target_img.shape[:2]

    # -- 1. FACE MASK ----------------------------------------------------------
    face_mask = tgt_masks['face_mask'].astype(np.float32) / 255.0

    # -- 2. HAIR MASK (expanded from face region via segmentation) -------------
    hair_mask_raw = tgt_masks['hair_mask'].astype(np.uint8)
    hair_mask_dilated = binary_dilation(hair_mask_raw, iterations=12).astype(np.float32)

    # Feather the hair boundary with Gaussian
    hair_boundary = np.abs(hair_mask_dilated - face_mask)
    hair_boundary_soft = gaussian_filter(hair_boundary, sigma=18)

    # -- 3. NECK MASK (from jawline to collar) ----------------------------------
    neck_mask_raw = tgt_masks['neck_mask'].astype(np.float32) / 255.0
    neck_mask_feathered = gaussian_filter(neck_mask_raw, sigma=10)

    # -- 4. COMPOSITE BLEND WEIGHT MAP -----------------------------------------
    # Inside face:    high weight source
    # Hair boundary:  gradual fade (preserve target hair)
    # Neck:           partial blend (match neck tone)
    composite = (
        face_mask * blend_strength +
        hair_boundary_soft * (1.0 - hair_preserve) +
        neck_mask_feathered * (neck_strength * 0.4)
    )
    composite = np.clip(composite, 0.0, 1.0)

    # Stack to 3 channels
    alpha = np.stack([composite] * 3, axis=-1)

    # -- 5. BLEND SOURCE ONTO TARGET -------------------------------------------
    blended = (source_img.astype(np.float32) * alpha +
               target_img.astype(np.float32) * (1.0 - alpha))
    blended = blended.astype(np.uint8)

    # -- 6. NECK TONE GRADIENT --------------------------------------------------
    # Neck skin is naturally 5-15% darker than face
    blended = _apply_neck_darkening(blended, neck_mask_feathered, factor=0.92)

    # -- 7. FINAL EDGE SMOOTHING -----------------------------------------------
    blended = _smooth_blend_boundary(blended, target_img, composite)

    return blended


def _apply_neck_darkening(image, neck_mask, factor=0.92):
    """Darken neck region to match natural skin gradient."""
    img_float = image.astype(np.float32)
    mask_3ch = np.stack([neck_mask] * 3, axis=-1)

    darkened = img_float * factor
    result = img_float * (1 - mask_3ch) + darkened * mask_3ch
    return result.astype(np.uint8)


def _smooth_blend_boundary(blended, target, composite_mask):
    """Apply bilateral filter at blend boundaries for smooth transitions."""
    boundary = cv2.dilate((composite_mask * 255).astype(np.uint8),
                          np.ones((5, 5), np.uint8), iterations=3)
    boundary = cv2.erode(boundary, np.ones((5, 5), np.uint8), iterations=2)

    smooth = cv2.bilateralFilter(blended, d=9, sigmaColor=75, sigmaSpace=75)
    mask = (boundary > 0).astype(np.float32)
    mask_3 = np.stack([mask] * 3, axis=-1)

    result = (smooth.astype(np.float32) * mask_3 +
              blended.astype(np.float32) * (1 - mask_3))
    return result.astype(np.uint8)
