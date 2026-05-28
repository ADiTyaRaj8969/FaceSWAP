import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import glob

from core.detector import detect_faces
from core.landmarks import extract_landmarks_468
from core.segmentor import segment_hair_neck_skin
from core.swapper import swap_face_insightface
from core.blender import laplacian_blend, poisson_blend
from core.skin_tone import analyze_skin_tone, match_skin_tone
from core.neck_integrator import seamless_hair_to_neck_blend
from core.color_corrector import harmonize_colors
from core.quality_checker import compute_quality_score
from utils.draw_utils import draw_landmarks, draw_bounding_boxes
from utils.image_io import load_image, save_image, resize_keep_aspect

# -- PAGE CONFIG
st.set_page_config(
    page_title="Face Swap – Hair to Neck",
    page_icon="🎭",
    layout="wide"
)

st.title("Face Swap - Hair to Neck Seamless Deepfake")
st.caption("Swap faces with precise skin tone matching and seamless hair-to-neck blending.")
st.warning(
    "This tool produces deepfake images. Use responsibly and only with consent. "
    "Output images may include a processing signature in metadata."
)

# -- SIDEBAR CONFIG
st.sidebar.header("Swap Configuration")
blend_strength   = st.sidebar.slider("Blend Strength",        0, 100, 85)
tone_match_str   = st.sidebar.slider("Skin Tone Match %",     0, 100, 90)
hair_preserve    = st.sidebar.slider("Hair Preservation %",   0, 100, 80)
neck_blend_str   = st.sidebar.slider("Neck Blend Strength",   0, 100, 75)
super_res        = st.sidebar.checkbox("Enable Super Resolution", value=False)
show_landmarks   = st.sidebar.checkbox("Show Landmarks Overlay",  value=False)
show_masks       = st.sidebar.checkbox("Show Segmentation Masks", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Quality Thresholds")
min_align_score  = st.sidebar.number_input("Min Alignment Score",  value=80)
max_delta_e      = st.sidebar.number_input("Max ΔE (Colour Diff)", value=15)

# -- SOURCE INPUT
st.header("Step 1 - Source Face Input")
input_method = st.radio(
    "Choose Input Method:", ["Live Camera", "Upload Image"], horizontal=True
)

source_image = None

if input_method == "Live Camera":
    st.info("Position your face clearly. Ensure good lighting and face the camera directly.")
    cam_img = st.camera_input("Capture Source Face")
    if cam_img:
        source_image = load_image(cam_img)

elif input_method == "Upload Image":
    uploaded = st.file_uploader(
        "Upload Source Image", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded:
        source_image = load_image(uploaded)

# -- SOURCE ANALYSIS
if source_image is not None:
    source_image = resize_keep_aspect(source_image, 1024)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Preview")
        faces_src = detect_faces(source_image)
        if len(faces_src) == 0:
            st.error("No face detected. Please retake or upload a clearer image.")
            st.stop()

        display_src = source_image.copy()
        if show_landmarks:
            lm468 = extract_landmarks_468(source_image)
            if lm468 is not None:
                display_src = draw_landmarks(display_src, lm468)
        else:
            display_src = draw_bounding_boxes(display_src, faces_src)

        st.image(display_src, channels="BGR", use_container_width=True)
        st.success(f"{len(faces_src)} face(s) detected")

    with col2:
        st.subheader("Source Skin Tone Analysis")
        src_tone = analyze_skin_tone(source_image, faces_src[0])
        st.metric("Lightness (L*)", f"{src_tone['L']:.1f}")
        st.metric("Hue (°)", f"{src_tone['hue']:.1f}°")
        st.metric("Saturation", f"{src_tone['saturation']:.1f}%")
        st.metric("Undertone", src_tone["undertone"])
        st.metric("Category", src_tone["category"])

        if show_masks:
            masks = segment_hair_neck_skin(source_image)
            st.image(masks["hair_mask"], caption="Hair Mask", use_container_width=True)

    st.markdown("---")

    # -- TARGET SELECTION
    st.header("Step 2 - Select Target Face")
    target_files = (
        glob.glob("targets/target_faces/*.jpg") +
        glob.glob("targets/target_faces/*.jpeg") +
        glob.glob("targets/target_faces/*.png") +
        glob.glob("targets/target_faces/*.webp")
    )
    target_names = [os.path.basename(f) for f in target_files]

    if not target_files:
        st.warning(
            "No target faces found in targets/target_faces/. "
            "Please add some images there."
        )
        st.stop()

    selected_target_name = st.selectbox("Choose Target Face:", target_names)
    target_path = f"targets/target_faces/{selected_target_name}"
    target_image = load_image(target_path)
    target_image = resize_keep_aspect(target_image, 1024)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Target Preview")
        faces_tgt = detect_faces(target_image)
        display_tgt = target_image.copy()
        if show_landmarks:
            lm_tgt = extract_landmarks_468(target_image)
            if lm_tgt is not None:
                display_tgt = draw_landmarks(display_tgt, lm_tgt)
        else:
            display_tgt = draw_bounding_boxes(display_tgt, faces_tgt)
        st.image(display_tgt, channels="BGR", use_container_width=True)
        if faces_tgt:
            st.success(f"{len(faces_tgt)} face(s) detected")
        else:
            st.warning("No face detected in target image.")

    with col4:
        st.subheader("Target Skin Tone & Colour Match")
        if faces_tgt:
            tgt_tone = analyze_skin_tone(target_image, faces_tgt[0])
        else:
            tgt_tone = src_tone.copy()

        delta_e = (
            (src_tone["L"] - tgt_tone["L"]) ** 2 +
            (src_tone["a"] - tgt_tone["a"]) ** 2 +
            (src_tone["b"] - tgt_tone["b"]) ** 2
        ) ** 0.5

        st.metric("Target Lightness (L*)", f"{tgt_tone['L']:.1f}")
        st.metric(
            "Colour Difference (ΔE)",
            f"{delta_e:.2f}",
            delta="Good Match" if delta_e < 15 else "Needs Correction",
        )

        if delta_e > 20:
            st.warning(
                f"High colour difference (ΔE={delta_e:.1f}). "
                "Auto-correction will be applied."
            )
        else:
            st.success("Skin tones are compatible.")

    st.markdown("---")

    # -- SWAP EXECUTION
    st.header("Step 3 - Execute Face Swap")

    if st.button("Swap Face (Hair-to-Neck Seamless)", type="primary",
                 use_container_width=True):

        progress = st.progress(0, text="Initialising pipeline...")

        with st.spinner("Processing..."):

            # Stage 1: Landmark & Segmentation
            progress.progress(10, "Extracting 468 landmarks...")
            src_lm = extract_landmarks_468(source_image)
            tgt_lm = extract_landmarks_468(target_image)

            progress.progress(20, "Segmenting hair, skin, neck regions...")
            src_masks = segment_hair_neck_skin(source_image)
            tgt_masks = segment_hair_neck_skin(target_image)

            # Stage 2: Deep Face Swap
            progress.progress(40, "Running InsightFace deep swap model...")
            swapped = swap_face_insightface(source_image, target_image)

            # Stage 3: Skin Tone Matching
            progress.progress(55, "Matching skin tones...")
            swapped = match_skin_tone(
                swapped, target_image, src_tone, tgt_tone,
                strength=tone_match_str / 100.0
            )

            # Stage 4: Hair-to-Neck Seamless Blend
            progress.progress(65, "Blending hair, face, neck seamlessly...")
            swapped = seamless_hair_to_neck_blend(
                source_img=swapped,
                target_img=target_image,
                src_masks=src_masks,
                tgt_masks=tgt_masks,
                src_landmarks=src_lm,
                tgt_landmarks=tgt_lm,
                hair_preserve=hair_preserve / 100.0,
                neck_strength=neck_blend_str / 100.0,
                blend_strength=blend_strength / 100.0,
            )

            # Stage 5: Laplacian + Poisson Blending
            progress.progress(75, "Applying multi-scale Laplacian blending...")
            blend_mask = tgt_masks["face_mask"]
            swapped = laplacian_blend(target_image, swapped, blend_mask, levels=4)
            swapped = poisson_blend(swapped, target_image, blend_mask)

            # Stage 6: Colour Harmonisation
            progress.progress(85, "Harmonising colours and lighting...")
            swapped = harmonize_colors(swapped, target_image, tgt_masks)

            # Stage 7: Super Resolution (optional)
            if super_res:
                progress.progress(92, "Enhancing resolution...")
                try:
                    from core.super_res import enhance_resolution
                    swapped = enhance_resolution(swapped)
                except Exception:
                    st.warning("Super resolution unavailable; skipping.")

            # Stage 8: Quality Check
            progress.progress(97, "Computing quality metrics...")
            quality = compute_quality_score(swapped, target_image, src_lm, tgt_lm)

            progress.progress(100, "Done!")

        # -- OUTPUT
        st.header("Step 4 - Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Source")
            st.image(source_image, channels="BGR", use_container_width=True)
        with c2:
            st.subheader("Target")
            st.image(target_image, channels="BGR", use_container_width=True)
        with c3:
            st.subheader("Swapped Result")
            st.image(swapped, channels="BGR", use_container_width=True)

        # Quality Metrics
        st.subheader("Quality Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Alignment Score",   f"{quality['alignment']:.1f}/100")
        m2.metric("Blend Quality",     f"{quality['blend']:.1f}/100")
        m3.metric("Skin Tone ΔE",      f"{quality['delta_e']:.2f}")
        m4.metric("Naturalness Score", f"{quality['naturalness']:.1f}/100")

        # Export
        st.subheader("Export Result")
        result_pil = Image.fromarray(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="Download Result (PNG)",
            data=buf.getvalue(),
            file_name="face_swap_result.png",
            mime="image/png",
            use_container_width=True,
        )

        # Save to outputs
        os.makedirs("outputs/results", exist_ok=True)
        save_image(swapped, "outputs/results/latest_swap.png")
