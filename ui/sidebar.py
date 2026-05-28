"""Streamlit sidebar settings panel."""
import streamlit as st


def render_sidebar() -> dict:
    """
    Render the configuration sidebar.
    Returns dict of all slider/checkbox values.
    """
    st.sidebar.header("Swap Configuration")

    blend_strength  = st.sidebar.slider("Blend Strength",        0, 100, 85)
    tone_match_str  = st.sidebar.slider("Skin Tone Match %",     0, 100, 90)
    hair_preserve   = st.sidebar.slider("Hair Preservation %",   0, 100, 80)
    neck_blend_str  = st.sidebar.slider("Neck Blend Strength",   0, 100, 75)
    super_res       = st.sidebar.checkbox("Enable Super Resolution", value=False)
    show_landmarks  = st.sidebar.checkbox("Show Landmarks Overlay",  value=False)
    show_masks      = st.sidebar.checkbox("Show Segmentation Masks", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quality Thresholds")
    min_align_score = st.sidebar.number_input("Min Alignment Score", value=80)
    max_delta_e     = st.sidebar.number_input("Max ΔE (Colour Diff)", value=15)

    st.sidebar.markdown("---")
    st.sidebar.caption("Face Swap - Hair to Neck v2.0")

    return {
        "blend_strength":   blend_strength / 100.0,
        "tone_match_str":   tone_match_str / 100.0,
        "hair_preserve":    hair_preserve / 100.0,
        "neck_blend_str":   neck_blend_str / 100.0,
        "super_res":        super_res,
        "show_landmarks":   show_landmarks,
        "show_masks":       show_masks,
        "min_align_score":  min_align_score,
        "max_delta_e":      max_delta_e,
    }
