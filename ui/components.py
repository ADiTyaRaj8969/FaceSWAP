"""Reusable Streamlit UI components."""
import streamlit as st
import cv2
import numpy as np
from PIL import Image


def render_image(image: np.ndarray, caption: str = "", use_container_width: bool = True):
    """Render a BGR numpy image in Streamlit."""
    if image is None:
        st.empty()
        return
    st.image(image, channels="BGR", caption=caption,
             use_container_width=use_container_width)


def render_tone_metrics(tone: dict, title: str = "Skin Tone"):
    """Display skin tone metrics as Streamlit metrics."""
    st.subheader(title)
    c1, c2 = st.columns(2)
    c1.metric("Lightness (L*)", f"{tone.get('L', 0):.1f}")
    c2.metric("Hue (°)", f"{tone.get('hue', 0):.1f}°")
    c1.metric("Saturation", f"{tone.get('saturation', 0):.1f}%")
    c2.metric("Undertone", tone.get("undertone", "-"))
    st.info(f"Category: **{tone.get('category', '-')}**")


def render_quality_metrics(quality: dict):
    """Display quality metrics as a 4-column metric row."""
    st.subheader("Quality Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alignment", f"{quality.get('alignment', 0):.1f}/100")
    m2.metric("Blend Quality", f"{quality.get('blend', 0):.1f}/100")
    m3.metric("ΔE (Colour)", f"{quality.get('delta_e', 0):.2f}")
    m4.metric("Naturalness", f"{quality.get('naturalness', 0):.1f}/100")


def render_delta_e_badge(delta_e: float):
    """Show a coloured badge for colour difference."""
    if delta_e < 10:
        st.success(f"ΔE = {delta_e:.2f}  - Excellent match")
    elif delta_e < 15:
        st.info(f"ΔE = {delta_e:.2f}  - Good match")
    elif delta_e < 20:
        st.warning(f"ΔE = {delta_e:.2f}  - Moderate mismatch (auto-correction applied)")
    else:
        st.error(f"ΔE = {delta_e:.2f}  - High mismatch - manual tone slider recommended")


def render_progress(pct: int, msg: str, bar=None):
    """Update a Streamlit progress bar. Creates one if not passed."""
    if bar is None:
        bar = st.progress(0)
    bar.progress(pct, text=msg)
    return bar


def image_download_button(image: np.ndarray, filename: str = "result.png",
                          label: str = "Download Result (PNG)"):
    """Streamlit download button for a BGR numpy image."""
    from io import BytesIO
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png",
        use_container_width=True,
    )
