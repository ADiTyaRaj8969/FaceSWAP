"""Output comparison and download panel."""
import streamlit as st
import numpy as np

from ui.components import (
    render_image,
    render_quality_metrics,
    render_delta_e_badge,
    image_download_button,
)


def render_results(
    source: np.ndarray,
    target: np.ndarray,
    result: np.ndarray,
    quality: dict,
    delta_e: float,
):
    """
    Display the three-panel comparison and download options.
    """
    st.header("Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Source")
        render_image(source)
    with c2:
        st.subheader("Target")
        render_image(target)
    with c3:
        st.subheader("Swapped Result")
        render_image(result)

    st.markdown("---")
    render_delta_e_badge(delta_e)
    render_quality_metrics(quality)

    st.markdown("---")
    st.subheader("Export Result")
    col_png, col_jpg = st.columns(2)
    with col_png:
        image_download_button(result, "face_swap_result.png", "Download PNG (lossless)")
    with col_jpg:
        import cv2, io
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=92)
        st.download_button(
            label="Download JPEG (compressed)",
            data=buf.getvalue(),
            file_name="face_swap_result.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
