"""Live camera capture UI logic."""
import streamlit as st
import numpy as np
from utils.image_io import load_image
from core.detector import detect_faces
from utils.draw_utils import draw_bounding_boxes, draw_landmarks
from core.landmarks import extract_landmarks_468


def render_camera_capture(show_landmarks: bool = False) -> np.ndarray | None:
    """
    Display live camera input widget.
    Returns captured image as BGR numpy array, or None if not yet captured.
    """
    st.info(
        "Position your face clearly. Ensure good lighting and face the camera directly."
    )
    cam_img = st.camera_input("Capture Source Face")
    if cam_img is None:
        return None

    image = load_image(cam_img)
    faces = detect_faces(image)

    if len(faces) == 0:
        st.error("No face detected in the capture. Please try again.")
        return None

    # Display preview with overlays
    preview = image.copy()
    if show_landmarks:
        lm = extract_landmarks_468(image)
        if lm is not None:
            preview = draw_landmarks(preview, lm)
    else:
        preview = draw_bounding_boxes(preview, faces)

    st.image(preview, channels="BGR", caption="Captured - face detected",
             use_container_width=True)
    st.success(f"{len(faces)} face(s) detected")
    return image
