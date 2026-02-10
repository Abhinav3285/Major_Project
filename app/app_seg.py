import streamlit as st
import os
import numpy as np

from PIL import Image
from ultralytics import YOLO

import pandas as pd
import base64
from io import BytesIO
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# -------------------------------------------------
# 1. SETUP PATHS
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "brain_tumor_seg4",
    "weights",
    "best.pt"
)

# -------------------------------------------------
# 2. LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_yolo():
    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH), True
        except Exception as e:
            st.error(f"Model load error: {e}")
            return None, False
    return None, False

model, is_active = load_yolo()

# -------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------
def lazy_cv2():
    import cv2
    return cv2

def get_image_download_link(img_array, filename="image.png"):
    pil_img = Image.fromarray(img_array)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/png;base64,{img_str}" download="{filename}">💾 Download</a>'

def calculate_tumor_volume(area_px, pixel_spacing=0.5):
    area_mm = area_px * (pixel_spacing ** 2)
    volume_mm3 = area_mm * 5
    return area_mm, volume_mm3

def get_clinical_recommendation(stage, area, tumor_type):
    if area < 5000:
        return ["Regular monitoring", "Follow-up in 3–6 months"]
    elif area < 15000:
        return ["Neurosurgeon consultation", "Advanced imaging advised"]
    else:
        return ["Urgent intervention", "Immediate treatment planning"]

# -------------------------------------------------
# 4. PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI",
    layout="wide",
    page_icon="🧠"
)

# -------------------------------------------------
# 5. SIDEBAR
# -------------------------------------------------
with st.sidebar:
    page = st.radio("Navigation", ["🔬 Analysis", "📊 Metrics"])

# -------------------------------------------------
# 6. ANALYSIS PAGE
# -------------------------------------------------
def render_analysis_page():
    st.title("🧠 NeuroScan AI")

    uploaded_files = st.file_uploader(
        "Upload MRI images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if not uploaded_files or not model:
        st.info("Upload images to start analysis")
        return

    conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.4)
    border_thickness = st.slider("Border thickness", 1, 5, 2)

    cv2 = lazy_cv2()

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        with st.spinner("Analyzing..."):
            res = model.predict(img_np, conf=conf_threshold, verbose=False)[0]

        st.image(img_np, caption="Original")

        if res.masks:
            overlay = img_np.copy()

            for mask in res.masks.xy:
                poly = np.array(mask, dtype=np.int32)
                area = int(cv2.contourArea(poly))

                color = (0, 255, 0) if area < 5000 else (255, 165, 0) if area < 15000 else (255, 0, 0)

                tmp = overlay.copy()
                cv2.fillPoly(tmp, [poly], color)
                overlay = cv2.addWeighted(tmp, 0.2, overlay, 0.8, 0)
                cv2.polylines(overlay, [poly], True, color, border_thickness)

            st.image(overlay, caption="Segmentation")

        else:
            st.success("No tumor detected")

# -------------------------------------------------
# 7. METRICS PAGE
# -------------------------------------------------
def render_metrics_page():
    st.title("📊 Model Metrics")

    results_path = os.path.join(PROJECT_ROOT, "results", "brain_tumor_seg4", "results.csv")
    if not os.path.exists(results_path):
        st.warning("Metrics file not found")
        return

    df = pd.read_csv(results_path)
    st.dataframe(df)

# -------------------------------------------------
# 8. ROUTING
# -------------------------------------------------
if page == "🔬 Analysis":
    render_analysis_page()
else:
    render_metrics_page()
