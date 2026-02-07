import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------
# 1. SETUP PATHS
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "brain_tumor_seg4", "weights", "best.pt")

# -------------------------------------------------
# 2. LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_yolo():
    if os.path.exists(MODEL_PATH):
        try:
            return YOLO(MODEL_PATH), True
        except:
            return None, False
    return None, False

model, is_active = load_yolo()

# -------------------------------------------------
# 3. PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="NeuroScan AI", layout="wide", page_icon="🧠")

# -------------------------------------------------
# 4. SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    conf_threshold = st.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)
    st.divider()
    
    if is_active:
        st.success("🧬 Segmentation Engine: ✅ ACTIVE")
        st.write("**Classes:**")
        for i, name in model.names.items():
            st.info(f"{i}: {name.upper()}")
    else:
        st.error("🧬 Segmentation Engine: ❌ INACTIVE")

# -------------------------------------------------
# 5. MAIN UI
# -------------------------------------------------
st.title("🧠 NeuroScan Vision: Analysis")
st.markdown("---")

uploaded_files = st.file_uploader("Upload MRI Scans", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and model:
    tabs = st.tabs([f"📄 {f.name}" for f in uploaded_files])
    
    for i, uploaded_file in enumerate(uploaded_files):
        with tabs[i]:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)
            
            # Predict
            results = model.predict(img_np, conf=conf_threshold)
            res = results[0]
            
            col_viz, col_data = st.columns([2, 1])
            
            with col_viz:
                st.subheader(" Analysis Visualizations ")
                viz_tabs = st.tabs([" Detection ", " GRAD-CAM ", " Segmentation "])
                
                with viz_tabs[0]:
                    st.image(res.plot(masks=False), use_container_width=True, caption="Basic YOLO Detection")
                
                with viz_tabs[1]:
                    # --- ROBUST GRAD-CAM LOGIC ---
                    if res.masks is not None:
                        try:
                            # 1. Get raw mask and ensure it is float32
                            mask = res.masks.data[0].cpu().numpy().astype(np.float32)
                            
                            # 2. Resize mask to match image size (fixing the func!=0 error)
                            mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
                            
                            # 3. Create heatmap
                            heatmap = np.uint8(255 * mask_resized)
                            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                            
                            # 4. Overlay
                            grad_cam = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
                            st.image(grad_cam, use_container_width=True, caption="AI Attention Heatmap")
                        except Exception as e:
                            st.error(f"Heatmap Error: {e}")
                    else:
                        st.info("No tumor detected for heatmap generation.")
                
                with viz_tabs[2]:
                    if res.masks:
                        st.image(res.plot(boxes=False, masks=True), use_container_width=True, caption="Precise AI Segmentation")

            with col_data:
                st.subheader("🧪 Staging & Reports")
                if res.masks:
                    for j, (box, mask) in enumerate(zip(res.boxes, res.masks.xy)):
                        label = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        
                        # Area calculation
                        poly = np.array(mask, dtype=np.int32)
                        area = int(cv2.contourArea(poly))
                        
                        # Medical Staging
                        if area < 5000: stage, color = "Stage I (Low)", "#28a745"
                        elif area < 15000: stage, color = "Stage II (Moderate)", "#ffc107"
                        else: stage, color = "Stage III (High)", "#dc3545"

                        with st.container(border=True):
                            st.markdown(f"### 🧬 Tumor {j+1}: {label.upper()}")
                            st.markdown(f"**Stage:** <span style='color:{color}; font-weight:bold'>{stage}</span>", unsafe_allow_html=True)
                            st.write(f"**Area:** {area:,} pixels²")
                            st.write(f"**Confidence:** {conf:.1%}")
                            st.progress(conf)
                else:
                    st.success("Clean Scan: No abnormalities detected.")