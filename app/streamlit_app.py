"""
streamlit_app.py — Enhanced Pneumonia Detection Web Application.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
import streamlit as st
import pandas as pd
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models import build_model
from src.config import (
    CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, CHECKPOINT_DIR
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoniaScan Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State for Patient Registry
if "patient_registry" not in st.session_state:
    st.session_state["patient_registry"] = []

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 1200px; padding-top: 2rem; }
    
    /* Sleek Dark Mode Overrides */
    .stApp { background-color: #0f172a; color: #f8fafc; }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px; font-size: 16px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: transparent; color: #38bdf8 !important; border-bottom: 3px solid #38bdf8; }
    
    /* Premium Metric Cards with Glassmorphism */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: #38bdf8; }
    .metric-label { font-size: 14px; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;}
    .metric-value { font-size: 36px; font-weight: 800; }
    
    /* Glowing Colors */
    .normal-color   { color: #4ade80; text-shadow: 0 0 15px rgba(74,222,128,0.3); }
    .pneumonia-color { color: #f87171; text-shadow: 0 0 15px rgba(248,113,113,0.3); }
    
    /* Hide some default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Headers */
    h1, h2, h3 { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
    
    /* Fix Streamlit Warning UI overrides */
    .element-container img { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ── Image Enhancement Helper ───────────────────────────────────────────────────
def apply_enhancements(img: Image.Image, brightness: float, contrast: float, apply_clahe: bool) -> Image.Image:
    if apply_clahe:
        # Convert PIL to cv2 grayscale
        img_cv = np.array(img.convert('L'))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_cv = clahe.apply(img_cv)
        img = Image.fromarray(img_cv).convert('RGB')
    else:
        img = img.convert('RGB')
        
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
        
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        
    return img

# ── Inference transform (no augmentation) ─────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_INFO = {
    "🌟 Ensemble Consensus (Recommended)": {
        "key":    "ensemble",
        "params": "117.5M",
        "desc":   "Runs all 3 models simultaneously and averages their probabilities to reach a robust consensus diagnosis.",
    },
    "ResNet50": {
        "key":    "resnet50",
        "params": "25.6M",
        "desc":   "50-layer CNN with residual (skip) connections.",
    },
    "EfficientNet-B0": {
        "key":    "efficientnet_b0",
        "params": "5.3M",
        "desc":   "Compound-scaled MobileNet backbone.",
    },
    "ViT-B/16": {
        "key":    "vit_b16",
        "params": "86.6M",
        "desc":   "Vision Transformer. 196 image patches with MHSA.",
    },
}

@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    model = build_model(model_key, freeze_backbone=False)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_key}_best.pt")
    if not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE).eval()
    return model, ckpt

@torch.no_grad()
def predict(model, image: Image.Image) -> dict:
    tensor = INFER_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    t0     = time.time()
    logits = model(tensor)
    ms     = (time.time() - t0) * 1000
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx   = int(probs.argmax())
    return {
        "label":      "Normal" if pred_idx == 0 else "Pneumonia",
        "confidence": float(probs[pred_idx]),
        "prob_normal":     float(probs[0]),
        "prob_pneumonia":  float(probs[1]),
        "inference_ms":    round(ms, 1),
    }

def get_gradcam_overlay(model, image: Image.Image, model_key: str):
    if model_key == "ensemble" or model_key == "vit_b16":
        return None  # Grad-CAM not easily supported for Ensemble or ViT out-of-the-box
        
    tensor = INFER_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    img_resized = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    
    # Identify target layers
    if model_key == "resnet50":
        # -1 is AdaptiveAvgPool2d (1x1). We must use -2 which is layer4 (7x7) for heatmaps.
        target_layers = [model.backbone[-2]]
    elif model_key == "efficientnet_b0":
        target_layers = [model.features[-1]]
    else:
        return None
        
    try:
        # GradCAM temporarily modifies the model to track gradients
        cam = GradCAM(model=model, target_layers=target_layers)
        # Passing targets=None automatically computes CAM for the predicted class
        grayscale_cam = cam(input_tensor=tensor, targets=None)[0, :]
        visualization = show_cam_on_image(img_resized / 255.0, grayscale_cam, use_rgb=True)
        return visualization
    except Exception as e:
        return None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #38bdf8;'>🩺 PneumoniaScan Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Advanced Chest X-Ray Analysis</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### ⚙️ Select Model")
    model_display = st.selectbox(
        "Model Architecture",
        options=list(MODEL_INFO.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    info = MODEL_INFO[model_display]
    st.info(f"**{model_display}**\n\nParams: {info['params']}\n\n{info['desc']}")
    
    st.divider()
    st.markdown("### 📋 Patient Data")
    patient_id = st.text_input("Patient ID", value="PID-1001")
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=35)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    with st.expander("➕ Add Clinical Symptoms"):
        patient_symptoms = st.text_area("Clinical Details", "e.g., Shortness of breath, fever, persistent cough...")
    
    st.divider()
    st.markdown("### 🖥️ Hardware")
    st.success(f"Running on: **{'GPU 🚀' if DEVICE.type == 'cuda' else 'CPU 🐢'}**")
    if DEVICE.type == "cuda":
        st.caption(f"Device: {torch.cuda.get_device_name(0)}")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0;'>Automated Pneumonia Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #94a3b8; margin-bottom: 2.5rem;'>Instantly analyze chest X-rays with state-of-the-art Deep Learning</p>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_diagnose, tab_registry, tab_eval, tab_arch = st.tabs(["🩺 Diagnostics", "🗂️ Patient Registry", "📊 Evaluation Metrics", "🧠 Architectures"])

with tab_diagnose:
    st.warning("⚠️ **Research Prototype:** This tool does not constitute medical advice or clinical diagnosis.")
    
    col_input, col_results = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.markdown("### 1. Provide X-Ray")
        input_method = st.radio("Input Method", ["Upload Image", "Use Camera"], horizontal=True, label_visibility="collapsed")
        
        raw_image = None
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Drop a chest X-ray image here", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                raw_image = Image.open(uploaded_file)
        else:
            camera_file = st.camera_input("Take a picture of an X-Ray")
            if camera_file:
                raw_image = Image.open(camera_file)
                
        if raw_image:
            with st.expander("🎛️ Radiology Enhancement Toolkit", expanded=True):
                st.markdown("Adjust the X-Ray to improve visibility of lung opacities before analysis.")
                c_clahe, c_reset = st.columns([3, 1])
                apply_clahe = c_clahe.checkbox("Apply CLAHE (Bone/Tissue Filter)")
                
                slider_contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                slider_brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                
            # Apply enhancements
            enhanced_image = apply_enhancements(raw_image, slider_brightness, slider_contrast, apply_clahe)
            st.image(enhanced_image, caption="Enhanced Input Image", use_container_width=True)
            
    with col_results:
        st.markdown("### 2. Analysis Results")
        if not raw_image:
            st.info("👈 Please provide an image using the panel on the left to see the analysis.")
        else:
            # We add a button so the user can control WHEN to run the analysis (prevents auto-running on every slider move)
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                model_key = MODEL_INFO[model_display]["key"]
                
                if model_key == "ensemble":
                    results_list = []
                    missing_models = []
                    with st.spinner("Running all 3 Neural Networks..."):
                        time.sleep(0.5)
                        for mk in ["resnet50", "efficientnet_b0", "vit_b16"]:
                            m, _ = load_model(mk)
                            if m is not None:
                                results_list.append((mk, predict(m, enhanced_image)))
                            else:
                                missing_models.append(mk)
                    
                    if len(missing_models) > 0:
                        st.error(f"Missing checkpoints for {', '.join(missing_models)}. Wait for them to finish training before using Ensemble.")
                    else:
                        avg_normal = sum(r[1]["prob_normal"] for r in results_list) / 3
                        avg_pneum = sum(r[1]["prob_pneumonia"] for r in results_list) / 3
                        total_time = sum(r[1]["inference_ms"] for r in results_list)
                        
                        consensus_label = "Normal" if avg_normal > avg_pneum else "Pneumonia"
                        conf = max(avg_normal, avg_pneum)
                        
                        # Log to registry
                        st.session_state["patient_registry"].append({
                            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Patient ID": patient_id,
                            "Age": patient_age,
                            "Gender": patient_gender,
                            "Model": "Ensemble Consensus",
                            "Diagnosis": consensus_label,
                            "Confidence": f"{conf:.2%}"
                        })
                        
                        color_cls  = "normal-color" if consensus_label == "Normal" else "pneumonia-color"
                        emoji      = "✅" if consensus_label == "Normal" else "🚨"
                        
                        st.markdown(
                            f"<div class='metric-card' style='margin-bottom: 20px;'>"
                            f"<div class='metric-label'>Consensus Diagnosis (3 Models)</div>"
                            f"<div class='metric-value {color_cls}'>{emoji} {consensus_label}</div>"
                            f"<div style='margin-top: 15px; font-size: 14px; color: #94a3b8;'>Average Confidence: <b>{conf:.1%}</b> &nbsp;|&nbsp; Combined Processing Time: <b>{total_time:.1f}ms</b></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        
                        st.markdown("#### Individual Model Votes")
                        breakdown_data = [{"Model": mk.upper(), "Diagnosis": r["label"], "Confidence": f"{r['confidence']:.1%}"} for mk, r in results_list]
                        st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)
                        
                        st.markdown("#### Average Probability Distribution")
                        chart_data = pd.DataFrame({"Probability": [avg_normal, avg_pneum], "Class": ["Normal", "Pneumonia"]})
                        st.bar_chart(chart_data.set_index("Class"), height=250, use_container_width=True)
                        
                        report_text = f"====================================\n" \
                                      f"     PNEUMONIA DIAGNOSTIC REPORT      \n" \
                                      f"====================================\n\n" \
                                      f"PATIENT DATA:\n- ID: {patient_id}\n- Age: {patient_age}\n- Gender: {patient_gender}\n"
                        if patient_symptoms and "e.g." not in patient_symptoms:
                            report_text += f"- Symptoms: {patient_symptoms}\n"
                        report_text += f"\nModel Architecture: ENSEMBLE CONSENSUS\n" \
                                      f"Final Diagnosis: {consensus_label}\n" \
                                      f"Average Confidence: {conf:.2%}\n\n" \
                                      f"Detailed Probabilities:\n" \
                                      f"- Normal:    {avg_normal:.2%}\n" \
                                      f"- Pneumonia: {avg_pneum:.2%}\n\n" \
                                      f"Processing Time: {total_time:.1f}ms\n"
                                      
                        st.download_button("📥 Download Diagnostic Report", data=report_text, file_name=f"report_{patient_id}.txt", mime="text/plain", use_container_width=True, type="secondary")
                        
                else:
                    with st.spinner("Initializing neural network..."):
                        model, ckpt = load_model(model_key)
                        
                    if model is None:
                        st.error(f"Checkpoint missing for **{model_display}**. Please wait for it to finish training.")
                    else:
                        with st.spinner("Analyzing image features..."):
                            time.sleep(0.5)
                            result = predict(model, enhanced_image)
                            
                        label = result["label"]
                        conf = result["confidence"]
                        
                        # Log to registry
                        st.session_state["patient_registry"].append({
                            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Patient ID": patient_id,
                            "Age": patient_age,
                            "Gender": patient_gender,
                            "Model": model_display,
                            "Diagnosis": label,
                            "Confidence": f"{conf:.2%}"
                        })
                        
                        color_cls  = "normal-color" if label == "Normal" else "pneumonia-color"
                        emoji      = "✅" if label == "Normal" else "🚨"
                        
                        st.markdown(
                            f"<div class='metric-card' style='margin-bottom: 20px;'>"
                            f"<div class='metric-label'>Diagnosis</div>"
                            f"<div class='metric-value {color_cls}'>{emoji} {label}</div>"
                            f"<div style='margin-top: 15px; font-size: 14px; color: #94a3b8;'>Confidence: <b>{conf:.1%}</b> &nbsp;|&nbsp; Processing Time: <b>{result['inference_ms']}ms</b></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        
                        st.markdown("#### Probability Distribution")
                        chart_data = pd.DataFrame({"Probability": [result["prob_normal"], result["prob_pneumonia"]], "Class": ["Normal", "Pneumonia"]})
                        st.bar_chart(chart_data.set_index("Class"), height=250, use_container_width=True)
                        
                        # Explainable AI (Grad-CAM)
                        with st.spinner("Generating Grad-CAM heatmap..."):
                            heatmap = get_gradcam_overlay(model, enhanced_image, model_key)
                        
                        if heatmap is not None:
                            st.markdown("#### Explainability (Grad-CAM Heatmap)")
                            st.image(heatmap, caption="Red areas = Regions the model associates with its final prediction", use_container_width=True)
                        
                        report_text = f"====================================\n" \
                                      f"     PNEUMONIA DIAGNOSTIC REPORT      \n" \
                                      f"====================================\n\n" \
                                      f"PATIENT DATA:\n- ID: {patient_id}\n- Age: {patient_age}\n- Gender: {patient_gender}\n"
                        if patient_symptoms and "e.g." not in patient_symptoms:
                            report_text += f"- Symptoms: {patient_symptoms}\n"
                        report_text += f"\nModel Architecture: {model_display}\n" \
                                      f"Final Diagnosis: {label}\n" \
                                      f"Overall Confidence: {conf:.2%}\n\n" \
                                      f"Detailed Probabilities:\n" \
                                      f"- Normal:    {result['prob_normal']:.2%}\n" \
                                      f"- Pneumonia: {result['prob_pneumonia']:.2%}\n\n" \
                                      f"Processing Time: {result['inference_ms']}ms\n"
                                      
                        st.download_button("📥 Download Diagnostic Report", data=report_text, file_name=f"report_{patient_id}.txt", mime="text/plain", use_container_width=True, type="secondary")

with tab_registry:
    st.markdown("### 🗂️ Patient Registry Database")
    st.markdown("This database logs all analyses performed during this active session.")
    
    if len(st.session_state["patient_registry"]) == 0:
        st.info("No records found. Run an analysis in the **Diagnostics** tab to log a patient here.")
    else:
        df_registry = pd.DataFrame(st.session_state["patient_registry"])
        st.dataframe(df_registry, use_container_width=True, hide_index=True)
        
        csv = df_registry.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Registry to CSV",
            data=csv,
            file_name="patient_registry_log.csv",
            mime="text/csv",
        )

with tab_eval:
    st.markdown("### Model Performance Comparison")
    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "comparison_table.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        
        # Style the dataframe (Highlight best scores in green)
        styled_df = df.style.highlight_max(subset=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'], color='#065f46') \
                            .highlight_min(subset=['params_M'], color='#065f46') \
                            .format(precision=4)
                            
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        plots_dir = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
        c1, c2 = st.columns(2)
        comp_plot = os.path.join(plots_dir, "model_comparison_metrics.png")
        roc_plot = os.path.join(plots_dir, "roc_curves_comparison.png")
        
        if os.path.exists(comp_plot):
            c1.image(comp_plot, caption="Metrics Overview", use_container_width=True)
        if os.path.exists(roc_plot):
            c2.image(roc_plot, caption="ROC-AUC Curves", use_container_width=True)
    else:
        st.info("Evaluation results not found yet. They will appear here once the training and evaluation pipeline completes!")

with tab_arch:
    st.markdown("### Deep Learning Architectures Explained")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="text-align: left; height: 100%;">
            <h3 style="color: #38bdf8 !important;">ResNet50</h3>
            <p style="color: #94a3b8; font-size: 14px;">25.6M Parameters</p>
            <hr style="border-color: #334155;">
            <ul>
                <li><b>Core:</b> 50-layer CNN with residual block skip connections.</li>
                <li><b>Pros:</b> Extremely stable, battles vanishing gradients effectively.</li>
                <li><b>Use Case:</b> Industry standard for medical imaging baselines.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: left; height: 100%;">
            <h3 style="color: #38bdf8 !important;">EfficientNet-B0</h3>
            <p style="color: #94a3b8; font-size: 14px;">5.3M Parameters</p>
            <hr style="border-color: #334155;">
            <ul>
                <li><b>Core:</b> Compound-scaled MobileNet backbone with Squeeze-and-Excitation.</li>
                <li><b>Pros:</b> Best accuracy-per-parameter, ultra-fast inference.</li>
                <li><b>Use Case:</b> Resource-constrained environments and real-time inference.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card" style="text-align: left; height: 100%;">
            <h3 style="color: #38bdf8 !important;">ViT-B/16</h3>
            <p style="color: #94a3b8; font-size: 14px;">86.6M Parameters</p>
            <hr style="border-color: #334155;">
            <ul>
                <li><b>Core:</b> Vision Transformers applied to 16x16 image patches.</li>
                <li><b>Pros:</b> Global spatial relationship tracking.</li>
                <li><b>Use Case:</b> State-of-the-art on massive datasets, but struggles slightly on small datasets compared to CNNs.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
