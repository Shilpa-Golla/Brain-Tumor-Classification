import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --border:    #1f2d45;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --danger:    #ef4444;
    --warning:   #f59e0b;
    --success:   #10b981;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin: 12px 0;
}

.result-card.glioma    { border-left: 4px solid var(--danger); }
.result-card.meningioma { border-left: 4px solid var(--warning); }
.result-card.notumor   { border-left: 4px solid var(--success); }
.result-card.pituitary { border-left: 4px solid var(--accent); }

.class-label {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.confidence-bar-bg {
    background: var(--border);
    border-radius: 999px;
    height: 10px;
    margin: 8px 0 4px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 10px;
    border-radius: 999px;
    transition: width 0.6s ease;
}

.metric-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
}

.metric-lbl {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.info-pill {
    display: inline-block;
    background: var(--border);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 12px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin: 2px;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 38px;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.hero-sub {
    color: var(--muted);
    font-size: 15px;
    margin-top: 8px;
}

.warning-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #fbbf24;
    margin: 16px 0;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.stSpinner > div { border-top-color: var(--accent) !important; }

div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--border);
}

</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES   = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_COLORS  = {
    'glioma':      '#ef4444',
    'meningioma':  '#f59e0b',
    'notumor':     '#10b981',
    'pituitary':   '#00d4ff',
}
CLASS_DESC = {
    'glioma':     'Tumour arising from glial cells. Can be benign or malignant. Requires immediate specialist review.',
    'meningioma': 'Tumour of the meninges (brain lining). Usually benign and slow-growing.',
    'notumor':    'No tumour detected in this scan. Brain tissue appears normal.',
    'pituitary':  'Tumour in the pituitary gland. Often benign; may affect hormone regulation.',
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224

# ─── Model Definition ─────────────────────────────────────────────────────────
def build_efficientnet_b0_v2(num_classes=4):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    model = build_efficientnet_b0_v2(num_classes=4)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

# ─── Transform ────────────────────────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─── GradCAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()
        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam    = (weights * self.activations).sum(dim=1, keepdim=True)
        cam    = torch.relu(cam)
        cam    = cam.squeeze().numpy()
        cam    = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def overlay_gradcam(original_img, cam, alpha=0.45):
    img_array = np.array(original_img.resize((IMG_SIZE, IMG_SIZE)).convert('RGB'))
    heatmap   = cm.jet(cam)[:, :, :3]
    heatmap   = (heatmap * 255).astype(np.uint8)
    overlay   = (alpha * heatmap + (1 - alpha) * img_array).astype(np.uint8)
    return Image.fromarray(overlay)

# ─── Prediction ───────────────────────────────────────────────────────────────
def predict(model, image: Image.Image):
    tensor    = infer_transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx  = int(np.argmax(probs))
    return pred_idx, probs

def predict_tta(model, image: Image.Image, n_aug=10):
    tta_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    base_tensor = infer_transform(image).unsqueeze(0)
    all_probs   = torch.zeros(1, 4)
    with torch.no_grad():
        all_probs += torch.softmax(model(base_tensor), dim=1)
        for _ in range(n_aug - 1):
            aug = tta_tf(image).unsqueeze(0)
            all_probs += torch.softmax(model(aug), dim=1)
    probs    = (all_probs / n_aug).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="section-label">⚙ Configuration</div>', unsafe_allow_html=True)

    model_path = st.text_input(
        "Model path (.pth)",
        value="Saved Models/EfficientNet_v2_Phase1_best.pth",
        help="Full path to your saved EfficientNet weights"
    )

    use_tta = st.toggle("Enable TTA (Test-Time Augmentation)", value=False,
                        help="Averages 10 augmented predictions — more accurate, slower")

    show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True,
                             help="Highlights the regions the model focused on")

    st.divider()

    st.markdown('<div class="section-label">📋 Class Guide</div>', unsafe_allow_html=True)
    for cls in CLASS_NAMES:
        color = CLASS_COLORS[cls]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0;"></div>'
            f'<span style="font-size:13px;color:#e2e8f0;">{cls.capitalize()}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        '<div style="font-size:11px;color:#475569;line-height:1.6;">'
        'Model: EfficientNet-B0 v2<br>'
        'Dataset: Brain Tumor MRI (Kaggle)<br>'
        'Classes: 4 | Input: 224×224<br>'
        'Framework: PyTorch'
        '</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

# Hero header
st.markdown(
    '<div class="hero-title">🧠 Brain Tumor<br>MRI Classifier</div>'
    '<div class="hero-sub">EfficientNet-B0 · Transfer Learning · Grad-CAM Explainability</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="warning-box">⚠️ <strong>Research use only.</strong> '
    'This tool is built for academic demonstration and must not be used for clinical diagnosis.</div>',
    unsafe_allow_html=True
)

st.divider()

# ─── Load Model ───────────────────────────────────────────────────
model = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        st.success(f"✅ Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.error("❌ Model file not found. Check the path in the sidebar.")

# ─── Upload ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a brain MRI scan here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded and model:
    image = Image.open(uploaded).convert("RGB")

    # ── Layout: image | results
    col_img, col_res = st.columns([1, 1.4], gap="large")

    with col_img:
        st.markdown('<div class="section-label">🖼 Input Scan</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption=uploaded.name)
        st.markdown(
            f'<div style="margin-top:8px;">'
            f'<span class="info-pill">📐 {image.size[0]}×{image.size[1]} px</span>'
            f'<span class="info-pill">🎨 {image.mode}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_res:
        st.markdown('<div class="section-label">🔬 Analysis Result</div>', unsafe_allow_html=True)

        with st.spinner("Running inference..."):
            if use_tta:
                pred_idx, probs = predict_tta(model, image, n_aug=10)
                mode_label = "TTA (10 augmentations)"
            else:
                pred_idx, probs = predict(model, image)
                mode_label = "Standard"

        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        color      = CLASS_COLORS[pred_class]

        # Main result card
        st.markdown(
            f'<div class="result-card {pred_class}">'
            f'<div style="font-size:11px;color:{color};font-family:Space Mono,monospace;'
            f'text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">Prediction</div>'
            f'<div class="class-label" style="color:{color};">{pred_class.upper()}</div>'
            f'<div style="color:#94a3b8;font-size:13px;margin-top:6px;">{CLASS_DESC[pred_class]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val">{confidence:.1f}%</div>'
                f'<div class="metric-lbl">Confidence</div>'
                f'</div>', unsafe_allow_html=True
            )
        with m2:
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val">{entropy:.3f}</div>'
                f'<div class="metric-lbl">Entropy</div>'
                f'</div>', unsafe_allow_html=True
            )
        with m3:
            sorted_p = np.sort(probs)[::-1]
            margin   = (sorted_p[0] - sorted_p[1]) * 100
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val">{margin:.1f}%</div>'
                f'<div class="metric-lbl">Margin</div>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-class probability bars
        st.markdown('<div class="section-label">📊 Class Probabilities</div>', unsafe_allow_html=True)
        for i, cls in enumerate(CLASS_NAMES):
            p     = float(probs[i]) * 100
            c     = CLASS_COLORS[cls]
            bold  = "font-weight:700;" if i == pred_idx else ""
            st.markdown(
                f'<div style="margin:10px 0;">'
                f'<div style="display:flex;justify-content:space-between;{bold}'
                f'font-size:13px;color:#e2e8f0;margin-bottom:4px;">'
                f'<span>{cls.capitalize()}</span><span>{p:.1f}%</span></div>'
                f'<div class="confidence-bar-bg">'
                f'<div class="confidence-bar-fill" style="width:{p}%;background:{c};"></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f'<div style="font-size:11px;color:#475569;margin-top:8px;">'
            f'Mode: {mode_label}</div>',
            unsafe_allow_html=True
        )

    # ── Grad-CAM Section ─────────────────────────────────────────
    if show_gradcam:
        st.divider()
        st.markdown('<div class="section-label">🔥 Grad-CAM Explainability</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#64748b;font-size:13px;margin-bottom:16px;">'
            'Heatmap shows which regions of the MRI the model focused on to make its prediction. '
            'Red/yellow = high attention. Blue = low attention.</div>',
            unsafe_allow_html=True
        )

        try:
            with st.spinner("Generating Grad-CAM..."):
                # Target last conv layer of EfficientNet-B0
                target_layer = model.features[-1][0]
                gradcam      = GradCAM(model, target_layer)
                tensor       = infer_transform(image).unsqueeze(0)
                cam          = gradcam.generate(tensor, class_idx=pred_idx)
                overlay      = overlay_gradcam(image, cam)

            gc1, gc2, gc3 = st.columns(3, gap="medium")
            with gc1:
                st.markdown('<div style="text-align:center;font-size:12px;color:#64748b;margin-bottom:6px;">Original</div>', unsafe_allow_html=True)
                st.image(image.resize((IMG_SIZE, IMG_SIZE)), use_container_width=True)
            with gc2:
                st.markdown('<div style="text-align:center;font-size:12px;color:#64748b;margin-bottom:6px;">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
                heatmap_img = Image.fromarray((cm.jet(cam)[:, :, :3] * 255).astype(np.uint8))
                st.image(heatmap_img, use_container_width=True)
            with gc3:
                st.markdown('<div style="text-align:center;font-size:12px;color:#64748b;margin-bottom:6px;">Overlay</div>', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

        except Exception as e:
            st.warning(f"Grad-CAM could not be generated: {e}")

elif uploaded and not model:
    st.error("Please fix the model path in the sidebar before uploading an image.")

elif not uploaded:
    # Empty state
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#334155;">'
        '<div style="font-size:64px;margin-bottom:16px;">🧠</div>'
        '<div style="font-family:Space Mono,monospace;font-size:16px;color:#475569;">'
        'Upload an MRI scan to begin</div>'
        '<div style="font-size:13px;color:#334155;margin-top:8px;">'
        'Supported formats: JPG, JPEG, PNG</div>'
        '</div>',
        unsafe_allow_html=True
    )
