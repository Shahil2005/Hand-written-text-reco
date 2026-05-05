import importlib

import streamlit as st
from PIL import Image
import torch


st.set_page_config(
    page_title="Text Recognition Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 30%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 26%),
            linear-gradient(180deg, #f8fafc 0%, #ffffff 50%, #f8fafc 100%);
        color: #0f172a;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: #e5e7eb;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span {
        color: #e5e7eb;
    }

    .hero {
        padding: 2rem 2rem 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(14px);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.25rem;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.10);
        color: #1d4ed8;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0.85rem;
    }

    .hero h1 {
        font-size: 2.35rem;
        line-height: 1.05;
        margin: 0 0 0.6rem 0;
        color: #0f172a;
    }

    .hero p {
        font-size: 1.02rem;
        color: #334155;
        max-width: 62rem;
        margin: 0;
    }

    .panel {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.9);
        padding: 1.1rem 1.1rem 0.9rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }

    .section-title {
        margin: 0 0 0.8rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
    }

    .muted {
        color: #64748b;
        font-size: 0.95rem;
    }

    .result-box {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 18px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        padding: 1rem 1rem 0.85rem;
        height: 100%;
    }

    .result-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    .result-text {
        color: #0f172a;
        font-size: 1rem;
        line-height: 1.6;
        min-height: 3rem;
        white-space: pre-wrap;
    }

    .footer-note {
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 0.75rem;
    }

    /* ── Button text always visible ── */
    div.stButton > button,
    div.stDownloadButton > button {
        color: #0f172a !important;
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid*="primary"] {
        color: #ffffff !important;
        background-color: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        color: #0f172a !important;
        background-color: #e2e8f0 !important;
        border-color: #94a3b8 !important;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid*="primary"]:hover {
        color: #ffffff !important;
        background-color: #1d4ed8 !important;
        border-color: #1e40af !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        st.error(
            "The `transformers` package is not installed in this environment. Install the project dependencies and restart Streamlit."
        )
        st.stop()

    TrOCRProcessor = transformers.TrOCRProcessor
    VisionEncoderDecoderModel = transformers.VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Set device and move model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return processor, model, device


@st.cache_data
def predict(img, _processor, _model, _device):
    pixel_values = _processor(images=img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(_device)
    ids = _model.generate(pixel_values)
    return _processor.batch_decode(ids, skip_special_tokens=True)[0]


with st.sidebar:
    st.markdown("## Text Recognition Studio")
    st.caption("Upload a handwritten image and extract text in a polished workflow.")
    st.markdown(
        """
        <div style="padding:0.9rem 1rem;border:1px solid rgba(255,255,255,0.10);border-radius:16px;background:rgba(255,255,255,0.04);margin:0.75rem 0;">
            <div style="font-size:0.82rem;opacity:0.85;text-transform:uppercase;letter-spacing:0.08em;">Model</div>
            <div style="font-size:1rem;font-weight:700;margin-top:0.25rem;">microsoft/trocr-base-handwritten</div>
            <div style="font-size:0.88rem;opacity:0.75;margin-top:0.35rem;">Cached loading for faster repeated use.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Tips")
    st.markdown(
        "- Use a clear, well-lit image.\n- Crop tightly around the text.\n- Best results come from handwritten notes with good contrast."
    )
    st.markdown("### Output")
    st.markdown("- Line 1 and line 2 are extracted separately.\n- Full text can be copied or downloaded.")


st.markdown(
    """
<div class="hero">
  <div class="eyebrow">AI OCR Studio</div>
  <h1>Turn handwritten images into clean, readable text.</h1>
  <p>Upload a no, devicete, receipt, or handwritten snippet and the app will extract the content in a streamlined interface designed for quick review.</p>
</div>
""",
    unsafe_allow_html=True,
)

processor, model, device = load_model()

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="muted">Supported formats: JPG, JPEG, PNG, BMP. The app will preview the image before processing.</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    image_format = (uploaded_file.type or "image").split("/")[-1].upper()

    top_left, top_right = st.columns([1.35, 1], gap="large")

    with top_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">File Details</div>', unsafe_allow_html=True)
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Width", f"{width}px")
        metric_col2.metric("Height", f"{height}px")
        st.metric("Format", image_format)
        st.caption("The image is split horizontally into two regions for recognition.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    action_col1, action_col2, action_col3 = st.columns([1.2, 1, 1])
    recognize = action_col1.button("Recognize Text", use_container_width=True, type="primary")
    action_col2.download_button(
        "Download OCR Placeholder",
        data="Upload an image and recognize text first.",
        file_name="ocr-instructions.txt",
        mime="text/plain",
        use_container_width=True,
    )
    action_col3.caption("Processing happens locally in the Streamlit session.")

    if recognize:
        with st.spinner("Analyzing image and extracting text..."):
            line1 = image.crop((0, 0, width, max(1, height // 2)))
            line2 = image.crop((0, max(1, height // 2), width, height))

            result1 = predict(line1, processor, model, device)
            result2 = predict(line2, processor, model, device)
            full_text = f"{result1}\n{result2}".strip()

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recognized Text</div>', unsafe_allow_html=True)

        result_col1, result_col2 = st.columns(2, gap="large")

        with result_col1:
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Line 1</div>
                    <div class="result-text">{result1 or 'No text detected.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_col2:
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">Line 2</div>
                    <div class="result-text">{result2 or 'No text detected.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
        st.text_area("Full Text", value=full_text, height=140)

        st.download_button(
            "Download Extracted Text",
            data=full_text or "",
            file_name="recognized-text.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown(
            '<div class="footer-note">Tip: If the output looks fragmented, try a tighter crop or a higher-contrast image.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
