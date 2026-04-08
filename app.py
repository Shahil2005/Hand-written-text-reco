import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io

# Set page config
st.set_page_config(page_title="Text Recognition", layout="centered")

# Title
st.title("📝 Handwritten Text Recognition")
st.markdown("Upload an image and get the recognized text")

# Load model (cached to avoid reloading)
@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

@st.cache_data
def predict(img, _processor, _model):
    pixel_values = _processor(images=img, return_tensors="pt").pixel_values
    ids = _model.generate(pixel_values)
    text = _processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text

# Load models
st.info("Loading AI model... This may take a moment on first run.")
processor, model = load_model()
st.success("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Image Details")
        width, height = image.size
        st.write(f"**Dimensions:** {width}x{height} px")
        st.write(f"**Format:** {image.format}")
    
    # Process image
    st.divider()
    
    if st.button("🔍 Recognize Text", use_container_width=True):
        with st.spinner("Processing image..."):
            # Split image into two lines
            line1 = image.crop((0, 0, width, height // 2))
            line2 = image.crop((0, height // 2, width, height))
            
            # Get predictions
            result1 = predict(line1, processor, model)
            result2 = predict(line2, processor, model) 
            
            # Display results
            st.subheader("📋 Recognized Text")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Line 1:**")
                st.info(result1)
            
            with col2:
                st.write("**Line 2:**")
                st.info(result2)
            
            # Combined text
            st.write("**Full Text:**")
            full_text = f"{result1}\n{result2}"
            st.text_area("", value=full_text, height=100, disabled=True)
            
            # Copy button
            st.button("📋 Copy to Clipboard", help="Copy the recognized text")
