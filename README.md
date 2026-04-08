# Handwritten Text Recognition

An AI-powered application that recognizes handwritten text in images using TrOCR (Transformer-based OCR).

## Features

- 📷 Upload images and recognize handwritten text
- 🎯 Splits images into two lines for better accuracy
- 🚀 Fast inference using pre-trained TrOCR model
- 💻 Beautiful Streamlit web UI

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

If you encounter issues with torch, you may need to install it separately based on your system:
```bash
# CPU version (lighter weight)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Option 1: Streamlit UI (Recommended for easy use)
```bash
streamlit run app.py
```

Then open your browser and go to `http://localhost:8501`

### Option 2: FastAPI Server
```bash
uvicorn api.py:app --reload
```

Then access the API at `http://localhost:8000`
- Upload images at: `http://localhost:8000/docs` (use the interactive Swagger UI)
- Or POST to: `http://localhost:8000/predict`

## How It Works

1. Upload an image containing handwritten text
2. The app automatically:
   - Loads the image
   - Splits it into two horizontal sections
   - Uses TrOCR to recognize text in each section
   - Displays the results

## Model Details

- **Model:** microsoft/trocr-base-handwritten
- **Task:** Handwritten Text Recognition (OCR)
- **Framework:** Transformers, PyTorch

## Notes

- Best results with clear, moderate-sized handwriting
- Supports JPG, PNG, BMP formats
- The model is downloaded on first run (may take a moment)
"# Hand-written-text-reco" 
