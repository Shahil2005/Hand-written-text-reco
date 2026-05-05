from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model ONCE (important)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.to(device)

def predict(img):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    ids = model.generate(pixel_values)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text

@app.get("/")
async def root():
    return {"message": "Text Recognition API is running"}

@app.post("/predict")
async def predict_text(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # same cropping logic
        width, height = image.size
        line1 = image.crop((0, 0, width, height // 2))
        line2 = image.crop((0, height // 2, width, height))

        result1 = predict(line1)
        result2 = predict(line2)

        return {
            "line1": result1,
            "line2": result2
        }
    except Exception as e:
        return {"error": str(e)}