from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io

app = FastAPI()

# Load model ONCE (important)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def predict(img):
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    ids = model.generate(pixel_values)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text

@app.post("/predict")
async def predict_text(file: UploadFile = File(...)):
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