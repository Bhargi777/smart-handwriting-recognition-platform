from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from model_service import ModelService

app = FastAPI(title="Handwriting Recognition API", version="0.1.0")

# Initialize model service
model_service = ModelService()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Handwriting Recognition API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint with model status"""
    model_status = "loaded" if model_service.model is not None else "not loaded"
    return {
        "status": "ok",
        "model_status": model_status,
        "api_version": "0.1.0"
    }

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """Predict digit from uploaded image with confidence scores"""
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Make prediction
        predicted_class, confidence, confidence_scores = model_service.predict(image_array)
        
        # Return detailed prediction with confidence scores for all digits
        return {
            "predicted_digit": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_scores": {k: round(v, 4) for k, v in confidence_scores.items()},
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/predict/base64")
async def predict_digit_base64(image_data: dict):
    """Predict digit from base64 encoded image"""
    try:
        import base64
        from io import BytesIO
        
        # Decode base64 image
        image_data_str = image_data.get("image", "")
        image_bytes = base64.b64decode(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Make prediction
        predicted_class, confidence, confidence_scores = model_service.predict(image_array)
        
        return {
            "predicted_digit": predicted_class,
            "confidence": confidence,
            "confidence_scores": confidence_scores
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
