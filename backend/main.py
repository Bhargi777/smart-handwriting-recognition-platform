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
    return {"status": "ok"}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """Predict digit from uploaded image"""
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
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
