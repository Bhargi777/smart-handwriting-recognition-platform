import tensorflow as tf
import numpy as np
import os
from typing import Tuple, Dict

class ModelService:
    def __init__(self, model_path: str = 'models/checkpoints/best_model.h5'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Model not found at {self.model_path}, using fallback")
            # Fallback to a simple model if checkpoint doesn't exist
            from cnn_models import build_final_cnn
            self.model = build_final_cnn()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for prediction"""
        # Ensure image is 28x28 and grayscale
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        # Resize to 28x28 if needed
        if image.shape != (28, 28):
            from PIL import Image
            image = Image.fromarray(image.astype('uint8'))
            image = image.resize((28, 28))
            image = np.array(image)
        
        # Normalize and add channel dimension
        image = image / 255.0
        image = image.reshape(1, 28, 28, 1)
        
        return image
    
    def predict(self, image: np.ndarray) -> Tuple[int, float, Dict]:
        """Make prediction on image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image, verbose=0)
        
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Return confidence scores for all classes
        confidence_scores = {str(i): float(predictions[0][i]) for i in range(10)}
        
        return predicted_class, confidence, confidence_scores

if __name__ == "__main__":
    print("Model service module loaded")
