import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, SparseCategoricalAccuracy

class ValidationLoop:
    def __init__(self, model):
        self.model = model
        self.best_accuracy = 0
        self.best_loss = float('inf')
    
    def validate(self, x_val, y_val, batch_size=32):
        """Run validation on validation set"""
        results = self.model.evaluate(x_val / 255.0, y_val, batch_size=batch_size, verbose=0)
        loss = results[0]
        accuracy = results[1]
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        print(f"Validation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def get_best_metrics(self):
        """Get best validation metrics"""
        return {
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss
        }

if __name__ == "__main__":
    print("Validation loop module loaded")
