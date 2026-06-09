import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

class LogisticRegressionBaseline:
    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.scaler = None
    
    def train(self, x_train, y_train):
        """Train logistic regression model"""
        # Flatten images for logistic regression
        x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
        
        print("Training Logistic Regression Baseline...")
        self.model.fit(x_train_flat, y_train)
        print("Training complete!")
    
    def evaluate(self, x_test, y_test):
        """Evaluate model on test set"""
        x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0
        
        y_pred = self.model.predict(x_test_flat)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("\nLogistic Regression Results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        return metrics
    
    def save_model(self, path='models/logistic_baseline.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    print("Baseline models module loaded")
