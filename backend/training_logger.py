import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def on_epoch_end(self, epoch, logs):
        """Callback to log metrics"""
        self.train_losses.append(logs.get('loss', 0))
        self.val_losses.append(logs.get('val_loss', 0))
        self.train_accs.append(logs.get('accuracy', 0))
        self.val_accs.append(logs.get('val_accuracy', 0))
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={logs.get('loss', 0):.4f}, Acc={logs.get('accuracy', 0):.4f}")
    
    def plot_training_curves(self, title="Training Progress"):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(len(self.train_losses))
        
        ax1.plot(epochs, self.train_losses, label='Training Loss')
        ax1.plot(epochs, self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.train_accs, label='Training Accuracy')
        ax2.plot(epochs, self.val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def save_logs(self, filepath):
        """Save logs to JSON"""
        logs = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'timestamp': str(datetime.now())
        }
        with open(filepath, 'w') as f:
            json.dump(logs, f)

if __name__ == "__main__":
    print("Training logger module loaded")
