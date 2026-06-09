import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
from training_logger import TrainingLogger

class CNNTrainer:
    def __init__(self, model, checkpoint_dir='models/checkpoints'):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger = TrainingLogger()
    
    def train(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=32):
        """Train CNN with checkpointing"""
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(self.checkpoint_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            x_train / 255.0, y_train,
            validation_data=(x_val / 255.0, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_callback, early_stop],
            verbose=1
        )
        
        # Log training history
        for epoch in range(len(history.history['loss'])):
            logs = {
                'loss': history.history['loss'][epoch],
                'val_loss': history.history['val_loss'][epoch],
                'accuracy': history.history['accuracy'][epoch],
                'val_accuracy': history.history['val_accuracy'][epoch]
            }
            self.logger.on_epoch_end(epoch, logs)
        
        return history
    
    def save_model(self, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def plot_training_curves(self, save_path='logs/training_curves.png'):
        """Plot and save training curves"""
        fig = self.logger.plot_training_curves()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"Training curves saved to {save_path}")
        return fig

if __name__ == "__main__":
    print("CNN trainer module loaded")
