import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Accuracy

def build_simple_nn():
    """Build a simple neural network baseline"""
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_simple_nn(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
    """Train simple neural network"""
    print("Training Simple Neural Network...")
    history = model.fit(
        x_train / 255.0, y_train,
        validation_data=(x_val / 255.0, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history

if __name__ == "__main__":
    print("Neural network models module loaded")
