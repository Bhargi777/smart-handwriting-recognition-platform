import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

def create_optimizer_with_lr(learning_rate=0.001):
    """Create Adam optimizer with custom learning rate"""
    return optimizers.Adam(learning_rate=learning_rate)

def experiment_with_learning_rates(model, x_train, y_train, x_val, y_val, 
                                   learning_rates=[0.0001, 0.001, 0.01], 
                                   epochs=10, batch_size=32):
    """Experiment with different learning rates"""
    results = {}
    
    for lr in learning_rates:
        print(f"\n--- Testing learning rate: {lr} ---")
        
        # Create new model instance
        model_copy = keras.models.clone_model(model)
        model_copy.compile(
            optimizer=create_optimizer_with_lr(lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model_copy.fit(
            x_train / 255.0, y_train,
            validation_data=(x_val / 255.0, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        val_acc = history.history['val_accuracy'][-1]
        results[lr] = {
            'history': history,
            'best_val_accuracy': val_acc
        }
        print(f"Best validation accuracy with lr={lr}: {val_acc:.4f}")
    
    best_lr = max(results.items(), key=lambda x: x[1]['best_val_accuracy'])[0]
    print(f"\nBest learning rate: {best_lr}")
    
    return results, best_lr

if __name__ == "__main__":
    print("Learning rate experiments module loaded")
