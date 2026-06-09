import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import pickle

DATA_DIR = "mnist_data"

def download_mnist():
    """Download MNIST dataset from Keras"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Save to disk
    with open(os.path.join(DATA_DIR, "train_images.pkl"), "wb") as f:
        pickle.dump(x_train, f)
    with open(os.path.join(DATA_DIR, "train_labels.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    with open(os.path.join(DATA_DIR, "test_images.pkl"), "wb") as f:
        pickle.dump(x_test, f)
    with open(os.path.join(DATA_DIR, "test_labels.pkl"), "wb") as f:
        pickle.dump(y_test, f)
    
    print(f"Dataset downloaded!")
    print(f"Train images: {x_train.shape}, Train labels: {y_train.shape}")
    print(f"Test images: {x_test.shape}, Test labels: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    """Load MNIST dataset from disk"""
    with open(os.path.join(DATA_DIR, "train_images.pkl"), "rb") as f:
        x_train = pickle.load(f)
    with open(os.path.join(DATA_DIR, "train_labels.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(DATA_DIR, "test_images.pkl"), "rb") as f:
        x_test = pickle.load(f)
    with open(os.path.join(DATA_DIR, "test_labels.pkl"), "rb") as f:
        y_test = pickle.load(f)
    
    return (x_train, y_train), (x_test, y_test)

def get_dataset():
    """Get MNIST dataset, download if necessary"""
    if not os.path.exists(DATA_DIR):
        return download_mnist()
    else:
        return load_mnist()

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = download_mnist()
