import numpy as np
from PIL import Image

def normalize_image(image):
    """Normalize image to [0, 1] range"""
    return image.astype('float32') / 255.0

def normalize_batch(images):
    """Normalize batch of images"""
    return images.astype('float32') / 255.0

def denormalize_image(image):
    """Convert normalized image back to [0, 255]"""
    return (image * 255).astype('uint8')

def resize_image(image, target_size=(28, 28)):
    """Resize image to target size"""
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    
    return np.array(img.resize(target_size))

def pad_image(image, target_size=(28, 28)):
    """Pad image to target size"""
    current_h, current_w = image.shape[:2]
    target_h, target_w = target_size
    
    pad_h = max(0, target_h - current_h)
    pad_w = max(0, target_w - current_w)
    
    if len(image.shape) == 2:  # Grayscale
        return np.pad(image, ((pad_h//2, pad_h - pad_h//2), 
                              (pad_w//2, pad_w - pad_w//2)), 
                      mode='constant', constant_values=0)
    else:  # Color
        return np.pad(image, ((pad_h//2, pad_h - pad_h//2), 
                              (pad_w//2, pad_w - pad_w//2), (0, 0)), 
                      mode='constant', constant_values=0)

if __name__ == "__main__":
    print("Preprocessing utilities loaded")
