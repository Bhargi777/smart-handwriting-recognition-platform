import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_augmentation_generator():
    """Create ImageDataGenerator for augmentation"""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )

def apply_augmentation_to_batch(images, labels, augmentation_gen, batch_size=32):
    """Apply augmentation to a batch of images"""
    augmented_images = []
    augmented_labels = []
    
    aug_iter = augmentation_gen.flow(images, labels, batch_size=batch_size)
    for _ in range(len(images) // batch_size):
        aug_batch_images, aug_batch_labels = next(aug_iter)
        augmented_images.append(aug_batch_images)
        augmented_labels.append(aug_batch_labels)
    
    return np.vstack(augmented_images), np.hstack(augmented_labels)

if __name__ == "__main__":
    print("Data augmentation utilities loaded")
