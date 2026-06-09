import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(x_data, y_data, test_size=0.2, random_state=42):
    """
    Split dataset into train and validation sets
    """
    return train_test_split(
        x_data, y_data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_data
    )

def create_train_val_split(x_train, y_train, val_size=0.15, random_state=42):
    """
    Create validation split from training data
    """
    x_train_split, x_val, y_train_split, y_val = split_dataset(
        x_train, y_train, 
        test_size=val_size, 
        random_state=random_state
    )
    return x_train_split, x_val, y_train_split, y_val

if __name__ == "__main__":
    print("Train/test split utilities loaded")
