import matplotlib.pyplot as plt
import numpy as np

def plot_sample_digits(x_train, y_train, n_samples=10):
    """Plot sample handwritten digits"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(x_train[i], cmap='gray')
            ax.set_title(f'Label: {y_train[i]}')
        ax.axis('off')
    plt.tight_layout()
    return fig

def plot_label_distribution(y_data, title="Label Distribution"):
    """Plot distribution of labels"""
    unique, counts = np.unique(y_data, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(unique, counts)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Count')
    ax.set_title(title)
    return fig

def display_digit_grid(x_data, y_data, n_rows=5, n_cols=5):
    """Display a grid of digits with labels"""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(x_data):
            ax.imshow(x_data[i], cmap='gray')
            ax.set_title(f'Label: {y_data[i]}')
        ax.axis('off')
    plt.tight_layout()
    return fig
