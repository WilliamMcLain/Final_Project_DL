"""
Brain Tumor Detection CNN Pipeline
COMP 576 Final Project - Data Loading & Visualization Module

Team: Lin Fang, Nanjia Song, William McLain, Michael Zhang, Kenneth Cadungog
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter
import random

# For deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

#############################################
# 1. DATA LOADING AND ORGANIZATION
#############################################

class BrainTumorDataset(Dataset):
    """Custom Dataset for Brain Tumor MRI Images"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of paths to MRI images
            labels: List of binary labels (0=no tumor, 1=tumor)
            transform: Optional transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path


def load_dataset_from_directory(data_dir, test_size=0.15, val_size=0.15):
    """
    Load images from directory structure and split into train/val/test
    
    Expected structure:
        data_dir/
            tumor/
                image1.jpg
                image2.jpg
                ...
            no_tumor/
                image1.jpg
                image2.jpg
                ...
    
    Args:
        data_dir: Path to dataset directory
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
    
    Returns:
        Dictionary with train/val/test splits
    """
    data_dir = Path(data_dir)
    
    # Find tumor and no_tumor directories
    tumor_dir = data_dir / 'tumor'
    no_tumor_dir = data_dir / 'no_tumor'
    
    if not tumor_dir.exists() or not no_tumor_dir.exists():
        raise ValueError(f"Expected 'tumor' and 'no_tumor' subdirectories in {data_dir}")
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    
    # Load tumor images (label = 1)
    for img_path in tumor_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_paths.append(str(img_path))
            labels.append(1)
    
    # Load no tumor images (label = 0)
    for img_path in no_tumor_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image_paths.append(str(img_path))
            labels.append(0)
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Tumor images: {sum(labels)}")
    print(f"No tumor images: {len(labels) - sum(labels)}")
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Split: first separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    # Split remaining into train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"Train: {len(X_train)} images ({sum(y_train)} tumor, {len(y_train)-sum(y_train)} no tumor)")
    print(f"Val:   {len(X_val)} images ({sum(y_val)} tumor, {len(y_val)-sum(y_val)} no tumor)")
    print(f"Test:  {len(X_test)} images ({sum(y_test)} tumor, {len(y_test)-sum(y_test)} no tumor)")
    
    return {
        'train': {'images': X_train, 'labels': y_train},
        'val': {'images': X_val, 'labels': y_val},
        'test': {'images': X_test, 'labels': y_test}
    }


#############################################
# 2. DATA PREPROCESSING & AUGMENTATION
#############################################

def get_transforms(image_size=224, augment=False):
    """
    Get image transforms for preprocessing and augmentation
    
    Args:
        image_size: Target image size (will resize to image_size x image_size)
        augment: Whether to apply data augmentation
    
    Returns:
        torchvision transforms
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(data_splits, batch_size=32, image_size=224, num_workers=2):
    """
    Create PyTorch DataLoaders for train/val/test sets
    
    Args:
        data_splits: Dictionary from load_dataset_from_directory()
        batch_size: Batch size for training
        image_size: Image size for resizing
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary of DataLoaders
    """
    # Get transforms
    train_transform = get_transforms(image_size=image_size, augment=True)
    eval_transform = get_transforms(image_size=image_size, augment=False)
    
    # Create datasets
    train_dataset = BrainTumorDataset(
        data_splits['train']['images'],
        data_splits['train']['labels'],
        transform=train_transform
    )
    
    val_dataset = BrainTumorDataset(
        data_splits['val']['images'],
        data_splits['val']['labels'],
        transform=eval_transform
    )
    
    test_dataset = BrainTumorDataset(
        data_splits['test']['images'],
        data_splits['test']['labels'],
        transform=eval_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


#############################################
# 3. DATA VISUALIZATION
#############################################

def visualize_sample_images(data_splits, n_samples=8, figsize=(15, 8)):
    """
    Visualize random sample images from both classes
    
    Args:
        data_splits: Dictionary from load_dataset_from_directory()
        n_samples: Number of samples to show per class
        figsize: Figure size
    """
    train_images = data_splits['train']['images']
    train_labels = data_splits['train']['labels']
    
    # Get indices for each class
    tumor_indices = np.where(train_labels == 1)[0]
    no_tumor_indices = np.where(train_labels == 0)[0]
    
    # Sample random images
    tumor_samples = np.random.choice(tumor_indices, min(n_samples, len(tumor_indices)), replace=False)
    no_tumor_samples = np.random.choice(no_tumor_indices, min(n_samples, len(no_tumor_indices)), replace=False)
    
    # Create subplot
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    fig.suptitle('Sample MRI Images from Dataset', fontsize=16, fontweight='bold')
    
    # Plot tumor images
    for i, idx in enumerate(tumor_samples):
        img = Image.open(train_images[idx])
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('TUMOR', fontsize=12, fontweight='bold')
    
    # Plot no tumor images
    for i, idx in enumerate(no_tumor_samples):
        img = Image.open(train_images[idx])
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('NO TUMOR', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(data_splits, figsize=(12, 4)):
    """
    Plot class distribution across train/val/test splits
    
    Args:
        data_splits: Dictionary from load_dataset_from_directory()
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    splits = ['train', 'val', 'test']
    colors = ['#2ecc71', '#e74c3c']
    
    for i, split in enumerate(splits):
        labels = data_splits[split]['labels']
        counts = Counter(labels)
        
        axes[i].bar(['No Tumor', 'Tumor'], 
                   [counts[0], counts[1]], 
                   color=colors,
                   alpha=0.7,
                   edgecolor='black')
        axes[i].set_title(f'{split.upper()} Set', fontweight='bold')
        axes[i].set_ylabel('Count')
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for j, count in enumerate([counts[0], counts[1]]):
            axes[i].text(j, count + 10, str(count), 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Class Distribution Across Splits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def analyze_image_properties(data_splits, n_samples=500):
    """
    Analyze image properties (dimensions, pixel statistics)
    
    Args:
        data_splits: Dictionary from load_dataset_from_directory()
        n_samples: Number of images to sample for analysis
    """
    train_images = data_splits['train']['images']
    
    # Sample images
    sample_indices = np.random.choice(len(train_images), 
                                     min(n_samples, len(train_images)), 
                                     replace=False)
    
    widths, heights, aspects = [], [], []
    mean_intensities, std_intensities = [], []
    
    print("Analyzing image properties...")
    for idx in sample_indices:
        img = Image.open(train_images[idx]).convert('RGB')
        img_array = np.array(img)
        
        widths.append(img.width)
        heights.append(img.height)
        aspects.append(img.width / img.height)
        mean_intensities.append(img_array.mean())
        std_intensities.append(img_array.std())
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Dimensions
    axes[0, 0].hist(widths, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Image Widths')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(heights, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Image Heights')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[0, 2].hist(aspects, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Aspect Ratios')
    axes[0, 2].set_xlabel('Width/Height')
    axes[0, 2].set_ylabel('Frequency')
    
    # Intensity statistics
    axes[1, 0].hist(mean_intensities, bins=30, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Mean Pixel Intensity')
    axes[1, 0].set_xlabel('Mean Intensity')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(std_intensities, bins=30, color='gold', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Std Dev of Pixel Intensity')
    axes[1, 1].set_xlabel('Std Dev')
    axes[1, 1].set_ylabel('Frequency')
    
    # Summary statistics
    stats_text = f"""
    Image Dimensions:
    Width: {np.mean(widths):.1f} ± {np.std(widths):.1f}
    Height: {np.mean(heights):.1f} ± {np.std(heights):.1f}
    Aspect: {np.mean(aspects):.2f} ± {np.std(aspects):.2f}
    
    Pixel Intensity:
    Mean: {np.mean(mean_intensities):.1f} ± {np.std(mean_intensities):.1f}
    Std: {np.mean(std_intensities):.1f} ± {np.std(std_intensities):.1f}
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    
    plt.suptitle('Image Property Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_augmentations(data_splits, n_examples=4, figsize=(15, 10)):
    """
    Visualize effect of data augmentations
    
    Args:
        data_splits: Dictionary from load_dataset_from_directory()
        n_examples: Number of images to show
        figsize: Figure size
    """
    train_images = data_splits['train']['images']
    
    # Sample random images
    sample_indices = np.random.choice(len(train_images), n_examples, replace=False)
    
    # Get transforms
    original_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor()
    ])
    
    fig, axes = plt.subplots(n_examples, 3, figsize=figsize)
    fig.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(sample_indices):
        img = Image.open(train_images[idx]).convert('RGB')
        
        # Original
        original = original_transform(img)
        axes[i, 0].imshow(original.permute(1, 2, 0))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontweight='bold')
        
        # Augmentation 1
        aug1 = aug_transform(img)
        axes[i, 1].imshow(aug1.permute(1, 2, 0))
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Augmented 1', fontweight='bold')
        
        # Augmentation 2
        aug2 = aug_transform(img)
        axes[i, 2].imshow(aug2.permute(1, 2, 0))
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Augmented 2', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


#############################################
# 4. EXAMPLE USAGE
#############################################

if __name__ == "__main__":
    DATA_DIR = "path/to/your/brain_tumor_dataset"
    IMAGE_SIZE = 224
    BATCH_SIZE = 32