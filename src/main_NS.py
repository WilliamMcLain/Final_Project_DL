"""
brain mri stuff for comp576 with PCA dimensionality reduction
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random

# torch stuff
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

# set some seeds i guess
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class BrainMRIData(Dataset):
    """dataset for mri images"""
    
    def __init__(self, image_list, labels, transform=None):
        self.image_list = image_list
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        # sometimes images fail to load?
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # dummy image if broken
            image = Image.new('RGB', (224, 224), color='gray')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label  # removed img_path for now


def load_mri_data(data_folder, test_size=0.15, val_size=0.15):
    # load mri data from folders
    data_path = Path(data_folder)
    
    yes_dir = data_path / 'yes'
    no_dir = data_path / 'no'
    
    # check dirs
    if not yes_dir.exists():
        print("yes folder missing??")
        return None
    if not no_dir.exists():
        print("no folder missing??")
        return None
    
    all_imgs = []
    all_labels = []
    
    # yeah load the tumor ones
    cnt = 0
    for f in yes_dir.iterdir():
        if f.suffix.lower() in ['.jpg', '.png']:
            all_imgs.append(str(f))
            all_labels.append(1)
            cnt += 1
    print(f"found {cnt} tumor images")
    
    # and the healthy ones
    cnt2 = 0
    for f in no_dir.iterdir():
        if f.suffix.lower() in ['.jpg', '.png']:
            all_imgs.append(str(f))
            all_labels.append(0)
            cnt2 += 1
    print(f"found {cnt2} healthy images")
    
    # random shuffle because why not
    temp = list(zip(all_imgs, all_labels))
    random.shuffle(temp)
    all_imgs, all_labels = zip(*temp)
    
    # split into train val test - this is probably wrong but whatever
    n_total = len(all_imgs)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val
    
    # just slice the lists
    X_train = all_imgs[:n_train]
    y_train = all_labels[:n_train]
    X_val = all_imgs[n_train:n_train+n_val]
    y_val = all_labels[n_train:n_train+n_val]
    X_test = all_imgs[n_train+n_val:]
    y_test = all_labels[n_train+n_val:]
    
    print(f"split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return {
        'train': {'images': np.array(X_train), 'labels': np.array(y_train)},
        'val': {'images': np.array(X_val), 'labels': np.array(y_val)},
        'test': {'images': np.array(X_test), 'labels': np.array(y_test)}
    }


def make_transforms(img_size=224, augment=False):
    # make image transforms
    if augment:
        # training with aug
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        # basic for eval
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def create_loaders(data_dict, batch_size=32, img_size=224):
    # make data loaders
    if data_dict is None:
        print("no data")
        return {}
    
    train_tf = make_transforms(img_size, augment=True)
    eval_tf = make_transforms(img_size, augment=False)
    
    train_dataset = BrainMRIData(
        data_dict['train']['images'],
        data_dict['train']['labels'], 
        train_tf
    )
    
    val_dataset = BrainMRIData(
        data_dict['val']['images'],
        data_dict['val']['labels'],
        eval_tf
    )
    
    test_dataset = BrainMRIData(
        data_dict['test']['images'], 
        data_dict['test']['labels'],
        eval_tf
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train': train_loader,
        'val': val_loader, 
        'test': test_loader
    }


def show_images(data_dict, n=6):
    # show a couple images
    if data_dict is None:
        print("no data")
        return
    
    imgs = data_dict['train']['images']
    lbls = data_dict['train']['labels']
    
    # find some tumor ones
    tumor_ones = []
    healthy_ones = []
    for i in range(len(lbls)):
        if lbls[i] == 1 and len(tumor_ones) < n:
            tumor_ones.append(i)
        elif lbls[i] == 0 and len(healthy_ones) < n:
            healthy_ones.append(i)
        if len(tumor_ones) >= n and len(healthy_ones) >= n:
            break
    
    # make the plot
    fig, axes = plt.subplots(2, n)
    if n == 1:
        axes = axes.reshape(2, 1)  # fix for single column
    
    # tumor row
    for i, idx in enumerate(tumor_ones):
        try:
            img = Image.open(imgs[idx])
            axes[0,i].imshow(img)
            axes[0,i].set_title('tumor')
        except:
            print(f"couldn't load {imgs[idx]}")
        axes[0,i].axis('off')
    
    # healthy row  
    for i, idx in enumerate(healthy_ones):
        try:
            img = Image.open(imgs[idx])
            axes[1,i].imshow(img)
            axes[1,i].set_title('healthy')
        except:
            print(f"couldn't load {imgs[idx]}")
        axes[1,i].axis('off')
    
    plt.show()


def check_stats(data_dict):
    # just look at some images real quick
    if data_dict is None:
        return
    
    imgs = data_dict['train']['images']
    if len(imgs) == 0:
        print("no images lol")
        return
    
    # check like 20 random ones
    widths = []
    heights = []
    
    # pick some random indices
    indices = random.sample(range(len(imgs)), min(20, len(imgs)))
    
    for idx in indices:
        try:
            img = Image.open(imgs[idx])
            w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception as e:
            print(f"oops couldn't open {imgs[idx]}")
            continue
    
    if widths:
        print(f"saw {len(widths)} images")
        print(f"widths: {min(widths)} to {max(widths)}")
        print(f"heights: {min(heights)} to {max(heights)}")
        # just average them manually
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)
        print(f"roughly {int(avg_w)} by {int(avg_h)} on average")
    else:
        print("didn't get any images to check")


"""
PCA Dimensionality Reduction Analysis for Brain MRI Images
"""

def load_images_as_matrix(data_dict, img_size=64, max_images=None, grayscale=True):
    """
    Load images and convert to matrix format for PCA
    Each row is a flattened image
    
    Args:
        data_dict: Data dictionary from load_mri_data
        img_size: Size to resize images to (creates square images)
        max_images: Maximum number of images to load (None = all)
        grayscale: Convert to grayscale (recommended for PCA)
    
    Returns:
        image_matrix: (n_images, n_pixels) matrix
        labels: corresponding labels
    """
    print(f"\nLoading images for PCA analysis (size={img_size}x{img_size})...")
    
    imgs = data_dict['train']['images']
    lbls = data_dict['train']['labels']
    
    if max_images is not None:
        n_imgs = min(max_images, len(imgs))
    else:
        n_imgs = len(imgs)
    
    # Calculate matrix size
    n_pixels = img_size * img_size
    if not grayscale:
        n_pixels *= 3  # RGB channels
    
    image_matrix = np.zeros((n_imgs, n_pixels))
    labels = []
    
    for i in range(n_imgs):
        try:
            img = Image.open(imgs[i])
            
            # Convert to grayscale if requested
            if grayscale:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            
            # Resize
            img = img.resize((img_size, img_size))
            
            # Flatten and store
            img_array = np.array(img).flatten()
            image_matrix[i] = img_array
            labels.append(lbls[i])
            
            if (i + 1) % 50 == 0:
                print(f"Loaded {i + 1}/{n_imgs} images")
        except Exception as e:
            print(f"Error loading image {i}: {e}")
            continue
    
    print(f"Final matrix shape: {image_matrix.shape}")
    return image_matrix, np.array(labels)


def perform_pca_analysis(image_matrix, img_size=64):
    """
    Perform PCA analysis on image matrix
    
    Args:
        image_matrix: (n_images, n_pixels) matrix
        img_size: Original image size for reshaping
    
    Returns:
        Dictionary with PCA results
    """
    print("\n" + "="*70)
    print("PERFORMING PCA ANALYSIS")
    print("="*70)
    
    num_images, num_pixels = image_matrix.shape
    print(f"Analyzing {num_images} images with {num_pixels} pixels each")
    
    # Compute mean face
    print("Computing mean image...")
    mean_face = np.mean(image_matrix, axis=0)
    centered_faces = image_matrix - mean_face
    
    # Compute covariance matrix
    print("Computing covariance matrix...")
    cov_matrix = (centered_faces.T @ centered_faces) / (num_images - 1)
    
    # Compute eigenvalues and eigenvectors
    print("Computing eigenvalues and eigenvectors...")
    eigenval, eigenvects = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    sortord = np.argsort(eigenval)[::-1]
    eigenval = eigenval[sortord]
    eigenvects = eigenvects[:, sortord]
    
    print(f"Total eigenvalues: {len(eigenval)}")
    
    # Compute variance explained
    tot_var = np.sum(eigenval)
    cum_var = np.cumsum(eigenval) / tot_var
    
    # Find components for 95% and 99% variance
    components_95 = np.where(cum_var >= 0.95)[0][0] + 1
    components_99 = np.where(cum_var >= 0.99)[0][0] + 1
    
    print("\n" + "="*70)
    print("Variance Analysis Results:")
    print("="*70)
    print(f"95% variance retained with: {components_95} components")
    print(f"Dimensionality reduction: {100*(1 - components_95/num_pixels):.1f}%")
    print(f"\n99% variance retained with: {components_99} components")
    print(f"Dimensionality reduction: {100*(1 - components_99/num_pixels):.1f}%")
    print(f"\nOriginal dimensions: {num_pixels}")
    print(f"Reduced dimensions (95%): {components_95}")
    print(f"Reduced dimensions (99%): {components_99}")
    print("="*70)
    
    return {
        'eigenvalues': eigenval,
        'eigenvectors': eigenvects,
        'mean_face': mean_face,
        'centered_faces': centered_faces,
        'cum_variance': cum_var,
        'components_95': components_95,
        'components_99': components_99,
        'img_size': img_size
    }


def plot_pca_results(pca_results, save_path='pca_analysis.png'):
    """
    Plot PCA analysis results
    """
    eigenval = pca_results['eigenvalues']
    cum_var = pca_results['cum_variance']
    components_95 = pca_results['components_95']
    components_99 = pca_results['components_99']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot eigenvalue spectrum
    axes[0].plot(eigenval[:100], linewidth=2)
    axes[0].axvline(x=components_95, color='r', linestyle='--', 
                    label=f'95% variance ({components_95} components)')
    axes[0].axvline(x=components_99, color='g', linestyle='--', 
                    label=f'99% variance ({components_99} components)')
    axes[0].set_xlabel('Principal Component Index', fontsize=12)
    axes[0].set_ylabel('Eigenvalue', fontsize=12)
    axes[0].set_title('Eigenvalue Spectrum (First 100 Components)', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # Plot cumulative variance
    axes[1].plot(cum_var[:100] * 100, linewidth=2)
    axes[1].axhline(y=95, color='r', linestyle='--', label='95% threshold')
    axes[1].axhline(y=99, color='g', linestyle='--', label='99% threshold')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    axes[1].set_title('Cumulative Variance Explained', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved eigenvalue plot to {save_path}")
    plt.show()


def plot_eigenfaces(pca_results, n_components=19, save_path='eigenfaces.png'):
    """
    Plot the mean face and top eigenfaces (principal components)
    """
    mean_face = pca_results['mean_face']
    eigenvects = pca_results['eigenvectors']
    img_size = pca_results['img_size']
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot mean face first
    mean_img = mean_face.reshape(img_size, img_size)
    axes[0].imshow(mean_img, cmap='gray')
    axes[0].set_title('Mean Brain MRI', fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # Plot eigenfaces
    for i in range(min(n_components, 19)):
        eigenface = eigenvects[:, i].reshape(img_size, img_size)
        # Normalize for visualization
        eigenface_norm = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
        
        axes[i + 1].imshow(eigenface_norm, cmap='gray')
        axes[i + 1].set_title(f'PC {i+1}', fontsize=10)
        axes[i + 1].axis('off')
    
    plt.suptitle('Principal Components of Brain MRI Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved eigenfaces plot to {save_path}")
    plt.show()


def reconstruct_image_pca(image_vector, pca_results, n_components):
    """
    Reconstruct an image using n principal components
    """
    mean_face = pca_results['mean_face']
    eigenvects = pca_results['eigenvectors']
    
    # Center the image
    centered = image_vector - mean_face
    
    # Project onto principal components
    coefficients = centered @ eigenvects[:, :n_components]
    
    # Reconstruct
    reconstructed = mean_face + coefficients @ eigenvects[:, :n_components].T
    
    return reconstructed


def demonstrate_reconstruction(pca_results, image_matrix, n_samples=3, 
                              component_counts=[10, 50, 100, 200]):
    """
    Demonstrate image reconstruction with different numbers of components
    """
    img_size = pca_results['img_size']
    
    # Select random samples
    indices = random.sample(range(len(image_matrix)), min(n_samples, len(image_matrix)))
    
    for idx in indices:
        original = image_matrix[idx]
        
        fig, axes = plt.subplots(1, len(component_counts) + 1, figsize=(15, 3))
        
        # Original image
        axes[0].imshow(original.reshape(img_size, img_size), cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Reconstructions
        for i, n_comp in enumerate(component_counts):
            reconstructed = reconstruct_image_pca(original, pca_results, n_comp)
            axes[i + 1].imshow(reconstructed.reshape(img_size, img_size), cmap='gray')
            axes[i + 1].set_title(f'{n_comp} components')
            axes[i + 1].axis('off')
        
        plt.suptitle(f'Image Reconstruction with Different PC Counts (Sample {idx})')
        plt.tight_layout()
        plt.show()


"""
basic cnn for mri classification
"""

import torch.nn as nn
import torch.nn.functional as F

class SimpleBrainCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(SimpleBrainCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(64)

        # pooling + classifier
        self.pool   = nn.MaxPool2d(2, 2)
        self.gap    = nn.AdaptiveAvgPool2d(1)  
        self.dropout = nn.Dropout(p=0.3)
        self.fc     = nn.Linear(64, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        # GAP + classifier
        x = self.gap(x) 
        x = torch.flatten(x, 1) 
        x = self.dropout(x)
        x = self.fc(x) 
        return x

# training function
def train_model(model, train_loader, val_loader, epochs=11):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
    
            # zero grad
            optimizer.zero_grad()
            # forward
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            # backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 25 == 0:
                print(f'batch {i}, loss: {loss.item():.3f}')
        
        model.eval()

        #val set 0
        val_corr = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_corr += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        val_acc = 100 * val_corr / val_total
        print(f'epoch: {epoch+1}, loss: {running_loss/len(train_loader):.3f}, train acc: {acc:.1f}%, val acc: {val_acc:.1f}%')
    
    return model

# test function
def test_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'test accuracy: {100 * correct / total:.1f}%')
    return 100 * correct / total

"""
GradCAM Implementation
"""

import cv2
import matplotlib.patches as patches

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into the target layer
        self.hook_layers()
    
    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        device = next(self.model.parameters()).device
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Zero gradients, backward pass for target class
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        
        # Pool gradients and generate CAM
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        return cam, target_class

def plot_gradcam(model, image_tensor, original_image, target_layer, target_class=None):
    """
    Generate and plot GradCAM heatmap
    
    Args:
        model: Trained model
        image_tensor: Input tensor (batch size 1)
        original_image: Original PIL image for display
        target_layer: Which layer to use for GradCAM
        target_class: Optional target class (if None, uses predicted class)
    """
    # Generate CAM
    gradcam = GradCAM(model, target_layer)
    cam, predicted_class = gradcam.generate_cam(image_tensor, target_class)
    
    # Prepare images
    original_img = np.array(original_image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Overlay heatmap on original image
    alpha = 0.5
    overlayed_img = np.uint8(original_img * alpha + heatmap * (1 - alpha))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlayed_img)
    axes[2].set_title(f'Overlay (Predicted: {"Tumor" if predicted_class == 1 else "Healthy"})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return cam, predicted_class

def visualize_gradcam_for_samples(model, data_dict, num_samples=3):
    """
    Visualize GradCAM for random samples from the test set
    
    Args:
        model: Trained model
        data_dict: Data dictionary containing test images
        num_samples: Number of samples to visualize
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Get test images
    test_images = data_dict['test']['images']
    test_labels = data_dict['test']['labels']
    
    # Choose random samples
    indices = random.sample(range(len(test_images)), min(num_samples, len(test_images)))
    
    # Create transform for preprocessing
    transform = make_transforms(img_size=224, augment=False)
    
    # Use the last convolutional layer as target
    target_layer = model.conv6  # Last convolutional layer
    
    for idx in indices:
        print(f"\nSample {idx}:")
        print(f"True label: {'Tumor' if test_labels[idx] == 1 else 'Healthy'}")
        
        try:
            # Load and process image
            img_path = test_images[idx]
            original_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(original_img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                probabilities = torch.softmax(output, dim=1)
            
            print(f"Predicted: {'Tumor' if prediction == 1 else 'Healthy'}")
            print(f"Confidence: Tumor={probabilities[0][1]:.3f}, Healthy={probabilities[0][0]:.3f}")
            
            # Generate and plot GradCAM
            cam, predicted_class = plot_gradcam(
                model, input_tensor, original_img, target_layer
            )
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue


if __name__ == "__main__":
    # This is the personal data path
    data_path_personal = r"C:\Users\mclai\Documents\codeprojects\deeplearning\final_project\Final_Project_DL\data\Brain_Tumor_Detection"
    
    # Load data
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    data = load_mri_data(data_path_personal)
    
    # ===================================================================
    # PCA ANALYSIS SECTION
    # ===================================================================
    print("\n" + "="*70)
    print("STARTING PCA DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*70)
    
    # Load images as matrix (using smaller size for PCA efficiency)
    # You can adjust img_size and max_images as needed
    image_matrix, labels = load_images_as_matrix(
        data, 
        img_size=64,  # Smaller for faster computation
        max_images=200,  # Use subset for faster analysis, or None for all
        grayscale=True
    )
    
    # Perform PCA analysis
    pca_results = perform_pca_analysis(image_matrix, img_size=64)
    
    # Plot eigenvalue spectrum and variance explained
    plot_pca_results(pca_results, save_path='mri_pca_eigenvalues.png')
    
    # Plot eigenfaces (principal components)
    plot_eigenfaces(pca_results, n_components=19, save_path='mri_eigenfaces.png')
    
    # Demonstrate reconstruction with different numbers of components
    print("\n" + "="*70)
    print("DEMONSTRATING IMAGE RECONSTRUCTION")
    print("="*70)
    demonstrate_reconstruction(
        pca_results, 
        image_matrix, 
        n_samples=3,
        component_counts=[10, 30, 50, 100]
    )
    
    # ===================================================================
    # CNN TRAINING SECTION
    # ===================================================================
    print("\n" + "="*70)
    print("STARTING CNN TRAINING")
    print("="*70)
    
    loaders = create_loaders(data, batch_size=10)
    
    # Create and train model
    model = SimpleBrainCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    trained_model = train_model(model, loaders['train'], loaders['val'], epochs=11)
    
    # Test model
    test_model(trained_model, loaders['test'])
    
    # ===================================================================
    # GRADCAM VISUALIZATION SECTION
    # ===================================================================
    print("\n" + "="*70)
    print("Generating GradCAM Visualizations...")
    print("="*70)
    
    visualize_gradcam_for_samples(trained_model, data, num_samples=3)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Generated files:")
    print("  - mri_pca_eigenvalues.png")
    print("  - mri_eigenfaces.png")
    print("\nFor more detailed analysis, you can:")
    print("1. Adjust PCA parameters (img_size, max_images)")
    print("2. Manually select specific images to analyze")
    print("3. Compare GradCAM for correct vs incorrect predictions")
    print("4. Analyze multiple layers to see feature evolution")
    print("="*70)