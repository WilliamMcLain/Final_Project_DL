#%%
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

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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
    X_val = all_imgs[n_train:n_train + n_val]
    y_val = all_labels[n_train:n_train + n_val]
    X_test = all_imgs[n_train + n_val:]
    y_test = all_labels[n_train + n_val:]

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
            axes[0, i].imshow(img)
            axes[0, i].set_title('tumor')
        except:
            print(f"couldn't load {imgs[idx]}")
        axes[0, i].axis('off')

    # healthy row
    for i, idx in enumerate(healthy_ones):
        try:
            img = Image.open(imgs[idx])
            axes[1, i].imshow(img)
            axes[1, i].set_title('healthy')
        except:
            print(f"couldn't load {imgs[idx]}")
        axes[1, i].axis('off')

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
basic cnn for mri classification
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleBrainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # ---- Block 1: 3 -> 32 ----
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)

        # ---- Block 2: 32 -> 64 ----
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        # ---- Block 3: 64 -> 128 ----
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global avg pool

        # small classifier head
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)  # 1/2 spatial

        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)  # 1/4 spatial

        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)  # 1/8 spatial

        # Global average pooling
        x = self.gap(x)  # -> [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # -> [B, 128]

        # Classifier
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# training function
def train_model(model, train_loader, val_loader, epochs=11, log_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    if log_dir is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("runs", "brain_cnn", run_id)

    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

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

            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            global_step += 1

            if i % 25 == 0:
                print(f'batch {i}, loss: {loss.item():.3f}')

        model.eval()

        # val set 0
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
        print(
            f'epoch: {epoch + 1}, loss: {running_loss / len(train_loader):.3f}, train acc: {acc:.1f}%, val acc: {val_acc:.1f}%')
        epoch_loss = running_loss / len(train_loader)

        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(f"Params/{name}", param.data.cpu().numpy(), epoch)
            if param.grad is not None:
                writer.add_histogram(f"Grads/{name}", param.grad.data.cpu().numpy(), epoch)

    writer.close()

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


#%% quick model test
if __name__ == "__main__":
    # This is the personal data path
    data_path_personal = r"/Users/lf55/PycharmProjects/576Final/Final_Project_DL-main/data/Brain_Tumor_Detection"
    # load data
    data = load_mri_data(data_path_personal)
    loaders = create_loaders(data, batch_size=10)
    model = SimpleBrainCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params here: {total_params}")

    trained_model = train_model(model, loaders['train'], loaders['val'], epochs=21)
    test_model(trained_model, loaders['test'])

