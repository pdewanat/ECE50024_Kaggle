"""
@author: Pandu Dewanatha
"""

import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from google.colab import files


# Create custom and clean dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, category_mapping, transform=None):
        img_labels = pd.read_csv(annotations_file)
        category_mapping = pd.read_csv(category_mapping).set_index('Category Name')['Category Number'].to_dict()

        # Remove non existent files from csv
        exists_mask = img_labels.apply(lambda row: (Path(img_dir) / row[1]).exists(), axis=1)
        self.img_labels = img_labels[exists_mask]

        self.img_dir = img_dir
        self.transform = transform
        self.category_mapping = category_mapping

    def __len__(self):
        return len(self.img_labels)
    
    # Get filename from csv
    def __getitem__(self, idx):
        filename = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")
        # Access the 'Category' column for labels
        label_name = self.img_labels.iloc[idx, 2]
        label = self.category_mapping[label_name]
        if self.transform:
            image = self.transform(image)
        return image, label

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

full_dataset = CustomImageDataset(
    annotations_file='/content/train.csv',
    img_dir='/content/train',
    category_mapping='/content/category.csv',
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)

# Split 80 to 20 training and validation
train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size
train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

# Additional transforms for the training data set
train_dataset.dataset.transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
model = models.efficientnet_b0(pretrained=True).to(device)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 100).to(device)

# Model setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training
num_epochs = 5  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
    for i, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

        # Model Accuracy
        test_accuracy = 100 * correct / total

        # Progress bar
        progress_bar.set_postfix({'loss': f'{running_loss/(i+1):.4f}', 'accuracy': f'{test_accuracy:.2f}%'})

    # Validation
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        val_progress_bar = tqdm(enumerate(validation_loader), total=len(validation_loader), desc='Validation in progress')
        for i, (images, labels) in val_progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # Validation Accuracy
            val_accuracy = 100 * correct / total

            # Update progress bar
            val_progress_bar.set_postfix({'val_loss': f'{running_loss/(i+1):.4f}', 'val_accuracy': f'{val_accuracy:.2f}%'})

# Save model
model_save_path = '/content/trained_efficientnet_b0_unlocked.pth'
torch.save(model.state_dict(), model_save_path)
files.download(model_save_path)