"""
@author: Pandu Dewanatha
"""

from google.colab import files
import os
import torch
import pandas as pd
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
from pathlib import Path
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx].split('.')[0]


# Cuda setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
    
# File initialization
test_dir = '/content/test_final'
category_path = '/content/category.csv'
model_path = '/content/trained_efficientnet_b0_unlocked.pth'

# Load category
category_df = pd.read_csv(category_path, header=None, names=['', 'Category'])
class_names = {k: v for k, v in category_df.values}

# Load test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = CustomImageDataset(img_dir=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=100)
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

# Start testing
predictions = []
image_names = []
for images, names in test_loader:
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().tolist())
        image_names.extend(names)

# Convert class number to class names
predicted_labels = [class_names[p] for p in predictions]

# Create CSV and sort data
results_df = pd.DataFrame({
    'Id': image_names, 
    'Category': predicted_labels
    }).sort_values(by='Id', key=lambda x: x.astype(int))
results_df_sorted = results_df.sort_values(by='Id', key=lambda x: x.astype(int))

# Save results to CSV file
results_file_name = 'test_results_ordered2epoch6.csv'
results_df_sorted.to_csv(results_file_name, index=False)

# Download the file to your local machine
files.download(results_file_name)
