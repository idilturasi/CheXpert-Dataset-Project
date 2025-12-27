import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import numpy as np
import math
import torchvision.models as models


class BottomCenterCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        w, h = img.size
        left = (w - self.width) // 2
        top = h - self.height
        return transforms.functional.crop(img, top=top, left=left, height=self.height, width=self.width)

# Create training set
def create_dataset(csv_file, pathology):
    data = pd.read_csv(csv_file)  # Load CSV file
    data = data.iloc[:-1]
    image_paths = data['Path'].values  # Image paths
    image_paths = ['/./groups/CS156b/data/' + path for path in image_paths]
    
    # Convert to 0/1 instead of -1/1
    labels = [(value + 1) / 2 for value in data[pathology].values]  # Corresponding labels
    
    positive = 0
    negative = 0
    uncertain = 0

    positive_size = 50000
    negative_size = 27612
    uncertain_size = 6500
    
    x = []
    y = []
    
    idx = 0
    while positive < positive_size or negative < negative_size or uncertain < uncertain_size:
        if positive < positive_size and labels[idx] == 1:
            x.append(image_paths[idx])
            y.append(labels[idx])
            positive += 1
        elif negative < negative_size and labels[idx] == 0:
            x.append(image_paths[idx])
            y.append(labels[idx])
            negative += 1
        elif uncertain < uncertain_size and labels[idx] == 0.5:
            x.append(image_paths[idx])
            y.append(labels[idx])
            uncertain += 1
        idx += 1
    print(f'datset size: {len(x)}')
    print(y[:20])
    return x, y

class ImageDataset(Dataset):
    def __init__(self, csv_file, pathology, transform=None):
        self.image_paths, self.labels = create_dataset(csv_file, pathology)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Open image using PIL
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImageTestDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_paths = ['/./groups/CS156b/data/' + path for path in self.data['Path'].values]
        self.ids = self.data['Id'].values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        item_id = self.ids[idx]
        # Open image using PIL
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return item_id, image

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
       # BottomCenterCrop(height=180, width=224),
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(1.5)),
      #  transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=1))),
        transforms.ToTensor(),
    ])

    # Initialize and prepare model
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(in_features=1024, out_features=1)

    # Load weights
    densenet_weights = torch.load('model3_lung_opacity1.pth', map_location='cpu')
    model.load_state_dict(densenet_weights)

    # ✅ Move model to GPU
    model.to(device)
    model.eval()

    pathology = 'Lung Opacity'

    # Load test data
    test_dataset = ImageTestDataset(csv_file='/./groups/CS156b/data/student_labels/solution_ids.csv', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    output_arr = []
    id_arr = []

    with torch.no_grad():
        for i, (ids, images) in enumerate(test_loader):
            images = images.to(device)
            images = images.repeat(1, 3, 1, 1)
            
            # ✅ Model is already on GPU
            output = model(images)
            probs = torch.sigmoid(output).squeeze(1).cpu().numpy()

            output_arr.extend(probs.tolist())
            id_arr.extend(ids)

            if i % 1 == 0:
                print(f"Processed batch {i}/{len(test_loader)}")

    scaled_output = [(value * 2) - 1 for value in output_arr]

    df = pd.DataFrame({
        'Id': id_arr,
        f'{pathology}': scaled_output
    })

    df.to_csv(f'0527{pathology}_final.csv', index=False)
    print(f'Saved predictions to 0527{pathology}_final.csv')

if __name__ == "__main__":
    main()