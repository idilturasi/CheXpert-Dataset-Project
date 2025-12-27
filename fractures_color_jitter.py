#Fractures trial 3: Color Jitter

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

    positive_size = 6877
    negative_size = 2699
    uncertain_size = 1072
    
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
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        #BottomCenterCrop(height=180, width=224),
        transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(1.5)),  # 1.5 = contrast factor
        #transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=1))),  # Apply Gaussian Blur
        transforms.ToTensor(),
    ])

    # Create dataset
    pathology = 'Fracture' # Replace with pathology
    dataset = ImageDataset(csv_file='/./groups/CS156b/data/student_labels/train2023.csv', pathology=pathology, transform=transform)
    train_loader = DataLoader(dataset, batch_size = 16, shuffle=True)

    model = models.densenet121()
    model.classifier = nn.Linear(1024, 1)

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ## Training
    # Train the model for n epochs
    n_epochs = 6

    # store metrics
    training_accuracy_history = np.zeros([n_epochs, 1])
    training_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}:', end='')
        train_total = 0
        train_correct = 0
        # train
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            labels = labels.float().unsqueeze(1)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            images = images.repeat(1, 3, 1, 1)
            # forward pass
            output = model(images)

            # calculate categorical cross entropy loss
            loss = criterion(output, labels)
            # backward pass
            loss.backward()
            optimizer.step()

            # track training accuracy
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            # track training loss
            training_loss_history[epoch] += loss.item()
            # progress update after 180 batches
            if i % 180 == 0: print('.',end='')
        training_loss_history[epoch] /= len(train_loader)
        print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}',end='')

    print("done")

    for param in model.parameters():
        print(param.data)
    
    # Save weights (update filename)
    torch.save(model.state_dict(), 'model3_fractures3.pth')

    print("saved weights to file: model3_fractures3.pth")

    print("starting predictions")
    test_dataset = ImageTestDataset(csv_file='/./groups/CS156b/data/student_labels/test_ids.csv', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    output_arr = []

    with torch.no_grad():
        model.eval()
        for i, (ids, images) in enumerate(test_loader):
            images = images.to(device)
            images = images.repeat(1, 3, 1, 1)
            output = model(images)
            probs = torch.sigmoid(output).squeeze(1).cpu().numpy()
            output_arr.extend(probs.tolist())

            if i % 1 == 0:
                print(f"Processed batch {i}/{len(test_loader)}")

    # Load the CSV
    submission_df = pd.read_csv("final_submission.csv") 
    submission_df[pathology] = [(value * 2) - 1 for value in output_arr]
    submission_df.to_csv("fractures_color_jitter.csv", index=False)
    print("saved predictions to csv")

if __name__ == "__main__":
        main()