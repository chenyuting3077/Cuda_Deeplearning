import torch
from torchvision import models
from PIL import Image
import os
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from CatDogDataset import CatDogDataset
import torch.optim as optim
import tqdm

# train the model 
def train_model(model, train_dataloader, dataset_size, criterion, optimizer, num_epochs=25, device=None):
    model.train()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / dataset_size
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return model

if __name__ == "__main__":
    # model initialization
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # transform
    # Define the image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # data loader
    train_dir = '/home/allen/Desktop/Cuda_Deeplearning/data/dogs-vs-cats/train'
    train_dataset = CatDogDataset(train_dir, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_dataloader_size = len(train_dataset)
    
    model = train_model(model, train_dataloader, train_dataloader_size, criterion, optimizer, num_epochs=1, device=device)
    # save the model
    weights_dir = '/home/allen/Desktop/Cuda_Deeplearning/weights'
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        
    weights_path = os.path.join(weights_dir, 'model.pth')
    torch.save(model.state_dict(), weights_path)
    
    print('Training completed successfully')
    