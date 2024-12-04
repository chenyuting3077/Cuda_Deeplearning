import torch
from PIL import Image
import os
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from CatDogDataset import CatDogDataset
from torch.utils.data import DataLoader



if __name__ == "__main__":
    # Load the pre-trained ResNet model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load('/home/allen/Desktop/Cuda_Deeplearning/weights/model.pth'))
    model.eval()
    
    # transform
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # data loader
    test_dir = '/home/allen/Desktop/Cuda_Deeplearning/data/dogs-vs-cats/test'
    test_dataset = CatDogDataset(test_dir, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_dataloader_size = len(test_dataset)
    
    # inference
    correct = 0
    total = 0
    for inputs, labels in tqdm(test_dataloader):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')
