import torch
from PIL import Image
import os
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from CatDogDataset import CatDogDataset
from torch.utils.data import DataLoader
from torchprofile import profile_macs

def calculate_parameters_and_flops(model):
    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    # turn the number of parameters into millions
    num_params /= 1e6

    # Calculate the number of FLOPs
    dummy_input = torch.randn(1, 3, 224, 224)
    flops = profile_macs(model, dummy_input)
    # turn the number of FLOPs into billions
    flops /= 1e9
    return num_params, flops

def init_model():
    # Load the pre-trained ResNet model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load('/home/allen/Desktop/Cuda_Deeplearning/weights/model.pth', weights_only=True))
    model.eval()
    return model

def init_data_loader():
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
    return test_dataloader, test_dataloader_size

def inference():
    model = init_model()
    test_dataloader, test_dataloader_size = init_data_loader()
    
    # inference
    correct = 0
    total = 0
    for inputs, labels in tqdm(test_dataloader):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total
    
if __name__ == "__main__":
    model = init_model()
     
    # Calculate the number of parameters and FLOPs
    num_params, flops = calculate_parameters_and_flops(model)
    print(f'Number of parameters: {num_params:.2f}M')
    print(f'Number of FLOPs: {flops:.2f}G')
    
    test_dataloader, test_dataloader_size = init_data_loader()
    
    # inference
    accuracy = inference()
    print(f'Accuracy: {accuracy:.2f}%')