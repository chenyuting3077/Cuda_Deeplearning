import torch
from PIL import Image
import os
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from CatDogDataset import CatDogDataset
from torch.utils.data import DataLoader
from torchprofile import profile_macs
import time
import torch_tensorrt

Batch_size = 4
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
    test_dataloader = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False)
    test_dataloader_size = len(test_dataset)
    return test_dataloader, test_dataloader_size

def inference(model):
    test_dataloader, test_dataloader_size = init_data_loader()
    # inference
    correct = 0
    total = 0
    # cal the time of inference of each image
    str_time = time.time()
    for inputs, labels in test_dataloader:
        
        inputs = inputs.half().cuda()
        labels = labels.half().cuda()
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    end_time = time.time()
    avg_time = (end_time - str_time) / test_dataloader_size
    # convert avg time to ms
    avg_time *= 1000
    
    return 100 * correct / total, avg_time


if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats(device=None)

    model = torch.export.load("/home/allen/Desktop/Cuda_Deeplearning/weights/model_fp16.ep").module()
    # # Calculate the number of parameters and FLOPs
    test_dataloader, test_dataloader_size = init_data_loader()
    # # # inference
    accuracy, avg_time = inference(model)
    
    peak_memory = torch.cuda.max_memory_allocated(device=None)/ 1e6
    
    print(f"Peak memory: {peak_memory:.2f}MB")
    print(f'Accuracy: {accuracy:.3f}%', f'Average time: {avg_time:.3f}ms')

