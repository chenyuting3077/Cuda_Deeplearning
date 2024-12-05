# Cuda Deeplearning
## Objective

## Set-Up
```
    conda create --name Cuda_Deeplearning python==3.10
    conda activate Cuda_Deeplearning
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
##  TensorRT

### Installation
```
    pip install torch-tensorrt
```
### Basic TensorRT Workflow:
1. Export the model
2. Select a precision
3. Convert the model
4. Deploy the model
   

## Results
### Inference Result
* ResNet-18
* RTX3080 12GB
* Batch size: 4
* \# of Testing Images: 7500
  
|                         | Precision   | Accuracy  | Per Image Time (ms) | GPU VRAM Peak Usage (MB)|
|  ----                   | ----        | ----      | ----                | ----                     |
|  Pytorch                | fp32        | 92.72     | 10.255              | 44.76                    |
|  TensorRT (Default)     | fp32        | 92.73     | 2.657               | 2.41                     |
|  TensorRT               | fp16        | 92.76     | 2.784               | 1.21                     |
|  TensorRT               | int8        |           |                     |                          |
|  My Cuda model          |             |           |                     |                          |

## To-Do List

- [ ] CNN
  - [ ] classification (Dog & Cat)
    - [x] Using Pytorch to create CNN (ResNet18)
    - [x] Compared accuracy & time
    - [x] Using tensorRT to optimize inference
    - [ ] Triton
    - [ ] Craft the Cuda code to optimize inference
- [ ] Transformer classification
- [ ] Optimize 

### Reference
[cats&dogs dataset Resnet50練習 by pytorch](https://ithelp.ithome.com.tw/articles/10288232?sc=rss.iron)

[Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data)

[Nvidia tensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)

[torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

[Torch-TensorRT](https://github.com/pytorch/TensorRT)