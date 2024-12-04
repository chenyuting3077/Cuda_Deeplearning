# Cuda Deeplearning


```
    conda create --name Cuda_Deeplearning python==3.10
    conda activate Cuda_Deeplearning
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### To-Do List

- [ ] CNN classification (Minist)
  - [ ] Using Pytorch to create CNN (VGG16)
  - [ ] Compared accuracy & time
  - [ ] Using tensorRT to optimize inference
  - [ ] Craft the Cuda code to optimize inference
- [ ] Transformer classification
- [ ] Optimize 