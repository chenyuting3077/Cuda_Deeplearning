# Cuda Deeplearning
## Objective


```
    conda create --name Cuda_Deeplearning python==3.10
    conda activate Cuda_Deeplearning
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
## Result

## To-Do List

- [ ] CNN classification (Dog & Cat)
  - [x] Using Pytorch to create CNN (ResNet18)
  - [ ] Compared accuracy & time
  - [ ] Using tensorRT to optimize inference
  - [ ] Craft the Cuda code to optimize inference
- [ ] Transformer classification
- [ ] Optimize 

### Reference
[cats&dogs dataset Resnet50練習 by pytorch](https://ithelp.ithome.com.tw/articles/10288232?sc=rss.iron)

[Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data)
