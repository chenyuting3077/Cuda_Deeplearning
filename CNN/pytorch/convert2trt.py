import torch
import torch_tensorrt
import torchvision.models as models

def init_model():
    # Load the pre-trained ResNet model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 2)
    model.load_state_dict(torch.load('/home/allen/Desktop/Cuda_Deeplearning/weights/model.pth', weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    model = init_model()
    model = model.half() # define your model here
    model = model.cuda() # define your model here

    
    batch_size = 4
    inputs = [torch.randn((batch_size, 3, 224, 224)).half().cuda()] # define what the inputs to the model will look like

    # trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
    
    trt_gm = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions={torch.float16})
    torch_tensorrt.save(trt_gm, "/home/allen/Desktop/Cuda_Deeplearning/weights/model_fp16.ep", inputs=inputs)
    