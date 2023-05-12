import os

import torch
from torch import nn

import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights

from dataset_custom import ImageNet_extract_1000



#Environment

os.environ["TORCH_HOME"] = "Model/"

####



def transform():
    
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((224, 224), antialias= True)])

    
def get_model():
    
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights= weights)
    
    model.classifier = nn.Flatten()
    
    return model


def extract_feature(images, model):
    
    if images.dim() < 4:
        images = images.unsqueeze(0)  
    
    return model(images)