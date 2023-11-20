from torchvision import models
import os 
import torch.nn as nn
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def build_vgg16():
    model = models.vgg16(pretrained=True)
    # VGG16 has 4096 out_features in its last Linear layer
    model.classifier[6] = nn.Linear(4096,1024)
    nn.init.trunc_normal_(model.classifier[6].weight)
    return model
def build_resnet18():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1024)  # ResNet18 has 512 out_features in its last layer
    nn.init.trunc_normal_(model.fc.weight)

    return model

def build_resnet50():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 1024)  # ResNet50 has 2048 out_features in its last layer
    nn.init.trunc_normal_(model.fc.weight)

    return model

def build_mobelnetv3_large():
    model=models.mobilenet_v3_large(pretrained=True)
    model.classifier[3]=nn.Linear(in_features=1280, out_features=1024, bias=True)
    nn.init.trunc_normal_(model.classifier[3].weight)

    return model
def build_mobelnetv3_small():
    model=models.mobilenet_v3_small(pretrained=True)
    model.classifier[3]=nn.Linear(in_features=1024, out_features=1024, bias=True)
    nn.init.trunc_normal_(model.classifier[3].weight)

    return model 

def build_model(model_name):
    if model_name.lower() == 'vgg16':
        return build_vgg16()
    elif model_name.lower() == 'resnet18':
        return build_resnet18()
    elif model_name.lower() == 'resnet50':
        return build_resnet50()
    elif model_name.lower() == 'mobilenetv3_large':
        return build_mobelnetv3_large()
    elif model_name.lower() == 'mobilenetv3_small':
        return build_mobelnetv3_small()
    else:
        raise ValueError(f"Model name {model_name} is not supported")