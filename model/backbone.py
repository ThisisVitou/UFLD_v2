import torch, pdb
import torchvision
import torch.nn.modules
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, 
    ResNet101_Weights, ResNet152_Weights,
    ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights,
    Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights
)

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)

class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained=False):
        super(resnet, self).__init__()
        
        # Use new weights parameter instead of pretrained
        if layers == '18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnet18(weights=weights)
        elif layers == '34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnet34(weights=weights)
        elif layers == '50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnet50(weights=weights)
        elif layers == '101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnet101(weights=weights)
        elif layers == '152':
            weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnet152(weights=weights)
        elif layers == '50next':
            weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnext50_32x4d(weights=weights)
        elif layers == '101next':
            weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.resnext101_32x8d(weights=weights)
        elif layers == '50wide':
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.wide_resnet50_2(weights=weights)
        elif layers == '101wide':
            weights = Wide_ResNet101_2_Weights.IMAGENET1K_V1 if pretrained else None
            model = torchvision.models.wide_resnet101_2(weights=weights)
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34', pretrained=True)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4