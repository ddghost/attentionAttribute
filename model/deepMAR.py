import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .resnet import *

__all__ = ['DeepMAR_ResNet50']

class DeepMAR_ResNet50(nn.Module):
    def __init__(self, net='resnet50', num_classes=10, pretrained=True):
        super(DeepMAR_ResNet50, self).__init__()
        self.num_classes = 10


        self.drop_pool5 = True 

        self.drop_pool5_rate = 0.5
        self.pretrained = True

        self.base = getattr(resnet, net)(pretrained=pretrained).get_features()
        
        self.classifier = nn.Linear(2048, self.num_classes)
        init.normal(self.classifier.weight, std=0.001)
        init.constant(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.drop_pool5:
            x = F.dropout(x, p=self.drop_pool5_rate, training=self.training)
        x = self.classifier(x)
        return x

