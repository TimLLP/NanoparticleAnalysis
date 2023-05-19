import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

import math

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone
# from adet.modeling.context.reason import CRABLayer
# import adet.modeling.context.common as common
class BasicConv(nn.Module):
    """docstring for conv"""
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride,
                 padding):
        super(BasicConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=kernel_size, stride=stride, padding=padding,bias= True)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return self.relu(x)

class DownSample(nn.Module):
    """docstring for conv"""
    def __init__(self,
                 in_planes,
                 out_planes,
                 stride
                 ):
        super(DownSample, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=stride)
    def forward(self, x):
        x = self.max_pool(x)
        return x
