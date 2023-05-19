from typing import Dict
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from skimage.measure import label,regionprops,regionprops_table
from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform
from adet.modeling.small.smallParameter import*

BASIS_MODULE_REGISTRY = Registry("BASIS_MODULE")
BASIS_MODULE_REGISTRY.__doc__ = """
Registry for basis module, which produces global bases from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


def build_basis_module(cfg, input_shape):
    name = cfg.MODEL.BASIS_MODULE.NAME
    return BASIS_MODULE_REGISTRY.get(name)(cfg, input_shape)

def small_gt(masks):
    tiny_masks = np.zeros(masks.shape)
    normal_masks = np.zeros(masks.shape)
    for i in range(0, len(masks)):
        img = masks[i, :, :]
        # print(img.shape)
        img = img.detach().cpu().numpy()
        binary = img > 0
        label_image = label(binary, connectivity=2)
        areas = regionprops(label_image)
        for a in areas:
            if a.area < 32 * 32:
                tiny_masks[a.coords.any()] = 1
            else:
                normal_masks[a.coords.any()] = 1
    tiny_masks = torch.from_numpy( tiny_masks).cuda()
    normal_masks = torch.from_numpy(normal_masks).cuda()
    return  tiny_masks, normal_masks
@BASIS_MODULE_REGISTRY.register()
class ProtoNet(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim          = cfg.MODEL.BASIS_MODULE.NUM_BASES
        planes            = cfg.MODEL.BASIS_MODULE.CONVS_DIM
        self.in_features  = cfg.MODEL.BASIS_MODULE.IN_FEATURES
        self.loss_on      = cfg.MODEL.BASIS_MODULE.LOSS_ON
        norm              = cfg.MODEL.BASIS_MODULE.NORM
        num_convs         = cfg.MODEL.BASIS_MODULE.NUM_CONVS
        self.visualize    = cfg.MODEL.ModifiedBlendMasK.VISUALIZE
        self.tiny_weight = cfg.MODEL.ModifiedBlendMasK.TINY_WEIGHT
        self.normal_weight = cfg.MODEL.ModifiedBlendMasK.NORMAL_WEIGHT
        self.btiny_weight = cfg.MODEL.ModifiedBlendMasK.BINARY_TINY_WEIGHT
        self.bnormal_weight = cfg.MODEL.ModifiedBlendMasK.BINARY_NORMAL_WEIGHT
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))
        tower = []
        for i in range(num_convs):
            tower.append(
                conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(
            conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Conv2d(planes, mask_dim, 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride   = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes          = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, num_classes, kernel_size=1,
                                                    stride=1))

    def forward(self, tiny_features, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                x_p = F.interpolate(x_p, x.size()[2:], mode="bilinear", align_corners=False)
                x = x + x_p
        outputs = {"bases": [self.tower(x)]}
        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.loss_on:
            sem_features = F.interpolate(features[self.in_features[0]], size= tiny_features.shape[2:], mode="nearest") + tiny_features
            sem_out = self.seg_head(sem_features)
            # resize target to reduce memory
            gt_sem = targets.unsqueeze(1).float()
            gt_sem = F.interpolate(
                gt_sem, scale_factor=1 / self.common_stride)

            tiny_gt_sem, normal_gt_sem = small_gt(masks = gt_sem)
            # # # #
            tiny_weight = torch.tensor(self.tiny_weight).cuda()
            normal_weight = torch.tensor(self.normal_weight).cuda()
            tiny_seg_loss = F.cross_entropy(
                sem_out, tiny_gt_sem.squeeze(1).long(), weight=tiny_weight)
            normal_seg_loss = F.cross_entropy(
                           sem_out, normal_gt_sem.squeeze(1).long(), weight=normal_weight)
            seg_loss = self.btiny_weight * tiny_seg_loss + self.bnormal_weight * normal_seg_loss
            losses['loss_basis_sem'] = self.sem_loss_weight * seg_loss
        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])
        return outputs, losses



























