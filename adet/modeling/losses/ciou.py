
import logging
import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



logger = logging.getLogger(__name__)
class CIOULoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, loc_loss_type='ciou'):
        super(CIOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type
        print("当前损失函数为ciou")
    def forward(self, b1, b2, weight):

            """
            输入为：
            ----------
            b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
            b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

            返回为：
            -------
            ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
            """
            # 求出预测框左上角右下角
            b1_xy = b1[..., :2]
            b1_wh = b1[..., 2:4]
            b1_wh_half = b1_wh / 2.
            b1_mins = b1_xy - b1_wh_half
            b1_maxes = b1_xy + b1_wh_half

            # 求出真实框左上角右下角
            b2_xy = b2[..., :2]
            b2_wh = b2[..., 2:4]
            b2_wh_half = b2_wh / 2.
            b2_mins = b2_xy - b2_wh_half
            b2_maxes = b2_xy + b2_wh_half

            # 求真实框和预测框所有的iou
            intersect_mins = torch.max(b1_mins, b2_mins)
            intersect_maxes = torch.min(b1_maxes, b2_maxes)
            intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            b1_area = b1_wh[..., 0] * b1_wh[..., 1]
            b2_area = b2_wh[..., 0] * b2_wh[..., 1]
            union_area = b1_area + b2_area - intersect_area
            iou = intersect_area / torch.clamp(union_area, min=1e-6)

            # 计算中心的差距
            center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

            # 找到包裹两个框的最小框的左上角和右下角
            enclose_mins = torch.min(b1_mins, b2_mins)
            enclose_maxes = torch.max(b1_maxes, b2_maxes)
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

            # 计算对角线距离
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
                b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
                b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            ciou = ciou - alpha * v
            losses = 1 - ciou
            if weight is not None:
                return (losses * weight).sum()
            else:
                # print(losses)
                return losses.sum()


