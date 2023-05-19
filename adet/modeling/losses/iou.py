import torch
import torch.nn as nn
import math
class IOULoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """
    def __init__(self, loc_loss_type='ciou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type
    def forward(self, pred, target, weight = None):
        # Get the coordinates of bounding boxes

        b1_x1 = pred[:, 0]
        b1_y1 = pred[:, 1]
        b1_x2 = pred[:, 2]
        b1_y2 = pred[:, 3]

        b2_x1 = target[:, 0]
        b2_y1 = target[:, 1]
        b2_x2 = target[:, 2]
        b2_y2 = target[:, 3]

        pred_aera = (b1_x1 + b1_x2) * \
                    (b1_y1 + b1_y2)
        target_aera = (b2_x1 + b2_x2) * \
                      (b2_y1 + b2_y2)

        w1 = b1_x2 + b1_x1
        print(b1_x2,b1_x1)
        print(b1_y2,b1_y1)
        h1 = b1_y2 - b1_y1

        w2 = b2_x1 + b2_x2
        h2 = b2_y1 + b2_y2

        w_intersect = torch.min(b1_x1, b2_x1) + \
                      torch.min(b1_x2, b2_x2)
        h_intersect = torch.min(b1_y2, b2_y2) + \
                      torch.min(b1_y1, b2_y1)

        g_w_intersect = torch.max(b1_x1, b2_x1) + \
                        torch.max(b1_x2, b2_x2)
        g_h_intersect = torch.max(b1_y2, b2_y2) + \
                        torch.max(b1_y1, b2_y1)
        ac_uion = g_w_intersect * g_h_intersect

        inter = w_intersect * h_intersect
        union = target_aera + pred_aera - inter
        iou = (inter + 1.0) / ( union + 1.0)
        loss_name = self.loc_loss_type
        if loss_name == "ciou" or loss_name == "giou" or loss_name == "diou":
            cw = torch.max(b1_x1, b2_x1) + \
                        torch.max(b1_x2, b2_x2)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) + \
                        torch.max(b1_y1, b2_y1)  # convex height
            if loss_name == "ciou" or loss_name == "diou":  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if loss_name == "diou":
                    diou = iou - rho2 / c2  # DIoU
                    losses = 1 - diou
                elif loss_name == "ciou":  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    # print(torch.atan(w2 / h2))
                    # print((h1))
                    v = (4 / math.pi ** 2) * \
                        torch.pow((torch.atan(w2 / h2) -torch.atan(w1 / h1)), 2)
                    # print(v)
                    with torch.no_grad():
                        S = 1 - iou
                        alpha = v / (S + v)
                    ciou = iou - (rho2 / c2 + v * alpha)
                    losses = 1-ciou
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch   # convex area
                giou = (iou - (c_area - union) / c_area)
                losses = 1- giou
        else:
             losses = iou  # IoU
        if weight is not None:
            return (losses * weight).sum()
        else:
            print(losses)
            return losses.sum()


