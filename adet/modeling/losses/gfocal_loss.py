import torch.nn as nn
import torch.nn.functional as F
import torch

from adet.modeling.losses.utils import weight_reduce_loss
def sigmoid_qfl_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    beta: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    scale_factor = (preds - target).abs().pow(beta)
    loss = F.binary_cross_entropy_with_logits(
        preds, target,reduction='none') * scale_factor
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


sigmoid_qfl_loss_jit = torch.jit.script(
    sigmoid_qfl_loss
)  # type: torch.jit.ScriptModule




def distribution_focal_loss(
            pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None):
    disl = label.long()
    disr = disl + 1

    wl = disr.float() - label
    wr = label - disl.float()

    loss = F.cross_entropy(pred, disl, reduction='none') * wl \
         + F.cross_entropy(pred, disr, reduction='none') * wr
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

