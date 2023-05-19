import torch
from torch import nn
import torch.nn.functional as F
import math

def sigmoid_dr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    postive: torch.Tensor,
    negtive:torch.Tensor,
    pos_lambda: float = 1,
    neg_lambda: float = 0.1/math.log(3.5),
    L:float = 6.,
    tau:float =4.,
    reduction: str = "none",
) -> torch.Tensor:
        margin = 0.5
        pos_ind = postive
        neg_ind = negtive
        pos_prob = logits[pos_ind].sigmoid()
        neg_prob = logits[neg_ind].sigmoid()
        neg_q = F.softmax(neg_prob/neg_lambda, dim=0)
        neg_dist = torch.sum(neg_q * neg_prob)
        if pos_prob.numel() > 0:
            pos_q = F.softmax(-pos_prob/pos_lambda, dim=0)
            pos_dist = torch.sum(pos_q * pos_prob)
            loss = tau*torch.log(1.+torch.exp(L*(neg_dist - pos_dist+margin)))/L
        else:
            loss = tau*torch.log(1.+torch.exp(L*(neg_dist - 1. + margin)))/L
        if reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.mean()
        return loss


sigmoid_dr_loss_jit = torch.jit.script(
    sigmoid_dr_loss
)  # type: torch.jit.ScriptModule
