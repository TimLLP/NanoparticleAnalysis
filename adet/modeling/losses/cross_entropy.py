import torch
from torch.nn import functional as F


def CrossEntropyLoss(reduction, logits, target, weights):
    N = 1e-7
    # print(logits.shape, target.shape)
    n = logits.size(0)
    c = logits.size(1)
    out_size = (n,) + logits.size()[2:]
    if target.size()[1:] != logits.size()[2:]:
        raise ValueError("Expected target size {}, got {}".format(out_size, target.size()))
    logits = logits.contiguous()
    target = target.contiguous()
    if logits.numel() > 0:
        logits = logits.view(n, c, 1, -1)
    else:
        logits = logits.view(n, c, 0, 0)
    if target.numel() > 0:
        target = target.view(n, 1, -1)
    else:
        target = target.view(n, 0, 0)
    print(logits.shape,target.shape)
    entro = (logits * torch.log(target + N) + (1 - logits) * torch.log(1 - target + N))
    # print(weight)
    loss = entro * weights
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss