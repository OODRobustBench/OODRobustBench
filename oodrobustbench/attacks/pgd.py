import math
import torch
from torch.autograd import grad, Variable

def CWLoss(output, target, confidence=0):
    """
    CW loss (Marging loss).
    """
    num_classes = output.shape[-1]
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = - torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss

def input_grad(imgs, targets, model, criterion):
    output = model(imgs)
    loss = criterion(output, targets)
    ig = grad(loss, imgs)[0]
    return ig

def perturb(imgs, targets, model, criterion, eps, eps_step, pert=None, ig=None):
    adv = imgs.requires_grad_(True) if pert is None else torch.clamp(imgs+pert, 0, 1).requires_grad_(True)
    ig = input_grad(adv, targets, model, criterion) if ig is None else ig
    if pert is None:
        pert = eps_step*torch.sign(ig)
    else:
        pert += eps_step*torch.sign(ig)
    pert.clamp_(-eps, eps)
    adv = torch.clamp(imgs+pert, 0, 1)
    pert = adv-imgs
    return adv.detach(), pert.detach()

def pgd(x, y, model, criterion, eps, eps_step, max_iter, batch_size):
    n_batches = math.ceil(x.shape[0] / batch_size)
    adv = []
    for counter in range(n_batches):
        x_curr = x[counter * batch_size:(counter + 1) *
                   batch_size]
        y_curr = y[counter * batch_size:(counter + 1) *
                   batch_size]

        pert = None
        for _ in range(max_iter):
            adv_curr, pert = perturb(x_curr, y_curr, model, criterion, eps, eps_step, pert)
        
        adv.append(adv_curr)
            
    adv = torch.cat(adv, dim=0)
    return adv

