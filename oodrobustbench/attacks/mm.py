import datetime
import numpy as np
import torch

from .attack_apgd import APGDAttack, APGDAttack_targeted


# loss for MM Attack
def mm_loss(output, target, target_choose, confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)

    target_onehot = torch.zeros(target_choose.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target_choose.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)

    other = (target_var * output).sum(1)
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def perturb(model, x, y, steps, eps, k, norm, loss_fn='mm', bs=100):
    starttime = datetime.datetime.now()
    model.eval()
    test_loss = 0

    apgd = APGDAttack_targeted(model,
                               n_restarts=1,
                               n_iter=steps,
                               verbose=False,
                               eps=eps,
                               norm=norm,
                               eot_iter=1,
                               rho=.75,
                               seed=1,
                               device='cuda')

    with torch.no_grad():

        x_orig, y_orig = x, y
        n_examples = x_orig.size(0)
        
        n_batches = int(np.ceil(n_examples/bs))
        robust_flags = torch.zeros(n_examples,dtype=torch.bool)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, n_examples)
            x = x_orig[start_idx:end_idx, :].clone().cuda()
            y = y_orig[start_idx:end_idx].clone().cuda()
            output = model(x)
            correct_batch = y.eq(output.max(dim=1)[1])
            robust_flags[start_idx:end_idx] = correct_batch.detach()
        
        robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

        x_adv = x_orig.detach().clone()

    for i in range(k):
        num_robust = torch.sum(robust_flags).item()

        n_batches = int(np.ceil(num_robust / bs))

        robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
        if num_robust > 1:
            robust_lin_idcs.squeeze_()
                
        for batch_idx in range(n_batches):


            start_idx = batch_idx * bs
            end_idx = min((batch_idx + 1) * bs, num_robust)

            batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
            if len(batch_datapoint_idcs.shape) > 1:
                batch_datapoint_idcs.squeeze_(-1)
            x = x_orig[batch_datapoint_idcs, :].clone().cuda()
            y = y_orig[batch_datapoint_idcs].clone().cuda()

            # make sure that x is a 4d tensor even if there is only a single datapoint left
            if len(x.shape) == 3:
                x.unsqueeze_(dim=0)
            
            apgd.n_target_classes = 1
            apgd.loss = loss_fn
            x_adv0 = apgd.perturb(x, y, i=i)

            output = model(x_adv0)
            false_batch = ~y.eq(output.max(dim=1)[1]).cpu()
            non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
            robust_flags[non_robust_lin_idcs] = False

            # x_adv[non_robust_lin_idcs] = x_adv[false_batch].detach().cuda()
            x_adv[batch_datapoint_idcs] = x_adv0.detach().clone().cpu()
                
        robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    return robust_accuracy, x_adv
