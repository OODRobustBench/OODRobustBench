from autoattack import AutoAttack
from . import mm, pgd

import torch
import torch.nn as nn
from torch.autograd import Variable

from perceptual_advex.attacks import ReColorAdvAttack, StAdvAttack
from perceptual_advex.perceptual_attacks import LagrangePerceptualAttack, PerceptualPGDAttack


class Attack:
    def __init__(self, attack, model, norm, eps, batch_size, dataset, device):
        if attack == 'aa':
            assert norm is not None
            self.adv = AutoAttack(model, norm=norm, eps=eps, version='standard', device=device)
        elif 'mm' in attack:
            assert norm is not None
            version = attack[-1]
            attack = attack[:-1]
            if version == '3':
                self.steps, self.k = 20, 3
            elif version == '5':
                self.steps, self.k = 20, 5
            elif version == '+':
                self.steps, self.k = 100, 9
        elif attack == 'recolor':
            self.adv = ReColorAdvAttack(model,
                                        bound=eps,
                                        num_iterations=100 if dataset=='cifar10' else 200)
        elif attack == 'stadv':
            self.adv = StAdvAttack(model,
                                   bound=eps,
                                   num_iterations=100 if dataset=='cifar10' else 200)
        elif attack == 'ppgd':
            if dataset == 'cifar10':
                self.adv = PerceptualPGDAttack(model,
                                               num_iterations=40,
                                               bound=eps,
                                               lpips_model='alexnet_cifar',
                                               projection='newtons')
            else:
                self.adv = PerceptualPGDAttack(model,
                                               bound=eps,
                                               lpips_model='alexnet',
                                               num_iterations=40)
        elif attack == 'lpa':
            if dataset == 'cifar10':
                self.adv = LagrangePerceptualAttack(model,
                                                    num_iterations=40,
                                                    bound=eps,
                                                    lpips_model='alexnet_cifar',
                                                    projection='newtons')
            else:
                self.adv = LagrangePerceptualAttack(model,
                                                    bound=eps,
                                                    lpips_model='alexnet',
                                                    num_iterations=40)
                
        self.attack = attack
        self.model = model
        self.eps = eps
        self.norm = norm
        self.batch_size = batch_size
        self.device = device
        
    def perturb(self, x, y):
        if 'aa' in self.attack:
            adv = self.adv.run_standard_evaluation(x, y, bs=self.batch_size)
        elif self.attack == 'mm':
            acc, adv = mm.perturb(self.model, x, y,
                                  steps=self.steps,
                                  eps=self.eps,
                                  norm=self.norm,
                                  k=self.k,
                                  bs=self.batch_size)
        elif self.attack == 'pgd':
            x, y = x.to(self.device), y.to(self.device)
            adv = pgd.pgd(x, y,
                          self.model,
                          nn.CrossEntropyLoss(reduction='sum'),
                          eps=self.eps,
                          eps_step=self.eps/4.0,
                          max_iter=100,
                          batch_size=self.batch_size)
        elif self.attack == 'cw':
            x, y = x.to(self.device), y.to(self.device)
            adv = pgd.pgd(x, y,
                          self.model,
                          pgd.CWLoss,
                          eps=self.eps,
                          eps_step=self.eps/4.0,
                          max_iter=100,
                          batch_size=self.batch_size)            
        else:
            data = torch.utils.data.TensorDataset(x, y)
            loader = torch.utils.data.DataLoader(data,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=4)
            adv = []
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                adv.append(self.adv(x, y).detach().clone().cpu())
            adv = torch.cat(adv, dim=0)
        return adv
