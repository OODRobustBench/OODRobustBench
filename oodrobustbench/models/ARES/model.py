import torch
import torch.nn as nn
from timm.models import create_model, safe_model_name, convert_splitbn_model

class SwitchableBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.bn_mode = 'clean'
        self.bn_adv = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: torch.Tensor):
        if self.training:  # aux BN only relevant while training
            if self.bn_mode == 'clean':
                return super().forward(input)
            elif self.bn_mode == 'adv':
                return self.bn_adv(input)
        else:
            return super().forward(input)

def convert_switchablebn_model(module):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.
    Args:
        module (torch.nn.Module): input module
        num_splits: number of separate batchnorm layers to split input across
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> model = timm.models.convert_splitbn_model(model, num_splits=2)
    """
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SwitchableBatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine,
            module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
            
        for aux in [mod.bn_adv]:
            aux.running_mean = module.running_mean.clone()
            aux.running_var = module.running_var.clone()
            aux.num_batches_tracked = module.num_batches_tracked.clone()
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_switchablebn_model(child))
    del module
    return mod

def build_model(args, _logger, num_aug_splits):
    # creating model
    _logger.info(f"Creating model: {args.model}")
    model_kwargs=dict({
        'num_classes': args.num_classes,
        'drop_rate': args.drop,
        'drop_connect_rate': args.drop_connect,  # DEPRECATED, use drop_path
        'drop_path_rate': args.drop_path,
        'drop_block_rate': args.drop_block,
        'global_pool': args.gp,
        'bn_momentum': args.bn_momentum,
        'bn_eps': args.bn_eps,
    })
    model = create_model(args.model, pretrained=False, **model_kwargs)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes
    
    _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # advprop conversion
    if args.advprop:
        model=convert_switchablebn_model(model)

    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        if args.amp_version == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            from apex.parallel import convert_syncbn_model
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        _logger.info(
            'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
            'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    return model