import torch.nn as nn
import torch
from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from contextlib import suppress
from timm.utils import NativeScaler

try:
    from apex import amp
    from timm.utils import ApexScaler
    has_apex = True
except ImportError:
    has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

def resolve_amp(args, _logger):
    args.amp_version=''
    # resolve AMP arguments based on PyTorch / Apex availability
    if args.apex_amp and has_apex:
        args.amp_version = 'apex'
    elif args.native_amp and has_native_amp:
        args.amp_version = 'native'
    else:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")


def build_loss_scaler(args, _logger):
    # setup loss scaler
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp_version == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif args.amp_version == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        _logger.info('AMP not enabled. Training in float32.')
        
    return amp_autocast, loss_scaler


def build_loss(args, mixup_fn, num_aug_splits):
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_fn is not None:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    return train_loss_fn, validate_loss_fn
