import os, json
import warnings
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
from enum import Enum

import numpy as np
import torch
import random
from torch import nn
from addict import Dict

from itertools import product

from robustbench.data import CORRUPTIONS_DICT, get_preprocessing, load_clean_dataset, CORRUPTION_DATASET_LOADERS
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from .utils import update_json, clean_accuracy, load_model
from .data import load_natural_shift_data, NATURAL_SHIFTS, CORRUPTIONS
from .attacks import Attack


'''
Args

'''

parser = ArgumentParser()
parser.add_argument('--model_name',
                    type=str,
                    default='Carmon2019Unlabeled',
                    help='the ID of the model to be loaded and evaluated')
parser.add_argument('--custom_checkpoint',
                    type=str,
                    default="",
                    help='Path to custom checkpoint')
parser.add_argument('-a', '--attack',
                    type=str,
                    default='aa',
                    choices=['aa', 'mm3', 'mm5', 'mm+',
                             'recolor', 'stadv', 'lpa', 'ppgd', 'pgd', 'cw'],
                    help='adversarial attack method')
parser.add_argument('--threat-model',
                    type=str,
                    default='Linf',
                    choices=[x.value for x in ThreatModel]+['nonLp'],
                    help='the threat model for loading trained models')
parser.add_argument('--adv-norm',
                    type=str,
                    default=None,
                    choices=[x.value for x in ThreatModel]+['L1'],
                    help='the adv norm for conducting adv attacks')
parser.add_argument('--dataset',
                    type=str,
                    default='cifar10',
                    choices=[x.value for x in BenchmarkDataset],
                    help='ID dataset i.e. the one models were trained on')
parser.add_argument('-ns', '--natural-shifts',
                    type=str,
                    nargs='+',
                    choices=['all']+NATURAL_SHIFTS.cifar10+NATURAL_SHIFTS.imagenet,
                    default=[],
                    help="natural shifts for OOD evaluation. Don't evaluate against natural shifts if not specified (by default); 'all' evalutes against all natural shifts.")
parser.add_argument('--corruption-models',
                    type=str,
                    nargs='+',
                    choices=['corruptions'],
                    default=[],
                    help='the corruption model for corruption shifts. No corruption shifts are evaluated if not specified (by default).')
parser.add_argument('--corruptions',
                    type=str,
                    nargs='+',
                    choices=CORRUPTIONS,
                    default=None,
                    help='the corruption shifts to eval. all corruptions are evaluated if not specified (by default).')
parser.add_argument('--severities',
                    type=int,
                    nargs='+',
                    choices=[1, 2, 3, 4, 5],
                    default=[1, 2, 3, 4, 5],
                    help='the severity level of corruption shift.')
parser.add_argument('--eps',
                    type=float,
                    default=8 / 255,
                    help='the perturbation budget for adv attack.')
parser.add_argument('-n', '--n_ex',
                    type=int,
                    default=10000,
                    help='number of examples to evaluate on')
parser.add_argument('-b', '--batch_size',
                    type=int,
                    default=100,
                    help='batch size for evaluation')
parser.add_argument('--data_dir',
                    type=str,
                    default='./data',
                    help='where to store downloaded datasets')
parser.add_argument('--model_dir',
                    type=str,
                    default='./models',
                    help='where to store downloaded models')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device',
                    type=str,
                    default='cuda',
                    help='device to use for computations')
parser.add_argument('--to_disk', type=bool, default=True)


'''
Benchmark evaluation

'''

def benchmark(
        model: Union[nn.Module, Sequence[nn.Module]],
        n_examples: int = 10000,
        dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
        attack: str = 'aa-',
        threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
        adv_norm: str = 'Linf',
        to_disk: bool = False,
        model_name: Optional[str] = None,
        data_dir: str = "./data",
        natural_shifts: list = [],
        corruption_models: list = [],
        corruptions: list = None,
        severities: list = [],
        device: Optional[Union[torch.device, Sequence[torch.device]]] = None,
        batch_size: int = 32,
        eps: Optional[float] = None,
        log_path: Optional[str] = None,
        preprocessing: Optional[Union[str, Callable]] = None,
        aa_state_path: Optional[Path] = None) -> Tuple[float, float]:
    
    """Benchmarks the given model(s).

    It is possible to benchmark on 3 different threat models, and to save the results on disk. In
    the future benchmarking multiple models in parallel is going to be possible.

    :param model: The model to benchmark.
    :param n_examples: The number of examples to use to benchmark the model.
    :param dataset: The dataset to use to benchmark. Must be one of {cifar10, cifar100}
    :param threat_model: The threat model to use to benchmark, must be one of {L2, Linf
    corruptions}
    :param to_disk: Whether the results must be saved on disk as .json.
    :param model_name: The name of the model to use to save the results. Must be specified if
    to_json is True.
    :param data_dir: The directory where the dataset is or where the dataset must be downloaded.
    :param device: The device to run the computations.
    :param batch_size: The batch size to run the computations. The larger, the faster the
    evaluation.
    :param eps: The epsilon to use for L2 and Linf threat models. Must not be specified for
    corruptions threat model.
    :param preprocessing: The preprocessing that should be used for ImageNet benchmarking. Should be
    specified if `dataset` is `imageget`.
    :param aa_state_path: The path where the AA state will be saved and from where should be
    loaded if it already exists. If `None` no state will be used.

    :return: A Tuple with the clean accuracy and the accuracy in the given threat model.
    """
    if isinstance(model, Sequence) or isinstance(device, Sequence):
        # Multiple models evaluation in parallel not yet implemented
        raise NotImplementedError

    try:
        if model.training:
            warnings.warn(Warning("The given model is *not* in eval mode."))
    except AttributeError:
        warnings.warn(
            Warning(
                "It is not possible to asses if the model is in eval mode"))

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_ = ThreatModel('Linf') if threat_model == 'nonLp' else ThreatModel(threat_model)
    corruption_models = [ThreatModel(corruption_model) for corruption_model in corruption_models]

    opt_path = Path(
        "model_info"
    ) / dataset_.value / threat_model / f"{model_name}.json"
    
    if os.path.exists(opt_path):
        with open(opt_path, "r") as f: opt = Dict(json.load(f))
    else: opt = Dict()
    
    device = device or torch.device("cpu")
    model = model.to(device)

    adv = Attack(attack,
                 model=model,
                 norm=adv_norm,
                 eps=eps,
                 batch_size=batch_size,
                 dataset=dataset,
                 device=device)
    
    adv_threat = '' if adv_norm is None else f'{adv_norm}-'
    adv_threat += f'{attack}-{eps:.6f}'
    adv_threat = adv_threat.rstrip('0')

    prepr = get_preprocessing(dataset_, threat_model_, model_name, preprocessing)
    clean_data_dir = os.path.join(data_dir, 'imagenet') if dataset == 'imagenet' else data_dir
    clean_x_test, clean_y_test = load_clean_dataset(dataset_,
                                                    n_examples,
                                                    clean_data_dir,
                                                    prepr)
    accuracy = opt.clean_acc
    if accuracy == {}:
        # ID clean accuracy has not been evalauted before
        accuracy = clean_accuracy(model,
                                  clean_x_test,
                                  clean_y_test,
                                  batch_size=batch_size,
                                  device=device)
        update_json(dataset_, threat_model, model_name, accuracy, eps=eps)
        
    print(f'ID clean accuracy: {accuracy:.2%}')
    
    if opt.adv_acc is None or opt.adv_acc[adv_threat] == {}:
        # ID adv accuracy of the specified threat model has not been evaluated
        x_adv = adv.perturb(clean_x_test, clean_y_test)
        adv_accuracy = clean_accuracy(model,
                                      x_adv,
                                      clean_y_test,
                                      batch_size=batch_size,
                                      device=device)
        if opt.adv_acc is None: opt.adv_acc = Dict()
        opt.adv_acc[adv_threat] = adv_accuracy
        update_json(dataset_, threat_model, model_name, accuracy, eps=eps,
                    adv_acc=opt.adv_acc)
    else:
        adv_accuracy = opt.adv_acc[adv_threat]

    print(f'ID adversarial accuracy: {adv_accuracy:.2%}')

    # remove clean data
    del clean_x_test, clean_y_test
    
    # corruption + adv
    tmp_c = []
    for corruption_model in corruption_models:
        CORRUPTIONS = CORRUPTIONS_DICT[dataset_][corruption_model]
        if corruptions is None:
            tmp_c += [(corruption_model, c) for c in CORRUPTIONS]
        else:
            tmp_c += [(corruption_model, c) for c in CORRUPTIONS if c in corruptions]
    corruptions = tmp_c
    corruption_prepr = get_preprocessing(dataset_, threat_model_, model_name, 'Res224')
    for (corruption_model, corruption), severity in product(corruptions, severities):
        x, y = CORRUPTION_DATASET_LOADERS[dataset_][corruption_model](n_examples,
                                                                      severity,
                                                                      data_dir,
                                                                      False,
                                                                      [corruption],
                                                                      corruption_prepr)
        y = y.long()
        x, y = x.contiguous(), y.contiguous()

        severity = str(severity)
        if opt.metrics is None: opt.metrics = Dict()
        corr_result = opt.metrics[corruption_model.value][corruption][severity]
        
        if corr_result.clean == {}:
            corr_result.clean = clean_accuracy(model,
                                               x, y,
                                               batch_size=batch_size,
                                               device=device)
            update_json(dataset_, threat_model, model_name, accuracy,
                        eps=eps, extra_metrics=opt.metrics)
            
        print(f'OOD {corruption_model.value}: ({severity}) {corruption:20} clean acc: {corr_result.clean:.2%}')
        
        if corr_result[adv_threat] == {}:
            x_adv = adv.perturb(x, y)
            corr_result[adv_threat] = clean_accuracy(model,
                                                     x_adv, y,
                                                     batch_size=batch_size,
                                                     device=device)
            update_json(dataset_, threat_model, model_name, accuracy,
                        eps=eps, extra_metrics=opt.metrics)
            
        print(f'OOD {corruption_model.value}: ({severity}) {corruption:20} adver acc: {corr_result[adv_threat]:.2%}')

        opt_str = f'OOD {corruption_model.value}: ({severity}) {corruption:20}'
        for k, v in corr_result.errs[adv_threat].items(): opt_str += f' {k}: {v:.2%}'
        print(opt_str)

        # clean corruption data
        del x, y
        
    if natural_shifts != [] and natural_shifts[0] == 'all':
        natural_shifts = NATURAL_SHIFTS[dataset]

    # natural shifts + adv
    for natural_shift in natural_shifts:
        ns_result = opt.metrics[natural_shift]
        
        acc = ns_result.clean
        rob = ns_result[adv_threat]

        if acc == {} or rob == {}:
            x, y = load_natural_shift_data(data_dir, dataset, natural_shift, n_examples, prepr)
        
        if acc == {}:
            acc = clean_accuracy(model, x, y, batch_size=batch_size, device=device)
            opt.metrics[natural_shift].clean = acc
            update_json(dataset_, threat_model, model_name, accuracy,
                        eps=eps, extra_metrics=opt.metrics)
            
        print(f'OOD {natural_shift:15} clean acc: {acc:.2%}')

        if rob == {}:
            x_adv = adv.perturb(x, y)
            rob = clean_accuracy(model, x_adv, y, batch_size=batch_size, device=device)
            opt.metrics[natural_shift][adv_threat] = rob
            update_json(dataset_, threat_model, model_name, accuracy,
                        eps=eps, extra_metrics=opt.metrics)

        print(f'OOD {natural_shift:15} adver acc: {rob:.2%}')

    return accuracy, adv_accuracy, opt.metrics

    
def main(args: Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"=> Model Name: {args.model_name}")
    print(f"=> Dataset: {args.dataset} ({args.n_ex} examples)")
    print(f"=> Threat norm: {args.threat_model}")
    print(f"=> Eps: {args.eps:.5f}")
    
    # A hack to automatically set batch size based on available GPU memory
    # Something like this would be more robust:
    # https://lightning.ai/docs/pytorch/latest/advanced/training_tricks.html#batch-size-finder
    total_mem, alloc_mem = 0, 0
    for device_id in range(torch.cuda.device_count()):
        total_mem += torch.cuda.get_device_properties(device_id).total_memory
        alloc_mem += torch.cuda.memory_allocated(device_id)
    free_mem = total_mem - alloc_mem
    print(f"=> Free GPU memory: {free_mem / 1024**3:.2f} GB")

    model, prepr = load_model(args.model_name,
                              model_dir=args.model_dir,
                              dataset=args.dataset,
                              threat_model=args.threat_model)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"=> Number of model parameters: {num_params * 1e-6:.2f}M")
    model_mem = num_params * 4  # float32 = 4 bytes
    # Multiply by a hacky constant (2 here) based on some observation...
    batch_size = int(free_mem // model_mem * 2)
    # Make batch size divisible by 8 for efficiency
    # https://sebastianraschka.com/blog/2022/batch-size-2.html
    batch_size = batch_size // 8 * 8
    if args.batch_size == -1:
        print(f"=> Batch size is not given, setting it to {batch_size}")
        args.batch_size = batch_size
        
    device = torch.device(args.device)

    benchmark(model,
              n_examples=args.n_ex,
              dataset=args.dataset,
              attack=args.attack,
              threat_model=args.threat_model,
              adv_norm=args.adv_norm,
              natural_shifts=args.natural_shifts,
              corruption_models=args.corruption_models,
              corruptions=args.corruptions,
              severities=args.severities,
              to_disk=args.to_disk,
              model_name=args.model_name,
              data_dir=args.data_dir,
              device=device,
              batch_size=args.batch_size,
              eps=args.eps,
              preprocessing=prepr)
    
if __name__ == '__main__':
    args_ = parser.parse_args()
    main(args_)
