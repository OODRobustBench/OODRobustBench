import dataclasses
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import _safe_load_state_dict, add_substr_to_state_dict
from robustbench.utils import load_model as load_model_from_rb
from torch import nn

from .models import models
from .models.custom_models.utils import build_custom_model
from .models.ARES import get_model

import copy
import numpy as np

failure_messages = [
    'Missing key(s) in state_dict: "mu", "sigma".',
    'Unexpected key(s) in state_dict: "model_preact_hl1.1.weight"',
    'Missing key(s) in state_dict: "normalize.mean", "normalize.std"',
    'Unexpected key(s) in state_dict: "conv1.scores"'
]

def load_model(model_name: str,
               model_dir: Union[str, Path] = './models',
               dataset: Union[str,
                              BenchmarkDataset] = BenchmarkDataset.cifar_10,
               threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
               custom_checkpoint: str = "") -> nn.Module:
    """Loads a model from the model_zoo.

     The model is trained on the given ``dataset``, for the given ``threat_model``.

    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.

    :return: A ready-to-used trained model.
    """
    if model_name.startswith("custom_"):
        model = build_custom_model(model_name, model_dir, dataset, threat_model)
        return model, None

    if model_name.startswith('ares_'):
        model = get_model(model_name[5:])
        return model, 'Res256Crop224'
    
    try:
        model = load_model_from_rb(model_name, model_dir, dataset, threat_model, custom_checkpoint)
        prepr = None
    except:
        if model_name not in models: raise Exception(f'invalid model name: {model_name}')
        model = models[model_name].model()
        prepr = models[model_name].prepr
        prepr = None if prepr == {} else prepr

        base_path = f'{model_dir}/{dataset}/{threat_model}/{model_name}'
        model_path = None
        for ext in ['.pth.tar', '.pt', '.pth']:
            _path = base_path + ext
            if os.path.isfile(_path):
                model_path = _path
                break
        if model_path is None: raise Exception(f'model state not found at: {base_path}')
        model_state = torch.load(model_path, map_location='cpu')
        model_state = model_state['state_dict'] if isinstance(model_state, dict) and 'state_dict' in model_state else model_state

        try:
            model.load_state_dict(model_state)
        except:
            if 'rn50trades' in model_name or 'rn50nfgsm' in model_name:
                model_state = filter_substr_to_state_dict(model_state, '1.')
                model_state.pop('0.mean', None)
                model_state.pop('0.std', None)
                model_state = add_substr_to_state_dict(model_state, 'model.')
                model.load_state_dict(model_state, strict=False)
            elif model_name in ['revisiting_vit_s', 'revisiting_vit_b', 'revisiting_convnext_iso_cvst', 'revisiting_vit_m_cvst']:
                model_state = filter_substr_to_state_dict(model_state, 'base_model.')
                model.load_state_dict(model_state, strict=True)
            elif model_name == 'revisiting_convnext_t':
                model_state = filter_substr_to_state_dict(model_state, 'module.base_model.')
                model.load_state_dict(model_state, strict=True)
            elif 'pat_' in model_name:
                model_state = model_state['model']
                model_state = add_substr_to_state_dict(model_state, 'model.')
                model.load_state_dict(model_state, strict=False)
            elif 'vr_' in model_name:
                if 'pat' in model_name:
                    model_state = model_state['model']
                model_state = filter_substr_to_state_dict(model_state, 'module.')
                model_state = add_substr_to_state_dict(model_state, 'model.')
                model.load_state_dict(model_state, strict=False)
            elif 'rex_' in model_name:
                model_state = model_state['current_model']
                model.load_state_dict(model_state, strict=True)
            else:
                model_state = add_substr_to_state_dict(model_state, 'model.')
                model.load_state_dict(model_state, strict=True)
                
    return model, prepr

def filter_substr_to_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace(substr, '', 1)] = v
    return new_state_dict

ACC_FIELDS = {
    ThreatModel.corruptions: "corruptions_acc",
    ThreatModel.L2: ("external", "autoattack_acc"),
    ThreatModel.Linf: ("external", "autoattack_acc")
}

def clean_accuracy(model: nn.Module,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         batch_size: int = 100,
                         device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
            
    return acc.item() / x.shape[0]

def update_json(dataset: BenchmarkDataset,
                threat_model: str,
                model_name: str,
                accuracy: float,
                eps: Optional[float] = None,
                extra_metrics: Optional[dict] = None,
                adv_acc: Optional[dict] = None) -> None:
    json_path = Path(
        "model_info"
    ) / dataset.value / threat_model / f"{model_name}.json"
    if not json_path.parent.exists():
        json_path.parent.mkdir(parents=True, exist_ok=True)

    model_info = ModelInfo(dataset=dataset.value,
                           eps=eps,
                           clean_acc=accuracy,
                           metrics=extra_metrics,
                           adv_acc=adv_acc)
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_json_dict = json.load(f)
            # then update only values which are not None
            existing_json_dict.update({k: v for k, v in dataclasses.asdict(model_info).items() if v is not None})
            with open(json_path, "w") as f:
                f.write(json.dumps(existing_json_dict, indent=2))
    else:
        with open(json_path, "w") as f:
            f.write(json.dumps(dataclasses.asdict(model_info), indent=2))

@dataclasses.dataclass
class ModelInfo:
    link: Optional[str] = None
    name: Optional[str] = None
    authors: Optional[str] = None
    additional_data: Optional[bool] = None
    number_forward_passes: Optional[int] = None
    dataset: Optional[str] = None
    venue: Optional[str] = None
    architecture: Optional[str] = None
    eps: Optional[float] = None
    clean_acc: Optional[float] = None
    reported: Optional[float] = None
    adv_acc: Optional[dict] = None
    corruptions_acc: Optional[str] = None
    corruptions_acc_3d: Optional[str] = None
    corruptions_mce: Optional[str] = None
    corruptions_mce_3d: Optional[str] = None
    footnote: Optional[str] = None
    metrics: Optional[dict] = None


def cvt_state_dict(state_dict):
    # deal with adv bn
    state_dict_new = copy.deepcopy(state_dict)

    if 1 >= 0:
        for name, item in state_dict.items():
            if 'bn' in name:
                assert 'bn_list' in name
                state_dict_new[name.replace(
                    '.bn_list.{}'.format(1), '')] = item

    name_to_del = []
    for name, item in state_dict_new.items():
        if 'bn' in name and 'adv' in name:
            name_to_del.append(name)
        if 'bn_list' in name:
            name_to_del.append(name)
        if 'fc' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # deal with down sample layer
    keys = list(state_dict_new.keys())[:]
    name_to_del = []
    for name in keys:
        if 'downsample.conv' in name:
            state_dict_new[name.replace(
                'downsample.conv', 'downsample.0')] = state_dict_new[name]
            name_to_del.append(name)
        if 'downsample.bn' in name:
            state_dict_new[name.replace(
                'downsample.bn', 'downsample.1')] = state_dict_new[name]
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    state_dict_new['fc.weight'] = state_dict['fc.weight']
    state_dict_new['fc.bias'] = state_dict['fc.bias']
    return state_dict_new
