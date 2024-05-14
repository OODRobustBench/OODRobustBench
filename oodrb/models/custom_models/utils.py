"""Utility functions for building models."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import requests
import torch
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from torch import nn
from tqdm import tqdm

from .cifar import (
    cifar_convmixer,
    cifar_densenet,
    cifar_dla,
    cifar_efficientnet,
    cifar_googlenet,
    cifar_inception,
    cifar_mobilenetv2,
    cifar_resnet,
    cifar_resnext,
    cifar_senet,
    cifar_simplevit,
    cifar_vgg,
    cifar_wideresnet,
    common,
)

_NormParams = Dict[str, Tuple[float, float, float]]
logger = logging.getLogger(__name__)


_MODEL_DATA = {
    BenchmarkDataset.cifar_10: {
        ThreatModel.Linf: {
            "convmixer_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "9c34eebe-8a9c-42eb-94f6-c5eefc48e843",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "95e3188a7fc42bb7d37c22346b14d883",
            ),
            "convmixer_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "bf05b04d-3d55-474e-9ab4-13416741fac7",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "c8fdf0bbea7721b087dd70afb8d88aa3",
            ),
            "convmixer_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "87370496-b18a-4eed-935d-2e17cfda5292",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "f45164670d37c092ed3450a97aa35467",
            ),
            "convmixer_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "7552626a-8816-4e09-bdb7-6573b71dc5d7",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "df3703d80c121d924c28be16959419e9",
            ),
            "convmixer_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "d8deef92-08b4-4be6-a42d-6269c39bc510",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "864ba4a5005793691bd04410de77d256",
            ),
            "convmixer_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "92f50cf4-e523-4a0a-94ad-6b4277e017c1",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "0f62af247a28cadf5de03a72690331b1",
            ),
            "convmixer_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "b79dacaa-9744-43be-b850-82e44b845afb",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "6b34b6656e031f3ffa8d4b22a8dc8721",
            ),
            "convmixer_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "0fc3f2ba-7ff9-4b1f-b0f4-01d40d26621f",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "a807f93231b0e46096aff105d12b417b",
            ),
            "convmixer_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "a964ce43-1dfd-4d8c-b7dd-5ef9a6428781",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "799a18611c96855f9f852fa604df535f",
            ),
            "convmixer_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "9438b9d4-551c-4aa0-a57a-cd7fc9fb4cee",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "6d65146eb66319de51d3299af77d4be5",
            ),
            "convmixer_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "1836f0b2-5fcf-417a-bae2-ebd7d8d71eb7",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "06cee24e7e4a4109a8864f997e4f21b3",
            ),
            "convmixer_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "97b92c1f-362a-4310-8255-17b8610f719a",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "b6914a529fe6f21d2fe6f00bb8bf5041",
            ),
            "convmixer_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "50dc0547-25dc-4763-b0cc-4ccedc06c299",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "46ded6e928bc5690aafcffb4ffbba5bf",
            ),
            "convmixer_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "2abb4be1-2468-40e2-a42d-6239b0180358",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "e76376d091d0444807ae4e102da6248f",
            ),
            "convmixer_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "aa0b1c18-0bc2-48c3-8aea-e4136403cceb",
                "https://zenodo.org/record/8300661/files/cifar10_convmixer_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "02d6ce23d13e1ea230f7f464956aef6a",
            ),
            "densenet121_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "8e45634f-e256-4ffa-9c1a-08f58175dba4",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "f967ab54f9fe0a50f51b924bdc697ea4",
            ),
            "densenet121_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "2f10e87e-eb62-491b-a63b-da81be05d7d1",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "8d57278cb129ce6a2fc2bbdb9ef5e1c2",
            ),
            "densenet121_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "136350c2-b65c-4d06-a377-d3a33f98a9e7",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "55cd780d204aa14be3ceca94b8925c0c",
            ),
            "densenet121_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "0daea354-af45-4692-93b4-d8142fa03f1b",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "3d726fe88e354b045cfaeaa09e094fd6",
            ),
            "densenet121_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "6d2ce0e0-8ff5-424f-891d-592ed3055ecb",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "cd8f2282d516687fd5d1fb833b9cc745",
            ),
            "densenet121_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "b5220d14-3e05-40f1-bc01-31c00c63def3",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "b09442972684ae92aea761da55b4e3d2",
            ),
            "densenet121_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "de401f35-61f7-48cd-9734-d5bb955af521",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "e05f71758ff26ca1f87194bc3035b17b",
            ),
            "densenet121_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "d79ec554-30db-417f-91a6-b64976433043",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "0dc3e89a75c3208b42f01706c1510035",
            ),
            "densenet121_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "3e89942c-07fa-4deb-bcbb-3ba390a6a262",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "cf04f88012c16a38cd3ebe500eda122e",
            ),
            "densenet121_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "a7c73107-71b7-4f48-ace1-8e85dd5b3536",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "49f5d1edf6bed20bc47f77e030f11098",
            ),
            "densenet121_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "348d1355-04d6-4f08-a9f7-be6d312921f8",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "1b433551a7f8068fb9d7e0a8b3fb0a9b",
            ),
            "densenet121_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "2986ae06-6201-47e3-8963-fd3391e486ef",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "a330bae909cda062f62bdca988b24b60",
            ),
            "densenet121_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "d3e1a01a-da76-4d5d-97af-c74ccf1aae2e",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "4af48b79145e5f74d9d8d3536b345e3e",
            ),
            "densenet121_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "678b3b4e-37db-40b8-99f4-49afdd4f3b5e",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "01f544bf477bd42b8967aee288ecaf93",
            ),
            "densenet121_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "54d2618b-68ea-4638-98a2-c5954cc5de51",
                "https://zenodo.org/record/8300661/files/cifar10_densenet121_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "112e41fc48f5af37791349df6aee1b38",
            ),
            "dla_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "d02c5af4-c59e-47b3-8beb-1f32f06676f6",
                "https://zenodo.org/record/8300661/files/cifar10_dla_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "8f0160c5fe123dfe9d53084cbe7d9e1c",
            ),
            "dla_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "997cc91e-ebcf-4176-b2bd-4b7ad6b26cd6",
                "https://zenodo.org/record/8300661/files/cifar10_dla_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "598df69a82b1de1e38457a286990a957",
            ),
            "dla_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "aadb085a-deeb-4bcd-876d-db29c6a83bbc",
                "https://zenodo.org/record/8300661/files/cifar10_dla_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "3fb11c0c24e189f2d17d6162364647f5",
            ),
            "dla_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "cbe125f6-acb8-456e-afb2-d79b4cd2435c",
                "https://zenodo.org/record/8300661/files/cifar10_dla_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "92efba9f8368f77b5668684cd9b5f9e9",
            ),
            "dla_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "37e33988-4263-4540-8f98-b94fa0c99060",
                "https://zenodo.org/record/8300661/files/cifar10_dla_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "3e82f56608706213ef7ccf63840a1972",
            ),
            "dla_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "7bef2812-cbd3-4cd4-b5b0-1da7f77d0c24",
                "https://zenodo.org/record/8300661/files/cifar10_dla_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "f3c1c86b8e373d3775f0aa6153db43d9",
            ),
            "dla_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "be5e1622-6d6e-405f-a3b4-4dfa6aab36b8",
                "https://zenodo.org/record/8300661/files/cifar10_dla_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "7402cffd1d0c7d4a485a79e2a9649567",
            ),
            "dla_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "11607faf-c0a3-40e5-9f97-e811d7a22e3c",
                "https://zenodo.org/record/8300661/files/cifar10_dla_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "954a468a3f4368957674db03fd241b16",
            ),
            "dla_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "9c21f131-805a-4c1b-b46a-2ec9fa223f9c",
                "https://zenodo.org/record/8300661/files/cifar10_dla_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "f17f07a944463ab65a5a07e321c900ae",
            ),
            "dla_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "51a270df-f944-4876-8a04-4359fe469b50",
                "https://zenodo.org/record/8300661/files/cifar10_dla_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "4b3e45896327183a4c4a640183ff69f5",
            ),
            "dla_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "b0cc5c3d-6c57-4df2-8270-c9feb2190614",
                "https://zenodo.org/record/8300661/files/cifar10_dla_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "ad1d696801e0d79d36ee6f8b78a1c4a2",
            ),
            "dla_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "2a42ab1c-ea40-435c-b0d6-960edfe99c1e",
                "https://zenodo.org/record/8300661/files/cifar10_dla_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "95d55ee466a74a72e3ee811c9988df33",
            ),
            "dla_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "69e29f4c-bab6-4433-a553-fa40811e1b55",
                "https://zenodo.org/record/8300661/files/cifar10_dla_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "7d4624b6f99b1fd4803e9d3a18984727",
            ),
            "dla_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "00de5382-8d84-4478-acf6-96041f0d2dc9",
                "https://zenodo.org/record/8300661/files/cifar10_dla_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "88e7e6482e34d0c1b0db9d4cc3de97b4",
            ),
            "dla_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "9572b0e8-1186-410d-8982-7ffbc5303ecb",
                "https://zenodo.org/record/8300661/files/cifar10_dla_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "7613463e82a1fb39a991da89b6a7e9d8",
            ),
            "efficientnetb0_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "732ea572-78be-450e-b324-89e9b0d7d328",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "bde75b7aeb71548ffd5736bc8abe692e",
            ),
            "efficientnetb0_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "cb70b6ff-41ec-405d-ab77-0578d57faf11",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "473bc245ddb79fee10c71c21f0ec689c",
            ),
            "efficientnetb0_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "b3ffb773-e1f9-40e8-9cb0-6f8bbc9af696",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "2865e05f3192df42fdf64af9431d83f5",
            ),
            "efficientnetb0_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "d2bedac7-545a-46c4-bfa8-cb684dfbb82f",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "cc2b0b02a03481d67a48398e7c60fc37",
            ),
            "efficientnetb0_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "942686bb-1c67-4d70-9d77-8aed0e19d07f",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "dcb1a40bfa5060bbfa216e8edd726e8b",
            ),
            "efficientnetb0_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "4e5ad2f3-e4b4-4803-b20c-d3be13a87dce",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "f4e3235280a0affc9dd9cb8433f14862",
            ),
            "efficientnetb0_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "956d66ca-8154-4d3e-ac6e-e546d2a30174",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "578e07a42e37f53e7e846b99a48be370",
            ),
            "efficientnetb0_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "17508feb-c057-4015-b6b9-c089b22589e7",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "bb5bcf87571d9ad9c92bb5396e7f03ca",
            ),
            "efficientnetb0_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "2502dc91-df73-473c-8e15-a7708cc4908a",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "beb20abc11a14c348b432a177d867f75",
            ),
            "efficientnetb0_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "ede355e0-8b84-495b-8da5-f744e3488ef4",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "aa474c462ca9ea00952d41abe6eb1b20",
            ),
            "efficientnetb0_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "82be3c90-9a41-45b6-8506-6d84db7cd433",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "7d51b62d12ae6f0ec4a207776a6f34d3",
            ),
            "efficientnetb0_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "a4efc6d7-eb4a-4898-a954-c9f2619fcdbc",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "b75e7462667f0036f152f0d0afcdf8e9",
            ),
            "efficientnetb0_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "04e500e0-bdb7-4cf9-86d2-078091c3a441",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "6c1e3bf902d76d3a2fb3d7f7ad4b108d",
            ),
            "efficientnetb0_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "f89e0387-491b-4974-be8b-f668238639e7",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "e13bcdb46a4c449b7f0ed4961708dc84",
            ),
            "efficientnetb0_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "8b0ad960-751c-4776-8b52-c25b1429df09",
                "https://zenodo.org/record/8300661/files/cifar10_efficientnetb0_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "126b921222bb345bb39afcd2ef476693",
            ),
            "googlenet_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "733f63ea-f7d8-4b24-935b-6f41fb92316d",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "e5b5207624e437a1566d982431382688",
            ),
            "googlenet_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "d5bb56f9-5ac4-4b8c-9a40-cfa5bcaff26b",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "cffd18bd70bc0cc923f61e1abfa13aeb",
            ),
            "googlenet_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "eaa6361b-2172-4281-a6ea-f7542ae383ad",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "8012dd0af14d407d5e7ebb39d3383e38",
            ),
            "googlenet_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "0ae0b7f1-51cf-461d-bfb5-8288e54315de",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "587a4cb652994539822613d5b29b1a40",
            ),
            "googlenet_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "629e5b2b-868e-4038-b3c2-5ea416b44ef4",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "87cc86a81235df66051188b17227f3c0",
            ),
            "googlenet_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "7594adc4-9a9b-4d06-9e2c-691cad1926dd",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "93f706355edf4b8e2eba52259e004d96",
            ),
            "googlenet_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "495c6acc-04c3-4185-b4b7-4c62a83765b7",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "85e7d514c92cb6c93ac37d3f40042a67",
            ),
            "googlenet_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "e875c60f-57f7-4975-9242-5ada642da25a",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "d87f84c46b19936057c15e32748ceb4b",
            ),
            "googlenet_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "516ee8cf-981e-4344-ab82-35a41bb32630",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "6ceec671407f2575d30b66d438f71927",
            ),
            "googlenet_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "cc7816d1-f2f9-4613-a924-8c870d81d66c",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "922d24aaa9bb97ccecd0b6c4a87ece26",
            ),
            "googlenet_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "8a26a4ac-19b4-41d4-9d38-a82fc31c1d40",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "2bf4e0485761c7bb04841b096a4a0dbc",
            ),
            "googlenet_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "3acaedb2-05d7-4683-84cc-6d9a69eff198",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "e16bceef0e7188cb458d8868e7d01b10",
            ),
            "googlenet_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "78eeb489-2bcf-4f8c-9526-25fbc1a390c5",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "4c23ab2c3b875c1fcf115b6a8014ccae",
            ),
            "googlenet_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "5deb7091-4f06-44e5-afc9-3b086abbbff4",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "39c580690695c41bb125a9134e268fdd",
            ),
            "googlenet_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "7e51f2bb-416b-46a5-8369-388dea3e6a53",
                "https://zenodo.org/record/8300661/files/cifar10_googlenet_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "88190109ff6a4932c18504faa08fc997",
            ),
            "inception-v3_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "5d398603-a1f0-472f-b9dd-7201755d070f",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "05213ffdc91019377ec68d2d1732e0d1",
            ),
            "inception-v3_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "5c4c985a-b2c6-40d6-9d85-d16e8a3f8277",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "6872094c2ee4198806d94e77e9657186",
            ),
            "inception-v3_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "da1614ca-5fd3-4c82-9d7b-12c616bcfeb4",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "833d1990dc746422ea93aab70f97fe89",
            ),
            "inception-v3_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "1b63cbac-2258-4dcd-a616-dcfe1964f027",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "e7a3c68d2e78899383e8a21f28dc1258",
            ),
            "inception-v3_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "6527a6ee-a7bd-4dea-ae19-fab45af24376",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "1bdb31d0de40d192285bcd22fb7ed100",
            ),
            "inception-v3_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "95afcf03-d31a-47a5-8233-cfb2fd967863",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "f7dd1f9395e0c087467a07284785d63f",
            ),
            "inception-v3_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "704b181f-9fdf-4a4a-8c1f-7b68e7743f1d",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "27f8c2e9551c57550db78c3dc62d1c8f",
            ),
            "inception-v3_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "416f2851-f42f-4d12-8737-58a42806a563",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "313ab4b2e97be6ed19fb4537b6e750dd",
            ),
            "inception-v3_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "9b9424d7-7c27-4e4f-9771-bf1c89e4371e",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "9dd6480e4389bfb1b1ee993bf30ee334",
            ),
            "inception-v3_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "b58778c3-3bf1-4837-a5fa-d40e454e635f",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "5f2839a4f23f086b5cb76fed76605cda",
            ),
            "inception-v3_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "214a4298-a245-4294-986e-211c011ddbc6",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "3b8cf1f1a2371cf1f5fe24cb75a3d537",
            ),
            "inception-v3_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "8db3a355-bbe4-4ac7-8789-043a426f6168",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "3bd90a1d4d28e045563d42258ae67abc",
            ),
            "inception-v3_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "47c94748-1fef-4eb3-9e3a-138f7e7b0eb3",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "afc9734e4dd1a1d5867b0bb8412a201f",
            ),
            "inception-v3_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "35e17a41-99b9-40a1-9f53-70bee5919ece",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "9d229e1e75d2b90c322a9dff32439b68",
            ),
            "inception-v3_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "4c866db7-5d9e-4c49-8f7a-02f5085a351c",
                "https://zenodo.org/record/8300661/files/cifar10_inception-v3_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "0b4ca3031656cee06f02e1b8c7254bec",
            ),
            "mobilenetv2_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "20813fa5-7c08-46d8-802c-003d83ace968",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "b949ba97f042f7201ab6fb840b72987d",
            ),
            "mobilenetv2_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "54e426ea-a4f5-452f-9a23-e0dcbb7291f6",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "4e7297200a4e00ec8e7242b298d52476",
            ),
            "mobilenetv2_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "f0a7a354-8922-4ddb-8fec-2da8a9b98362",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "3505352748ee2286b16d42a9e85501d8",
            ),
            "mobilenetv2_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "4068e3ce-847a-498c-a124-522c2f3a46c2",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "408a459d31825a87e1ad33da59732b52",
            ),
            "mobilenetv2_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "cd78dbcb-12cf-4a4f-a5b6-6a88a75dc9ed",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "5b8fdd333e2faa5efb0c17b24c31d6c8",
            ),
            "mobilenetv2_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "f3c8bf46-1815-434f-92d5-5ce0023a632e",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "11d2ee00759d06022f2436670c3771d0",
            ),
            "mobilenetv2_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "44707637-b39c-4a10-bc09-89eb4f6bfba6",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "6143827a47fe2d5a01961a3825f64d96",
            ),
            "mobilenetv2_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "c4b471d2-fa68-4124-b73b-87d632af02db",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "3858700c45883c667ba6c92d0753743b",
            ),
            "mobilenetv2_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "f310ad59-fa6f-41db-8c10-9c0777654e2e",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "2c9912d337e615c558a00ef7a3d76f44",
            ),
            "mobilenetv2_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "8fda6bdc-eefc-45a9-b3e0-c0135ab9ef27",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "e576c499c07786a252f73df1ca929c20",
            ),
            "mobilenetv2_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "fd757afb-969a-426b-8aa6-60dc9c8236bb",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "7ee354d080fa9d2c62f758dfefb3a0aa",
            ),
            "mobilenetv2_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "68701042-e510-48eb-be83-22c6b4cd1e41",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "731bfb52a3dcf2411f626f1dc3d9dff1",
            ),
            "mobilenetv2_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "06426db2-73aa-4877-94ba-167f4a3b5fff",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "ac09391d0675008f91c1b156da17d753",
            ),
            "mobilenetv2_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "d6f5972a-ddf3-4072-9a0c-ad0a73be465b",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "bd76f4eb2c24ac34a3adb84b5d5bc1cf",
            ),
            "mobilenetv2_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "35c3f836-9d1a-4b69-b378-83d14d9cd1ed",
                "https://zenodo.org/record/8300661/files/cifar10_mobilenetv2_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "108eb7fc56e56a438eefabe57bcc8a21",
            ),
            "resnet101_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "62fa309b-6727-43f2-9ddd-8a835441b5b9",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "d3811e7941c7ba03d7a2b8f1d5e3ad09",
            ),
            "resnet101_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "a2d5221a-6979-40b4-b49f-4c61b9381cc2",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "4ea633e6a8a0fa9df6e473242d6ec3ca",
            ),
            "resnet101_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "a26c163b-394c-4efc-8cc6-fca1c86e23e4",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "2bcc40ef01f8110e98f3c5c1ef47d510",
            ),
            "resnet101_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "f0002881-8a10-44fc-a644-a5526f17afe6",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "72dcdb9ab2a4bf940be616e471fa3b13",
            ),
            "resnet101_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "b9ab52f9-9bea-490c-94e1-09f1909a1354",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "9acd9ce41cb877c05fef647ed40bd0a1",
            ),
            "resnet101_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "90e77962-66b2-460a-bb1f-346216b112f2",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "92ba26b5b4d04c997d54e339ea521a46",
            ),
            "resnet101_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "db1a3d83-e40f-4a2d-9665-cd2f4e668398",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "9394a1e0741d397738216bcf8be61623",
            ),
            "resnet101_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "db62672d-1785-4468-ac68-3af13b430d73",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "dc15d49940524083f60fd320f72a6f66",
            ),
            "resnet101_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "2cea0af3-b38e-43fe-83f1-49c7b6a266d3",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "4a5b573f551edfd7c24f1a6b060ba364",
            ),
            "resnet101_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "1371ceec-3770-42cf-a59e-407b860477ea",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "cb9e79ddefa4b8bf6671cbce4bf328e8",
            ),
            "resnet101_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "fe8ff4b3-2d17-468b-8767-7ef8f6f75feb",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "3160443c2165933ffb3dc6d64c2b1d38",
            ),
            "resnet101_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "f115d8cf-7159-402c-8a7c-c2af0b10b5ec",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "dce95b7bea6f71aa6880ed21a3050aaa",
            ),
            "resnet101_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "a5ebaf71-a221-4bac-831a-5df7dbe4c604",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "4510603b7e0db99be4e4f5e595ec6cb3",
            ),
            "resnet101_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "1cd2a232-96a6-4582-b8c2-80ff645be258",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "344291a77a0b24f17f62ca0379abaab6",
            ),
            "resnet101_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "b53a999b-d3b6-4b48-bb26-27c17f18cbb6",
                "https://zenodo.org/record/8300661/files/cifar10_resnet101_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "215a8abe47715bdb25f004ccca86fb74",
            ),
            "resnet152_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "2ac2c9ba-bf04-494d-8a07-79df456eec25",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "7b5d59550d4d30dfdbb236fcfda5cd95",
            ),
            "resnet152_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "0cb47955-2738-4884-9421-79c0362b23bf",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "d9451f2a1c8b2a218973c74f8704545c",
            ),
            "resnet152_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "ea9f9a0e-d684-4d1b-9a6e-5cb3988dcc9a",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "c7ed0eace9234c437ea15e30173d5cc4",
            ),
            "resnet152_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "da2bed53-b86a-4227-9ea3-7ee5dec3ef18",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "f27cd51fc8eabf8722b754ec8c6efd5b",
            ),
            "resnet152_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "af7d1f46-bb33-4e41-badc-238342e4927a",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "2eccb7c679e5b5e62d5479c6c2ef4c45",
            ),
            "resnet152_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "17ff0b29-a1f1-4bba-b4b9-31770dcf18cc",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "79e25fd8c608fd1fa8a19db45e0e2c46",
            ),
            "resnet152_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "d7eccf2e-e5e8-48d0-9aae-e16846f6a8b5",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "fac86bef1fa926e231bc534620c71aff",
            ),
            "resnet152_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "803ef81f-17de-4a0c-aa80-4be33dc0f7ec",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "970f4ff84fa8641b1786be891b1cce34",
            ),
            "resnet152_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "a6a16e5f-5354-45fc-8f6f-67cf4796b353",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "ca52fe6c5209457c80cda378801938d0",
            ),
            "resnet152_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "d82e8119-e0bc-48e6-ad72-a6c2c6593dbe",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "80435de8a54f958b024209ccf05896ec",
            ),
            "resnet152_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "6944a338-464e-484a-a370-1d7269a702c5",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "8f0ced6a38456f4e773b03d3a5b0924a",
            ),
            "resnet152_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "5f0f20b3-5c42-451c-be97-55d95d0834a9",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "3571213b5fa630c33262cee2502e01ab",
            ),
            "resnet152_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "1fed3a26-18d3-41ff-a785-4bf687c32d20",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "ba32c4785df920a175fcae25f0f75438",
            ),
            "resnet152_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "a3a6cd26-c3bd-4571-a560-89d5f1a02e6e",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "cc89f4cc31ff33a906482f5ea46dc809",
            ),
            "resnet152_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "8c48627b-b1af-4d0f-9f32-ec80ffbb68ef",
                "https://zenodo.org/record/8300661/files/cifar10_resnet152_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "11f9cb8d00c5a88edfe938c82563037e",
            ),
            "resnet34_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "0ed3eb9a-2fd2-4920-81c6-f2ebbd56ffe2",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "e4ac4fa567d7eb397dfc627b57020f37",
            ),
            "resnet34_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "fd4a2b16-6527-4221-96ad-3d047e57f6f8",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "e28e527bc81bacaf53081d559f0383c6",
            ),
            "resnet34_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "f838c85e-c269-4f50-8722-60f537a499e1",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "fa2ecbbd71ae66029995656a98537d09",
            ),
            "resnet34_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "17789200-e64c-4acc-b640-29812efd535f",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "bd803617ee5137be9b05f1540c01adc7",
            ),
            "resnet34_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "8e78bf04-71fd-4669-b4aa-50c380375485",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "9ff11e122186c826d2b1aeefc85b0fe7",
            ),
            "resnet34_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "81e7aa31-7038-4bc7-8bed-eaefeeeae89c",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "ef637d23d5f947a40c7e68677691905f",
            ),
            "resnet34_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "41122c2f-d3d2-4753-b851-81541c834490",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "10a39dc9a842adae7874dc2724284bb6",
            ),
            "resnet34_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "08d8cc74-e057-42b8-9f23-1268ffef118a",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "1e6a2de563bf58bd46517c2201df70c9",
            ),
            "resnet34_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "dc2f2e8c-fb7a-47b4-ac17-962de8f3b194",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "4770cd0316092d1adfda69314a367b24",
            ),
            "resnet34_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "fd6c6970-e0f9-47a7-af84-6ba96e4f1919",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "d23ba3e632889e6e6b1810089ed46c01",
            ),
            "resnet34_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "82462a77-4317-4f18-8a2c-3cd1bd4e3937",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "db02eae6d6d1cbdeb3e8f5a953b095e0",
            ),
            "resnet34_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "644cd740-012b-4803-a711-38e84f41bf93",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "ff5eb6a3ea020f573d4e0ac8ac822fef",
            ),
            "resnet34_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "4dbdece9-4706-484d-b93c-34dc15dc345c",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "915007d7390f251f4c72c4329e2202a9",
            ),
            "resnet34_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "43a40134-d401-46af-8d31-c086aef7edbc",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "8f314f8be653c44e27d3e1ccad2e9119",
            ),
            "resnet34_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "b7e5647d-e716-4214-9bd4-86f90174dfc4",
                "https://zenodo.org/record/8300661/files/cifar10_resnet34_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "572c1ab98e71e3bd740f3a7b02a2b14e",
            ),
            "resnet50_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "f0841d14-155b-4e20-b4d2-f7cf7e8c2ebe",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "0195aa49cff552ae7650cde537ff8611",
            ),
            "resnet50_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "2d26ffdf-8399-4381-a6d0-c9f9f26de5a4",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "3ce7c878fd74d8797521fdd9a1f97a47",
            ),
            "resnet50_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "58dbaf0a-45bc-4762-8c8c-7cc6c6f07da8",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "8ccd5352750b9a6f46ea9cb036b9b28e",
            ),
            "resnet50_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "e299906f-f94d-4298-8622-f899c83e5ddc",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "71c677323031e149c70ea78a6b5b0892",
            ),
            "resnet50_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "cd0e0deb-f6fd-4d71-b81e-5dff43dc4e5b",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "d90de4930b47ccebc06b43a300e62ddf",
            ),
            "resnet50_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "66e07f28-b612-40d6-a957-78a64e9d7f4e",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "574f674cc97fb8d3fae3e431768537ef",
            ),
            "resnet50_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "c7384cf3-8b2f-4a50-8860-da46778a64ac",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "fba3e53c97e668d64de13ba914fc11ad",
            ),
            "resnet50_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "031b4d22-3483-43e5-9cb6-3c0952a6a91d",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "cd2cf7a0a0a71758dc2478ea68694cd3",
            ),
            "resnet50_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "1afb1477-28d7-4ff1-8fca-d97b0767d70f",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "c4c1b576fd301394593d13055254e1c4",
            ),
            "resnet50_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "749246a1-1a18-4d7b-ab3b-12474e29ca23",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "122cadb26ff26c4e8cd45446de7e8ea3",
            ),
            "resnet50_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "2ef752d0-2002-4ea2-8303-f69b7061a8a2",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "eaaf7e3191e7bf5b7f088810fa755bc7",
            ),
            "resnet50_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "5fca84f6-e589-49de-b41d-3120739db3d3",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "1215e29f49bfa0c28e8de0ac80763d09",
            ),
            "resnet50_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "33baf91e-3eb6-44f7-98d2-c9dc0ce28096",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "181a053630de2c8247d0bc1a1c45e6a8",
            ),
            "resnet50_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "9cef3d07-006a-428c-a2e8-8c7cbf841a80",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "f5199fdcae098f8ce822ebae19eec7cd",
            ),
            "resnet50_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "74d52748-8e4c-4b29-a256-b6392e26b42c",
                "https://zenodo.org/record/8300661/files/cifar10_resnet50_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "b394e28acdab0330d7ae950c8ade4b40",
            ),
            "resnext29-2x64d_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "2db2d34c-7e4a-4bb6-8058-c09998ee74d1",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "a25c1b86e19591dc1f4d036feeffc655",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "fbd7c340-9077-4df9-8ad1-b8211a07d4a9",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "ea8262c2c71da077f8cdb3e17fa30a99",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "f5bee78c-84fc-44b6-b083-895b4fba14bc",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "3a02444c26956b2e32318632b9f1333d",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "ebaa32dd-1154-42f1-94af-2f101ffa5e64",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "37056dad4966fb4a6558dd92d2b9b65a",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "e0a29a7a-01bb-4650-9d3f-e08bdd84ce60",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "a0ed9e6f5c49ce9d742b0e471f0cd44a",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "7cd1b8d3-f013-4827-adb0-f625d30a500e",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "87d0e476cd9c0ab089bd7b7369d2b7f7",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "a13f26c1-98a8-4490-a715-a328adde7ae2",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "fcd0a75eb41fb2f7990bf2886a926497",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "65b4627d-26c8-4eb7-b950-4131e1f775aa",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "20ba95fcc74d6fe531b5b39f6719eca8",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "feef6380-314a-4bc3-8057-1811e6f52fa1",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "dd109dedae9292b9ec191501899207bb",
            ),
            "resnext29-2x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "22b542fa-f50f-4977-8451-aaa17ab76585",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "1a59f5a12956a4ef9fc43a9de0a8f164",
            ),
            "resnext29-2x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "9be6e042-183a-4d93-8340-4ad1d8234ab9",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "4b6dc1e5379d7991b7ea03e4ce6dba07",
            ),
            "resnext29-2x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "1472cdbb-9f46-48f8-a074-70ea323c8dd5",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "5b299179db87b1c219361dcbe058ad1e",
            ),
            "resnext29-2x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "e7ad37b3-bcec-4add-9ce7-b5c8548dd961",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "004e746d4cb7915f46bc954a9021aabc",
            ),
            "resnext29-2x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "6f306258-e964-4267-98c0-2d03d044f069",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "c53898c479e6fd6fdff6a07bfcb85f2f",
            ),
            "resnext29-2x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "782e89e8-b113-40ba-a2d0-d691158ebef8",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-2x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "b2f3cf83ad1628f3ab303287dd71d3e9",
            ),
            "resnext29-32x4d_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "c1b167ed-8204-41dd-ab43-79f12e915c2f",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "3eec15d2c12600513e513d3463db276b",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "05c24836-58b1-4030-a0f3-33b78e66dfa8",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "7fcf08733fd9e602991ea8600c5139f0",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "a3830fd7-d9d6-4644-ac0f-e6fc26c8555d",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "0536125eff85e5228ed85222d2dc7a66",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "e46778a5-b25e-4f28-9bff-235558128422",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "780fd7472b8f520027dfb2f9a5f10d0d",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "5409607f-a59b-4ca9-85c1-8b2be1e7ff9a",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "9248168247074d38070e0196e00c49fb",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "d27160c2-74e2-4bc4-9505-0c4a5db62cf3",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "05e0eecf09744fccb9bbd4bc9cf3d850",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "bd06df61-b1cd-404a-92b4-cbf1b81ab150",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "ccf5a11832d4d8aa48213a98a108e461",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "09d31f10-e19e-4069-944a-6065102d1ec6",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "61999ddf5d010c6fd530b73b05188e0f",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "6ad9c47b-8023-406b-8934-c5eaeb3022b0",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "ec31df4829759acfbe6f6a0052cdc38e",
            ),
            "resnext29-32x4d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "28895ff0-7e15-4e7b-8845-e633e7bd045a",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "0a7d3cf0bb8dc00a12f8da207e848012",
            ),
            "resnext29-32x4d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "6218168c-8d25-4cb9-bd6d-c0f7ac8b80dd",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "fb70cad8d52955f7dea05ce538aa679c",
            ),
            "resnext29-32x4d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "c241599c-483f-4aa3-bef7-6401929d14af",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "ffc9780f447686b518aefbb5f73d97b3",
            ),
            "resnext29-32x4d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "99a21c72-5d26-4aa3-8cc3-3440eee4ebd0",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "9b4c4434005c1d0560b72b10cb628711",
            ),
            "resnext29-32x4d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "f056c037-6c2d-4966-84fc-15f573724e27",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "b199d5501944edb7e3fd5bafaac54aa6",
            ),
            "resnext29-32x4d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "4b2cb192-a3fe-4bfc-87f3-067035cd4f6c",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-32x4d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "1b3fdc7f9b33559fb1227eb4463d12bc",
            ),
            "resnext29-4x64d_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "34e98cbf-331b-4bfc-b06c-be1a0eabfc58",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "0a8439bfc11e968a64aecd70ed0deaff",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "57051b17-bce4-4c21-bdaa-e2b6c4f13c16",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "4eb17a982991b418655fd2cc94d523a5",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "5989f68c-3009-4a52-bef1-08a11393639e",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "32944d8bcac338817ca076bb28cdbc08",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "892a091f-e5f1-4e45-bacd-d92ebc7ed702",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "b3ffb64e25b3bdc7f3ad11849c4aafdc",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "13160f26-1f29-4e85-9fc1-74c5ca6234e3",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "3b1d348197a85f633d0bbaa715ef2151",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "cfa796cd-7058-44d3-8852-24ddcd4cd52f",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "58dbc792d2b0e3c28d4e26e5d67089de",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "fa05670a-317b-454b-a11e-1e863accd250",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "05944d170bfd994d7deab0a1926aba3f",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "5bb2e924-9a6b-4f93-97db-c9d26cae7120",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "e36d78b30c37a626d065f214611a4152",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "41fc1e8f-78e6-4873-855c-b6274071215e",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "0d85efe9142a557a7aeb945d61a9ef4c",
            ),
            "resnext29-4x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "3574d96a-7f89-4833-ab5e-d6cecc7ac45c",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "660765cbe7d6e6c0d5b53896842dcb2e",
            ),
            "resnext29-4x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "841d7387-4a82-4c4a-8e22-2eebea718916",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "3e03f84dd2aa8b310e68603e00d5edb8",
            ),
            "resnext29-4x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "a67ddb5f-efc2-4e1c-9d90-52093488ee93",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "1dd2a319a28dc29e917fb3e902f4f844",
            ),
            "resnext29-4x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "1388a9ff-f2db-4e2a-8eba-ad5c2651a529",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "766937f0a8bb812be6e925327e0cc1b5",
            ),
            "resnext29-4x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "af6070bf-add6-450c-a6dc-473c7bdab00e",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "4d1ea29247e5ad404d73c7a8708ad828",
            ),
            "resnext29-4x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "c03bf39f-5eef-4c8a-b6e4-0a133edff907",
                "https://zenodo.org/record/8300661/files/cifar10_resnext29-4x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "d9a42ed453a617024df434ada886e684",
            ),
            "resnext29-8x64d_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8304919",
                "9c18c84e-fb37-4d37-81f4-f18e713c3d69",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt?download=1",
                "9d168f0f32f88ac896aa5634cd2e45bb",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8304919",
                "fef68915-7968-404a-bf67-3d4e08207b8e",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt?download=1",
                "4801cbc1d7c284920f156fc3e1c6e78d",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8304919",
                "7865543d-87d7-46d1-9dc2-4024e552f090",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt?download=1",
                "6752d0a8db9fa02a32aa8a20a9d2cd26",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8304919",
                "ae8e1a5c-f554-4854-8e84-f43ae1416a9b",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt?download=1",
                "aafccd2b2a934b075030153a2d6a5e8b",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8304919",
                "e7262ada-3d43-4a06-b595-8ee74c5f5bac",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt?download=1",
                "5b8f3bf77dc97d7957b8c57cdb122e45",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8304919",
                "e30089c5-6a6a-4b32-8486-495d00ef6151",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt?download=1",
                "3a9b8a799547008b7fef963f54b1fe8b",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8304919",
                "89782a93-9cf7-4f93-9ad4-e5e210247d33",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt?download=1",
                "be5e62bcede269323e960b944912cfaa",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8304919",
                "a95adb00-6d2e-44a2-ba49-d1b82fb9bae8",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt?download=1",
                "30e099838b5659bd13fc0f32294d0f7f",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8304919",
                "c8eac385-daaf-43ef-b878-961967d27a9a",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt?download=1",
                "bbbd46207e10a3d104689551322e3e53",
            ),
            "resnext29-8x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8304919",
                "28f49a2e-2fc0-46e9-a019-382ccc6441cb",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt?download=1",
                "89db25a9e0ac000b3ae8aa9d9d6af2ba",
            ),
            "resnext29-8x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8304919",
                "92fc16ce-2906-49c4-a191-6d2e269d9b1e",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt?download=1",
                "17bfb65ca6a51884199b8c6623d30aa8",
            ),
            "resnext29-8x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8304919",
                "0f058aa4-b3ff-4a3a-bb73-5f371a1fc679",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt?download=1",
                "87f0267a5ac92d6682907151143a1840",
            ),
            "resnext29-8x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8304919",
                "835c6a75-be5b-4f96-8f08-a2e0e3a414ab",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt?download=1",
                "1659922740dc00523617c0d948237683",
            ),
            "resnext29-8x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8304919",
                "cb7ff0ec-645f-4774-b6fc-f8040a49e4e3",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt?download=1",
                "a3479ecb1011d7ef170ab690b169fcef",
            ),
            "resnext29-8x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8304919",
                "f8a06177-7f35-48be-bc2b-8272cb3185bc",
                "https://zenodo.org/record/8304919/files/cifar10_resnext29-8x64d_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt?download=1",
                "87728d24380e1c173250c064de9fcc28",
            ),
            "senet18_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "6250e1ba-44f7-45b2-880c-ee59bc773b90",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "3ef0cbe9dc86981cbe2d2fe90e342b3f",
            ),
            "senet18_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "0942f45f-e11d-45bf-a1f7-bce1ea50440e",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "19a425ff9bad743d58f15c623c14dc24",
            ),
            "senet18_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "2d065bd2-e913-4791-bed7-534b52076622",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "20d54d1178e2c94217a358c8d43b630a",
            ),
            "senet18_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "43e102a7-881a-4c83-ac7c-29b541909df4",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "c0207dcc13e340dd2633d00d9df4f841",
            ),
            "senet18_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "1100ae6e-f888-45d1-a781-088f53404d06",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "c03c7bb8148b1853c4fccf9f5e1c0459",
            ),
            "senet18_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "c1a79348-4ec0-4e79-9d4f-1f8a3f5dfa62",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "06114afd7119f6eff7dbda20d89fa097",
            ),
            "senet18_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "65d4b469-88f4-4191-ad1a-86893b0b1e7f",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "ef578ecac755038c47552dece8ab7bd8",
            ),
            "senet18_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "bc58afce-92c9-45df-a2ec-df0762ac958c",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "592d43d1d11e2cea187fe921843c3318",
            ),
            "senet18_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "94753d02-7a94-4254-ad0a-a91031147af9",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "094338dd3d7486155267f18076d58e0e",
            ),
            "senet18_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "89d20132-ace5-45a8-8790-c5988166c339",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "90fbf71c244dc141db9bb6de3bd4311c",
            ),
            "senet18_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "4386e415-8644-4b26-a2e4-b4cf2b852255",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "456cecc357a817b4e552222357f5cc39",
            ),
            "senet18_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "085da836-e50d-4b09-9237-174e26fe24af",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "735dc9edce8a0701826890a705eab404",
            ),
            "senet18_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "969a0f46-2c9e-4b23-80fb-540f2c224c31",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "c695529abbc3b058b99d1ee0d0041870",
            ),
            "senet18_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "0cf3bbdc-dd08-4c2c-acc5-04b8109295cb",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "5c2f340beef44c9b4742608283ec304a",
            ),
            "senet18_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "6c775e9e-82d0-4d9d-8352-a54577ffb513",
                "https://zenodo.org/record/8300661/files/cifar10_senet18_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "fc6a1ebb907912d2fea111b281e72fff",
            ),
            "vgg11-bn_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "b7556e99-5340-4e5f-aee0-ed597fa2eca5",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "90c09a9d6bccf5e48f41ac35958e94bf",
            ),
            "vgg11-bn_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "ba7077d2-8bdb-4040-92a0-991850331b9a",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "a6dde1daf151d6d33036344d3e64c756",
            ),
            "vgg11-bn_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "7c544d46-9982-41dd-ac8e-a875dd8bd3ef",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "9e8fd1295c7ce1573437f785c4be1add",
            ),
            "vgg11-bn_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "ff323bad-645b-4cb1-9615-556b429e6cd0",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "be8016716c9281692274d2a346e41535",
            ),
            "vgg11-bn_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "be56c666-ad23-45a9-96ed-be52e151cf92",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "3c0cc89bc4976e60426fac96846bb552",
            ),
            "vgg11-bn_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "671bea4d-635b-4727-8e12-de1813eafd50",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "35176293fd75dd7388d0ae47d1b20605",
            ),
            "vgg11-bn_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "fd0819a0-18b6-4532-9ca2-0dcbf7b39e91",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "85b67e75fab008841d9a3bf4d4b83fac",
            ),
            "vgg11-bn_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "74159141-1d88-4660-a86b-6dd3fc20ceff",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "49d7f4691438ba9804549a44933a1fd7",
            ),
            "vgg11-bn_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "7d718944-d4f4-4996-a569-6a41240e85b8",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "a098207904cc55336c5944cdad1cf303",
            ),
            "vgg11-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "2539a280-782e-4fd1-999c-8dce13c06898",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "5fe6248dce237cfb8eb53704118d84fd",
            ),
            "vgg11-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "af9f4fe8-947d-4591-9e5d-0453b5fb05e6",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "facfa86aa2809d5b6a36884a2daf72dd",
            ),
            "vgg11-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "46e47d44-841a-476f-a6dd-0647dede5ab1",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "0a8e85e1a18e0e1e567731956d80dab2",
            ),
            "vgg11-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "c0f66558-4342-4015-9be6-5f662cc72238",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "dddbff07c518256f883834584bf8ea33",
            ),
            "vgg11-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "1a7eb043-41af-4870-b9fc-d5d5da355a7d",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "9e25209deb7ea26bd3115ac27398738c",
            ),
            "vgg11-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "75959a43-e094-4c46-a8bf-8997cba81423",
                "https://zenodo.org/record/8300661/files/cifar10_vgg11-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "b5fb763de7f152fdc5e47a97252be469",
            ),
            "vgg13-bn_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "564bf264-c353-48a1-9eef-1c107516d329",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "13886dede914e060facca724525924d6",
            ),
            "vgg13-bn_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "1541d54b-c13a-482d-b5fe-193fb07a1a9c",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "604249dc201fefe0a92176702b3be380",
            ),
            "vgg13-bn_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "f5b4d532-f6b7-467b-86aa-ec7204981b66",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "9ed74f538d490c14846101fd8f2612c4",
            ),
            "vgg13-bn_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "2aed2539-ddbb-45c8-97e1-9c9687b44d6a",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "a884442500b0ba22aa4c03752e4e43fa",
            ),
            "vgg13-bn_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "2e41cd0d-30b7-435f-b7bb-d13e68c5cdf3",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "67212970bbe18f4915b60f56a7a5212a",
            ),
            "vgg13-bn_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "92e61899-11eb-41b5-8c6c-46f92bf4c76d",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "9fc6413322f7c561a057beab26993f22",
            ),
            "vgg13-bn_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "1f041e64-c624-4e40-aa12-23ea802eda42",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "088e9b528d447ae4a9722bda8aab4bca",
            ),
            "vgg13-bn_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "00c3f3c0-e977-452e-b4bd-8d3236d859cb",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "9d1ff52d18a1c947e8e4b86ae75a5887",
            ),
            "vgg13-bn_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "9f66ea2b-ef10-4dd2-a037-bcca0d1e5c2c",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "1936135e3a64391f0e183d566177424f",
            ),
            "vgg13-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "356e7ae2-7e7a-47ea-b610-c5189fa6dfb5",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "be893ee16bdb16086cc61ebda911bc0c",
            ),
            "vgg13-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "4f3d576c-0fb5-4439-a73d-be68abddf85f",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "1e13ce2bc18e0ddde054c8ea0ceb13bf",
            ),
            "vgg13-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "a01b8bea-f825-4ad0-a03a-364e4a455b65",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "400fc9e4ae5f358e84e4489773578d29",
            ),
            "vgg13-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "04861280-8eca-4ccc-a3a1-ccb511152bfc",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "4366efd87eb561b23a09b4c692c452a7",
            ),
            "vgg13-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "661a1bc8-0575-4632-b948-8db81f0f197c",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "d6abd8733af8b89e74b528132760422e",
            ),
            "vgg13-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "a1d20341-0747-4c9d-bd4a-5d44949755e9",
                "https://zenodo.org/record/8300661/files/cifar10_vgg13-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "95167591b778b86600dfddd39859c089",
            ),
            "vgg16-bn_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "3d9db29a-71f0-48f5-8ed0-c56c7d681769",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "a04ade82c4fe16dabc2a223de420d0e6",
            ),
            "vgg16-bn_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "d17b953f-d7a8-4486-9c5d-465dd80702d2",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "c1a90391685a199815d9d9da7d17bd9f",
            ),
            "vgg16-bn_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "c6be34dd-6728-4ebe-a1cd-6eaf5453c8f8",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "7dca6d75333783b82a1d3e7a5f8e9d5a",
            ),
            "vgg16-bn_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "07da1f0d-5d49-432b-9791-f51c2c50de42",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "013061299b7e96021f5d41958ca49c9a",
            ),
            "vgg16-bn_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "479ed3a9-22de-409e-af28-c7f35020a031",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "2bed2352f04a80509593719a334de73e",
            ),
            "vgg16-bn_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "b8e0acab-8dd5-490c-9a66-ea1cff1ae5d8",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "5c9de711e0a193a231950951363959a8",
            ),
            "vgg16-bn_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "6169c517-6efb-4d92-b333-5153c549d74a",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "eb93d24e1fba3927d3db3076d8918202",
            ),
            "vgg16-bn_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "ba552fd6-6471-4258-a284-a854bb52d91e",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "b509c1183c823308099ce7387f9fb283",
            ),
            "vgg16-bn_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "988df6b5-e6c7-4995-87b4-ff1de4ff79fe",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "88b83c07e99e5e512a8cbaff208c7e45",
            ),
            "vgg16-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "50091a28-93d1-48ac-8371-2be606d983f0",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "5cee4d9b18db5f0a1b23ef1f1dea34aa",
            ),
            "vgg16-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "c595e4d4-a0f1-47fa-a70d-6230db33a7c4",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "31736e10bb8becce0ec71548d2f78d0d",
            ),
            "vgg16-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "086449f6-d038-4b5f-af78-e5fe24322051",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "ba1c87345370e17c7a72568606572721",
            ),
            "vgg16-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "6d6c5ae5-5d4d-4975-bf9c-0259fd4c3777",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "63f69c94e165543a14968299025171fd",
            ),
            "vgg16-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "4e68c8f6-c7b2-4d09-908e-6e1ea4aa99dc",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "a4ccac8868d199750001c17e374cc497",
            ),
            "vgg16-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "6fcbd753-4ff6-4fd2-90ac-ff75e1c8c953",
                "https://zenodo.org/record/8300661/files/cifar10_vgg16-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "a38247337a6b8d9e3ec80391271ea6a0",
            ),
            "vgg19-bn_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "c8acfef1-f2d0-4842-bc26-0825848a0c19",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_lse_lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "f0166284d764d6b88aa1b74799ae871b",
            ),
            "vgg19-bn_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "e6fd891a-cee5-4b01-9d4b-9b32973cabbb",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_pgd_ce_seed0_bs128_lr0.05_wd0.0001_sgd_50ep_eps8_orig.pt",
                "f5028965e19149968a55af227fee9c86",
            ),
            "vgg19-bn_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "ebc11557-297c-4b91-8c43-a9fa49b6bc72",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_pgd_ce_seed0_bs128_lr0.05_wd1e-05_sgd_50ep_eps8_orig.pt",
                "e3e787350b7c62d52352692f4026bdae",
            ),
            "vgg19-bn_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "74c422b0-5452-479e-ab28-b051a74a788a",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_pgd_ce_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "9619d90f77e6d43d286f956ea2180eba",
            ),
            "vgg19-bn_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "33e2870d-0fe1-49c4-8767-7ebc18f078fb",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps8_orig.pt",
                "eaa5c90a64cb62a4f1d94f2801d73098",
            ),
            "vgg19-bn_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "5d5c06a3-6405-4a45-a11b-a6ed2e604d8a",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps8_orig.pt",
                "527f4fc583ab51574364367dc6bc7c61",
            ),
            "vgg19-bn_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8.pt": (
                "8300661",
                "65a71429-5509-469d-b02f-5e1643512d06",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps8_orig.pt",
                "a1e23c03530be2fca4dbd9fc798a2e0e",
            ),
            "vgg19-bn_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8300661",
                "c462eed6-f265-4bcb-b0ba-91c84c1129f2",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt",
                "0c030ef93b26ff020e95acbc6cb8ff99",
            ),
            "vgg19-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "a3a1eeb1-29b2-4af6-bf3f-084a392ca513",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "b3e4c6353f09e5337fddb6378f1bf111",
            ),
            "vgg19-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "3be528d8-67f0-4088-9f50-7cb300c30a95",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "9e0991c6a9c453684315c66c892efeef",
            ),
            "vgg19-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "efaa4a00-c32f-4fda-9184-bf5d009401eb",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_trades-lse_trades-lse_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "7aa32676abca75e0abf3c78c2c2c9996",
            ),
            "vgg19-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1.pt": (
                "8300661",
                "156105e1-cf0a-4bfd-927a-c68f4fdfd918",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.1_orig.pt",
                "9b0c1ede5218c54b4617e27a85116e32",
            ),
            "vgg19-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3.pt": (
                "8300661",
                "a0b08470-0d91-48cb-9a03-5256fb8b237f",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta0.3_orig.pt",
                "141c59e9141137658ef246d5aaa47b27",
            ),
            "vgg19-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0.pt": (
                "8300661",
                "6c0f847d-75b5-4926-93ac-b7a11d5544c0",
                "https://zenodo.org/record/8300661/files/cifar10_vgg19-bn_trades_trades_seed0_bs128_lr0.1_wd0.0001_sgd_50ep_eps8_beta1.0_orig.pt",
                "537bd1e0d41dc1dcbe6be4facf243cfc",
            ),
            "vgg19-bn_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8.pt": (
                "8304919",
                "ebc3a8da-7c74-430c-bf69-0eb2c1ac0c2f",
                "https://zenodo.org/record/8304919/files/cifar10_vgg19-bn_pgd_ce_seed0_bs128_lr0.1_wd1e-05_sgd_50ep_eps8_orig.pt?download=1",
                "65913862d0de4b06a4db3893bb3b8e7f",
            ),
        },
        ThreatModel.L2: {
            "convmixer_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "a54aa648-dc58-42a9-8eff-eda95ee3e5d5",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "c92d980cc91bf82d9cea56eebad120e2",
            ),
            "convmixer_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "775c74f5-e781-4d95-8738-bc85a4a2ae6f",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "f5da699673cbf1ca279be8fce403518f",
            ),
            "convmixer_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "32340cb7-562f-443e-aa9d-1feef630a95e",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "1e0711aedc23aab90636a0203f80a8e2",
            ),
            "convmixer_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "3d8c6612-cdbd-41c4-aa2b-5253411730ea",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "5a12d4f8d34a299650821b1f2ca5b5a9",
            ),
            "convmixer_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "f7c6c83e-2610-4351-a6ff-0f9487ed359c",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "1f03607e04444f0bd25c0d7bf12a9d47",
            ),
            "convmixer_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "bbf26a8f-c39c-4fd5-9235-e75da6014d3c",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "1522aaf6928d92cf24d45efcc70ebdf3",
            ),
            "convmixer_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "60833031-a857-4bae-a3a2-a86f9d606747",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "ce6a0c89653c0d3fe00bf649155e96b5",
            ),
            "convmixer_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "009c4d47-e9d7-4c17-b355-73e456417600",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "38f2e52dd7795ab4b8c6256416d82e98",
            ),
            "convmixer_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "163a193e-4ff1-4a33-8a70-61cb9e619672",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "f23e70cfbeb64569fcf102745e355901",
            ),
            "convmixer_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "ef7deead-c02d-4352-a33b-3d7930ec54fb",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "03985d111b44a6128f034ab8c6a5c5ef",
            ),
            "convmixer_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "fb321213-2912-41c9-a90b-949e15fd8f9a",
                "https://zenodo.org/record/8285099/files/cifar10_convmixer_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "dccf5a91d5fa95844714ae1114436947",
            ),
            "densenet121_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "2400ddbe-b97c-4a74-8030-ec8d9727ea80",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "854d5fd58b120930a4e2b25cc6ccc267",
            ),
            "densenet121_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "f2def789-922f-4eb7-8263-4f19407bab12",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "65ed317cb4abf0511064b66560978cd7",
            ),
            "densenet121_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "6e2bc93a-4a7a-4534-9999-151e49e4a364",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "e3d1281d687ce57450d89f1f22f1511d",
            ),
            "densenet121_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "5292a0f4-f207-417b-9bc2-524a5237f41d",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "40aa0b97d35752a13ad06e5957097b66",
            ),
            "densenet121_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "6184a6e5-59e9-42a3-be7f-3abdcedcc649",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "956ac06289df455691c96feadd5c6dd3",
            ),
            "densenet121_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "f807b4c0-96c2-450b-a8dc-143aacaa900c",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "719281b896d788e955fff1d98c0dc7fd",
            ),
            "densenet121_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "a394adb1-2f44-4954-9c19-5a169bfab047",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "d5f7f3d13857cc8fff97f26aa478c8c6",
            ),
            "densenet121_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "ea27fae2-3a58-4e68-aca7-248e5af7502b",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "8389f98c4fdf0ef282355ddd5ad713f1",
            ),
            "densenet121_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "637ae013-968d-4dc5-a4a7-db17b5df33e3",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "89da719dd2ef9e7a673aca78fb3b5468",
            ),
            "densenet121_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "4aa6e42b-e9ee-4f66-a409-39b3fb866434",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "65a3163d7bbdb16245b857b71eda8e16",
            ),
            "densenet121_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "c6efacc4-b841-4f52-a11b-77b12cb7818e",
                "https://zenodo.org/record/8285099/files/cifar10_densenet121_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "4437e95d1101c49998887faeec14026b",
            ),
            "dla_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "f2bcfe30-ea0e-40d8-9a1c-4689c6e2a382",
                "https://zenodo.org/record/8285099/files/cifar10_dla_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "b9490c27f0aefec9e21b8b3141b35f83",
            ),
            "dla_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "939c8a5a-a3c7-4afe-987c-4045a5a3ffc5",
                "https://zenodo.org/record/8285099/files/cifar10_dla_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "be865187c2c2b1075ead8a2d2e282f7b",
            ),
            "dla_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "f6d49bab-6cf8-46dd-b849-46f3de1acc0c",
                "https://zenodo.org/record/8285099/files/cifar10_dla_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "13c96efdacd09d3b9046eb33b12e05c4",
            ),
            "dla_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "e202aab5-a1c4-4a2e-886c-8b30439a9608",
                "https://zenodo.org/record/8285099/files/cifar10_dla_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "a88f4b100ba6bfa1117b704b924d765c",
            ),
            "dla_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "4704c1f7-a16e-4ae4-9653-6d63cce3bf13",
                "https://zenodo.org/record/8285099/files/cifar10_dla_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "9ac1888988fe1d730ad6b4b0f5f3f8da",
            ),
            "dla_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "86fb5e11-e333-4bd0-9298-0056f7f959ac",
                "https://zenodo.org/record/8285099/files/cifar10_dla_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "2c968bc3361e9053f4058548200ac40d",
            ),
            "dla_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "8e886c6f-6289-4997-baa9-98208057b388",
                "https://zenodo.org/record/8285099/files/cifar10_dla_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "0785e467113fa7f8c078fdb97bad32f6",
            ),
            "dla_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "43a6577f-2db0-40df-bff7-dba0dd47ac0d",
                "https://zenodo.org/record/8285099/files/cifar10_dla_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "3f71956cc296091c107ba30b62d6eb3d",
            ),
            "dla_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "69a087da-cb66-421b-8a4d-a2d464108bf7",
                "https://zenodo.org/record/8285099/files/cifar10_dla_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "c2cac529f62e31a908db0597882cad50",
            ),
            "dla_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "c34d7968-701a-43b7-b038-02a1bfd38d1f",
                "https://zenodo.org/record/8285099/files/cifar10_dla_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "32306eeb9380e63d3e69a462ad55504e",
            ),
            "dla_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "6e79858f-560e-42f3-8171-c0f21b2ce7f3",
                "https://zenodo.org/record/8285099/files/cifar10_dla_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "cf734a6b7e9a22c62dff3d7a101d7ac5",
            ),
            "efficientnetb0_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "28850a85-98e6-4f25-b5bc-9fe6f9fef370",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "abd4d8877d4ecac639260d5eab0a1197",
            ),
            "efficientnetb0_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "80907c24-9ac1-4838-9f5c-e318e4cbe72b",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "78650feb535c193dfb6bc77f369e29ad",
            ),
            "efficientnetb0_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "e571a9f1-2cb5-4f6b-90af-26d4022ef30c",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "2b35703e6e2467ba2bba77514b07ec8e",
            ),
            "efficientnetb0_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "14e0991b-4c92-4599-9117-8a6de2a12357",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "a2f297b56b9c0b6467688cb2c693eafb",
            ),
            "efficientnetb0_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "1fd31414-22cf-4173-9891-d74ef19acc15",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "65c6100b11f8f73be657627e958b0c97",
            ),
            "efficientnetb0_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "3a37c9d7-c917-4b56-a80e-717cb72254af",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "030c89e186a92d939749eb88f1dab904",
            ),
            "efficientnetb0_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "0888d52e-6742-4935-a03f-74badb50be70",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "b8d085dcf5d5ff9609693fb6332779cc",
            ),
            "efficientnetb0_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "38da134c-b359-4cb9-abe7-dd281670545a",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "e045179e1f236a12ca700b7c66926a1f",
            ),
            "efficientnetb0_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "69f6c15c-e83f-4c64-af97-7a314a305fd3",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "5040d63bf08123e81a34b636d2fee4bb",
            ),
            "efficientnetb0_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "853e7ddf-1a48-4b43-a444-04c68034ab28",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "1749dba526b6ae29d71e96ca00faf68e",
            ),
            "efficientnetb0_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "71d68e7d-6097-4223-b9c5-28f4b3510572",
                "https://zenodo.org/record/8285099/files/cifar10_efficientnetb0_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "9b3dc351b8e4290e1a1e2705e8841eaa",
            ),
            "googlenet_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "d368cf4b-da85-4fe1-ad62-0cae7f55bd1f",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "5d4e39ce9fb024bf8c83c3e5a33c4755",
            ),
            "googlenet_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "f228801a-b0eb-46dd-b095-71b1238ca664",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "1a91a09740a9cb7ddfa70f20ab79528f",
            ),
            "googlenet_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "b6ed591a-91fb-4e08-8f91-4a60f2951527",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "1330ad9cddf51718bf6b185c922cbe0b",
            ),
            "googlenet_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "6fee871c-bfd2-4de3-9e6f-78e29836bb1b",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "c3a515d72b3c271eeda4706151c20b92",
            ),
            "googlenet_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "6ad6501a-96a5-4a50-9996-03b236cfb29d",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "1e65202e590de15e592d8b3261601652",
            ),
            "googlenet_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "1092100a-72a3-4579-8f99-36a1f5694e84",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "dc70d88ca59a30188f02d3ce543a73b2",
            ),
            "googlenet_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "5f853dd8-2812-4abd-948f-56d70577a0ca",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "9e5da77d506208e5679872e39abad491",
            ),
            "googlenet_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "d365d343-d10b-4af4-a145-29c350062e48",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "cbb90bde4292d33152be83f2de90a621",
            ),
            "googlenet_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "01502781-db68-4406-8cc8-88a092299029",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "0de32c42501e7aa1b0af175f46010af2",
            ),
            "googlenet_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "9dc151a0-1770-43a0-a741-e6ed54f7234e",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "e77859c507eb9e0626ab630b0b4fb590",
            ),
            "googlenet_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "e78e3644-b2bb-4227-8bcb-cf3da677048b",
                "https://zenodo.org/record/8285099/files/cifar10_googlenet_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "4689fa3067d90d5509eb77e2278a0074",
            ),
            "inception-v3_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "075b9c0b-81e1-43a6-a95a-cee58ae67b80",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "a66cce7b78178a7d8672e5b3fb5f6eb2",
            ),
            "inception-v3_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "b64ae06f-32fe-46cc-8f8a-5ee7b24f2aa3",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "d9c6ac26fa162c073c77f79799819a01",
            ),
            "inception-v3_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "17c08ea8-b604-46ef-9fae-9ec95b682cd5",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "9b00bf41f1e72a1743761e2a1e61e183",
            ),
            "inception-v3_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "0403313b-70f3-48a8-baab-031c1462ce90",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "3d437d8ab2fbcff41aec100f45be2d18",
            ),
            "inception-v3_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "9ec1f53d-e727-4cb0-9486-b91f28a720c5",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "c12be4bbd59c21c003dbc67f45b78070",
            ),
            "inception-v3_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "4733741b-a7fe-4e2d-a841-9ef1b1335a84",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "29208cc47b3df8eb3c59e5b58c1fa8ca",
            ),
            "inception-v3_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "ba39ab42-5eef-44e4-9bed-2f1b9f25acf0",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "3ad8ab6755e77980d8ab01ea979d7ad2",
            ),
            "inception-v3_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "b708c134-9502-49ad-8b05-f54debfede2f",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "bc20a75dd24a82d060a4e12501e41b42",
            ),
            "inception-v3_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "3a1fc8d8-458d-48ca-9ed6-ecc7ee2ceef5",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "6d7d70d2b5608d661249f992ebbf31d0",
            ),
            "inception-v3_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "e8838da5-8da0-4553-bc2a-01f9fb6de8e0",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "be94adb748253143e5bd42649652807e",
            ),
            "inception-v3_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "61a487d1-3096-4740-ba09-3867cca74eb0",
                "https://zenodo.org/record/8285099/files/cifar10_inception-v3_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "64ed80397fdfc52408be12635a2f6c48",
            ),
            "mobilenetv2_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "3b984043-9fbc-4f8b-a17b-0e0365e2ae4e",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "a8a651f68e2a35af3c624da5eff2d67e",
            ),
            "mobilenetv2_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "76463e3f-0b0a-4b26-8ad7-4d196d992773",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "1364e67a35ddcfe1b158fbf2cabf6c98",
            ),
            "mobilenetv2_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "25ac27d9-94a1-4729-97c1-e1a2d667b19a",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "911063101651e367466ff3a5be114a1e",
            ),
            "mobilenetv2_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "6220e1a7-9f5e-40d5-82e0-68544072ceb3",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "b816e14bd476b2927d0bed2a32070478",
            ),
            "mobilenetv2_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "f21fdc2a-69f5-402c-94a4-5d84edc891db",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "8ca00f2954e65a2889cec5b6568910f4",
            ),
            "mobilenetv2_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "f7a7bb2c-7a06-4146-a47b-7b74ff2ed830",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "9a047d4bede2f44d3d28ba4382d198da",
            ),
            "mobilenetv2_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "96403f67-e4a9-4b6a-ac02-f60c92fe6c06",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "d65c5dafc60597bea2d2112d30b7d479",
            ),
            "mobilenetv2_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "4ad30f70-3148-45e4-a7f1-5686e5fb54af",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "5b3636239a78aa15298fc96ccbe19752",
            ),
            "mobilenetv2_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "b17a011d-9c5b-445e-950f-2d1520a4bc74",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "c36513f4b157f5e6c345b80e4c81d18a",
            ),
            "mobilenetv2_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "c5a777cd-f1c4-4a4a-ba1b-1c6ba9d56c50",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "69ead021ad9553fc898f08c8ac0fde81",
            ),
            "mobilenetv2_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "c4d3f55a-f6c3-46a0-a412-b88323dc5339",
                "https://zenodo.org/record/8285099/files/cifar10_mobilenetv2_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "4eca298feb7d13b1b109651427085b59",
            ),
            "resnet101_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "34b9f5c7-764f-43b6-8bdb-29f4536e6231",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "8676beff34b8717cf0e57de5e31317d2",
            ),
            "resnet101_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "4c01a991-d406-4147-b1ed-95667407c865",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "a1520dd122bc7a3240d92bd401b607e8",
            ),
            "resnet101_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "1cb2aada-9ff1-4d0b-83c8-eec9810e51f9",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "22dddde29527130664c9cc91bf8d3606",
            ),
            "resnet101_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "5977571c-7b9d-4077-a2fa-e9caf137186b",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "c5b5fe783a8e008b45e8426d11820c7a",
            ),
            "resnet101_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "ef638241-4afd-4926-990d-1fe1915ac8b7",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "18d268962bc8d29a88e012ae8b30b0fd",
            ),
            "resnet101_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "77cbc3a0-8836-4f91-8187-17d5262014ab",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "31f0f53d16946d6c93715c34aaf281f5",
            ),
            "resnet101_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "7803b63e-5be6-4dc2-8fc8-8dec250ab967",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "592e79234adf846e54403b7c4b2bb3f2",
            ),
            "resnet101_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "7a706f71-d19b-4788-9036-c4772ac7a9ca",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "a23c7cbc6b00130fabec6c31b20c9676",
            ),
            "resnet101_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "c4b56c2b-a62d-45d0-a802-9cfe43be9642",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "12bfe801fcb3decb5d0779cff6152283",
            ),
            "resnet101_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "e7750158-ae9c-4807-aa57-68d5ebcfac26",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "a050b864abd9663ae0363d4859cb9d11",
            ),
            "resnet101_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "0c7822e6-276e-4687-a329-67202de65776",
                "https://zenodo.org/record/8285099/files/cifar10_resnet101_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "747b313e4e773b89b5b2addc0f7e188c",
            ),
            "resnet152_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "cd89443d-8571-4de0-a472-3a5b268827bb",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "eff9971c74d9ba765204528914a3fe7b",
            ),
            "resnet152_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "1c563887-4a62-4ab0-ada2-bcbc2be63fb4",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "a2f410be6acf8967240fba5c85d30a19",
            ),
            "resnet152_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "bd524109-660a-4c62-9cff-8e0b5550e75c",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "14f11d5081d5d1ce15f49cf18a919f99",
            ),
            "resnet152_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "af99f917-85c7-4950-aa83-4a6deadc59de",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "6840dcdc2e40ccd641dfabf3a749a638",
            ),
            "resnet152_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "44efb435-ef0c-453e-a64a-ae6002b6a9e6",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "b8cd88fa6ab5f5ee63b0784027ddba9b",
            ),
            "resnet152_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "c6de47a9-bfb3-4f30-bbd4-7918c2d5d587",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "0111cc38a73be02bbd3785e06fa91f58",
            ),
            "resnet152_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "db068900-91e7-483d-aee1-93bfd61e0bc9",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "442dd305e7060acc50d7c7cc9df8306a",
            ),
            "resnet152_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "0abcca82-3a53-426b-980d-c05cd6e62ede",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "8fdacba7c325a5923e5108b2433c33fc",
            ),
            "resnet152_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "e74126a3-a7ab-4139-bd3f-dc9af021b3ce",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "c85c5bcafc70827ee17192201234c3a9",
            ),
            "resnet152_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "e31eb5eb-1d5b-4d8d-bf29-54981a1fe825",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "13336ea2728037a52e4df58aad9de013",
            ),
            "resnet152_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "f652aee1-acc3-4147-953c-81e0ce6ca88f",
                "https://zenodo.org/record/8285099/files/cifar10_resnet152_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "dc1502d30e881e625bd97eb5e6ae2d1e",
            ),
            "resnet34_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "b5afd21a-32d4-4a99-a781-0eac42bfc8a7",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "09aaaf3c20ee62df98a1dd637be7fe3b",
            ),
            "resnet34_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "3b5e093f-1852-4d6d-8725-9aeb0ef209b8",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "4e6ff38c439125d75929f1c2006f3b9e",
            ),
            "resnet34_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "474dfb0b-a1c5-4f7b-9419-03a4944a8cfb",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "9f735409d86dbc4e93451a5272c9de03",
            ),
            "resnet34_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "f490e5c0-4cbb-4efc-8d65-cf85b7a5cb13",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "88d07c9201007ccc6ee30f9842335a16",
            ),
            "resnet34_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "795607aa-5c50-4684-a364-3ef9b639733a",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "53888cadcb646a472db59141d3e724f6",
            ),
            "resnet34_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "c8792934-28c8-486d-bebf-399261aa9db6",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "dcaabd444fa828c181acf564bc2f4e84",
            ),
            "resnet34_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "475dfbda-8e59-4988-a849-6623bb870176",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "ec36831ca2657ac6d3093dd41f8f9cd2",
            ),
            "resnet34_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "88d48d41-4c39-4339-8ce5-8e2ea38abd6e",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "acb3b30c6ae525e03be32e9ef3de240e",
            ),
            "resnet34_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "8df69b41-9f94-4882-983c-5e604fc27578",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "32267da40d9b5c1dab44cd5f47fc5017",
            ),
            "resnet34_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "a60e1124-8208-465e-906f-a5e2f9995266",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "aefedbc3a9a292122c45eb0ffcae381b",
            ),
            "resnet34_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "3f076d2f-d310-44c7-a574-d59d50a786dc",
                "https://zenodo.org/record/8285099/files/cifar10_resnet34_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "c3d28e42793f1c07e904a508bda54afa",
            ),
            "resnet50_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "a23d2cd4-2d8e-4f4f-bb66-70836adc7cf3",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "c73652467c59fc7a9678183cf15d464d",
            ),
            "resnet50_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "3a81bf9f-fb6b-427b-a35f-0e9627b62896",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "d4e3db1e84d72bd6aaafe3aac9b9f29a",
            ),
            "resnet50_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "0aa288f6-34bc-4e24-aa36-57110be65d74",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "36545ab7ad600e6c7c11a2a6ba6c25f4",
            ),
            "resnet50_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "33439f6b-c834-4a6e-aa88-848109739f41",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "85be2fb6efa001297899ff5f99f107f2",
            ),
            "resnet50_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "6da44889-15da-452a-bc0b-8cfbcef737dd",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "5012ab40a3a0a615997fcd0fb5a7c6a7",
            ),
            "resnet50_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "fdaca994-2113-4ce2-b169-17079d031534",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "b00434cf8e019740669b23a779003289",
            ),
            "resnet50_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "ecacc0e0-236d-44cd-be13-7e394522ee38",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "ded271d74eab8e9bf41218592fcf5966",
            ),
            "resnet50_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "21092547-d335-484d-b3ec-2f3e950dc6e0",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "608885e13421c415767d4f70f423935f",
            ),
            "resnet50_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "0919676f-9474-4b4a-b3e7-a7d396448c6b",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "c8a7946bb1562a57894bf7992e58fe06",
            ),
            "resnet50_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "73e4a4b8-f94a-4359-a19a-2d8cb498a5f1",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "17a24491e819802ea6cec623176d9f05",
            ),
            "resnet50_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "8d3695d5-aa45-403c-818f-b8aaa3695d0e",
                "https://zenodo.org/record/8285099/files/cifar10_resnet50_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "c5b086ea36a0ecd39883c2de3402a70e",
            ),
            "resnext29-2x64d_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "d0c87468-da3e-4e18-a23a-ec8638bc8777",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "b45cb70842434e0d114ffac3fe5f3494",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "7fc3c5e2-82fb-4891-bddd-740433b34b1d",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "f3448a8fa73f4615120aa69e331fbce2",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "48ec7670-4fb5-415b-a1fc-68b4edb39952",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "e42d364ce5a0d15b0a3cc852318c1ff8",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "30e1473c-6f0b-4555-aa74-1ed16d5162ff",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "06ad8f3b66256a49e7b9e2bd6f892031",
            ),
            "resnext29-2x64d_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "123fffd8-0f8a-44fe-99ca-82b605561232",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "dc6bad0f0649e86ac848758ed35df113",
            ),
            "resnext29-2x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "68e6bf3c-928d-4e9d-ad01-94e75480f673",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "7671d9d49b041c26b638c308f14993a7",
            ),
            "resnext29-2x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "85368f83-1faa-4c56-9b14-902f278134de",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "8eb466363ce2490698a15173f8684a0c",
            ),
            "resnext29-2x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "ce19911a-5837-4a7e-a473-83ad668154fd",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "b76d3c2e21fbe54216c3b6c6345553de",
            ),
            "resnext29-2x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "8aa13d9a-423b-4a59-b881-f999352ff97e",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "613cb8dbdf2d99762cf98388d857b264",
            ),
            "resnext29-2x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "12bfe1cb-3523-45bb-9458-23218d40261e",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "53acfbffc541e8d27def2729399875f7",
            ),
            "resnext29-2x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "d4db2091-c5a4-4c59-ba0d-64858786e86c",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-2x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "9e54c59294fa24de7022423ce4e67107",
            ),
            "resnext29-32x4d_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "9e92673e-9ec3-4041-a995-3d55c3574259",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "dffc23d5fe2fd2635dea42c2411f5ca5",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "88a62564-8ac6-4150-9607-d972360a908f",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "3cd33cff0e72050c2da60229bdc2a578",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "4aa834e4-810e-41b2-b18c-818b7f781e7f",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "166415b1dd686127ea51e8025231fcaf",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "ec00389a-2a1d-4c32-ac14-9ca2cf8f9a1d",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "ca780839716a8987305c3048b5ceab31",
            ),
            "resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "851bedd8-55df-4cfa-9138-b9d654252744",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "c675085c3aa4c2b18647e38e1a7a62a8",
            ),
            "resnext29-32x4d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "142b337e-bf32-4092-94e4-24b805486a22",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "de9e81f3bd996d6a5d3037ca2490c042",
            ),
            "resnext29-32x4d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "d0e2599c-777e-4f71-9c3c-f36cfc86fef5",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "d985b4bc553f382d859ae9da284a186d",
            ),
            "resnext29-32x4d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "ede26687-185f-4be0-91c5-73a624ecbd82",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "969e861e3c6d16f238d959c6db0d6b9b",
            ),
            "resnext29-32x4d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "5582520e-6cf1-4322-890c-fdd9de3e8f9c",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "d13b49c992c70c69aca5fd5d0dbc286b",
            ),
            "resnext29-32x4d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "87fc6915-b317-4de8-834e-f88bd1e33f2a",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "ae644c05bf62a9eb2ee88e5d4081dcca",
            ),
            "resnext29-32x4d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "932e3146-e427-4f94-a9fc-34a2f211e094",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-32x4d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "2ca86c99bda60fbe88fda2346d9c9d52",
            ),
            "resnext29-4x64d_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "433541c5-224c-47ee-b547-76f8ee62191a",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "3e522486360df519c1788abb5ed8b01f",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "9cdd0ea0-f80e-43b0-89ba-715b0510aa05",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "50eafebee13869ad77b17c4480651562",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "ea2cba02-d74e-4cb7-98ca-4b0933c2ae89",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "77081206befe46f867794fb30bc4842a",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "81d37d5b-f544-4b22-9255-c99d7d5232de",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "4e96a271c3e08e56a532d4f9a34e9c0f",
            ),
            "resnext29-4x64d_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "9cfba97d-8702-40ec-8367-52f399cdce37",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "c3a44e2ae12ec45bc721d0c5ab8ed013",
            ),
            "resnext29-4x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "4cf59d99-3ee0-4a81-bba9-b5ff4daaa696",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "acb8f7142d65a0ac7d9ba59c47bea84e",
            ),
            "resnext29-4x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "6ef3772e-2387-420d-bc5e-0fd99c1d655d",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "06728be9bade6c3132027ac458b44a9d",
            ),
            "resnext29-4x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "03666192-c2df-402c-b88b-a690516e0d57",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "a1b4eac538793b2a66e5927730737863",
            ),
            "resnext29-4x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "6842596f-c1ce-4cd9-ab09-632e653c1fed",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "4887929022275c5af0762bd7dbdacb68",
            ),
            "resnext29-4x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "80402040-cc68-48c4-9303-cb80a6d1b279",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "6de0a5ce015fe79e51a1db24f3787c54",
            ),
            "resnext29-4x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "ec4cfbbb-af1f-42ae-9587-8ed38ab32a34",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-4x64d_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "f0d82d625f3fa4643c88f2ca84acaf89",
            ),
            "resnext29-8x64d_lse_lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "0a3e6079-f3cb-40bc-adbd-511baf90ef38",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_lse_lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "fcd23ba7f83934148ab4c2917038892d",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "aecd9965-76b7-4472-8fc1-b6a355e02bed",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs512_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "756b64321e95458bf6a3575ce4ca4a5f",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "5a5974cf-4acd-48ed-ba04-f1afb152a098",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs512_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "cb54c88f31dc013b5162d090224792c3",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "5d3a9ee0-f3fb-42bf-8926-e7a7be320741",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "1a89e68a9ac804136f1eef3ef2e4266f",
            ),
            "resnext29-8x64d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "257e523c-5fef-4ea3-8b1c-bb6e509f6a39",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_pgd_ce_seed0_bs512_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "d93d59cfe96004956c9bc62bb6d79575",
            ),
            "resnext29-8x64d_trades-lse_trades-lse_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "d6254afd-d5f0-484c-bbe3-4ab07157309c",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_trades-lse_trades-lse_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "2ef94af39432a87cb9db925ede7017f2",
            ),
            "resnext29-8x64d_trades-lse_trades-lse_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "c0d42da9-4327-4a19-8d74-7a4e94c6afab",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_trades-lse_trades-lse_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "9802a43bea8e200ae6ecba9bd5834d20",
            ),
            "resnext29-8x64d_trades-lse_trades-lse_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "3d6fdfe7-33a5-4f92-9d46-bf022d9815bf",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_trades-lse_trades-lse_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "18b8a6becdf5efee39eaeb2736e375fb",
            ),
            "resnext29-8x64d_trades_trades_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "bafd1cd4-ed87-474b-9235-82eef52ac8fc",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_trades_trades_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "2ca8111e3e5cbe28a009b80b8221b22f",
            ),
            "resnext29-8x64d_trades_trades_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "7a9c1033-1c2d-4b2b-9798-7dedca0f5cc7",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_trades_trades_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "cd60fe1c2205a6dcb19e856b78f28e5b",
            ),
            "resnext29-8x64d_trades_trades_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "893f8131-f5b5-4176-a9a2-d2f1acf126df",
                "https://zenodo.org/record/8285099/files/cifar10_resnext29-8x64d_trades_trades_seed0_bs256_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "ff7ec66d41497256030a284ed1e05482",
            ),
            "senet18_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "f5abc7fc-e5e4-4271-a5ef-d0627008375a",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "941c4bf10b85acb526a22ab145563744",
            ),
            "senet18_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "fd4dc051-4f27-40f3-af0e-81ea8342d8f5",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "1ff5bccb9c39d8874613b0dd3f530b2d",
            ),
            "senet18_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "c793fb18-e8a7-496c-ad23-ff75c5a325ef",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "5ac3c7320804bd2f8229b299a41e855b",
            ),
            "senet18_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "641a8a4b-0e07-4143-aeb7-a68576592374",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "a21750ff73d485e7e7dfac745ab320fc",
            ),
            "senet18_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "0f59736f-2b2f-4051-9d08-81189b6ebb3f",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "d03749b4fa83aa5e0bb624e2450284f3",
            ),
            "senet18_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "262f60e6-ef23-4985-b678-f1d90659f674",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "cd0646a3e09c2bd80d3f453d2acc9c6a",
            ),
            "senet18_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "7d318554-86e0-4f64-9168-fefafb5b2045",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "7e145ed8262bae752017b6cee1f41d7c",
            ),
            "senet18_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "d9f6e89a-0523-42c5-a719-aa307fc88084",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "fcc000698cafaa9d9a79eb8f4b0c70c8",
            ),
            "senet18_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "ec54269f-b55e-43d9-830c-459d566250b4",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "c3c320ae7e40334fa76027b41626c41a",
            ),
            "senet18_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "f0e43f8a-5c32-4e57-a67f-048d594c5101",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "6017fd2c7393562f794c3d7ecb10b49a",
            ),
            "senet18_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "6de32f08-e130-4970-bfb2-c6c3f5b948d6",
                "https://zenodo.org/record/8285099/files/cifar10_senet18_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "aa8e37a03cabfe714e242c84ac4669e6",
            ),
            "vgg11-bn_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "57a47a64-90a5-47c9-ad60-d49f47fbf027",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "2b771188a08fb2842e008f6e371208ad",
            ),
            "vgg11-bn_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "e1f0e012-71ca-4cd7-9bb0-8b2a06ba6a34",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "e400033bb2696f45c718e45e9039bc00",
            ),
            "vgg11-bn_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "474a1c47-bf74-4737-979c-30e8b720b616",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "fdcabfaaedf0875101a5e56889cbfbc4",
            ),
            "vgg11-bn_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "67acb156-bfe4-427c-ae4e-a01421328596",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "ed5d1215b3db34139eff629c0399f3f3",
            ),
            "vgg11-bn_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "34bf3924-49ea-4611-9ac7-92acd70ac8d2",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "47cd8dfbee12cd8fbdd979f8cf8c3a94",
            ),
            "vgg11-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "6202ad1d-59c1-43ca-b976-feee9d700485",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "eed1e65d5b64c7ea2fe8d9bb5c695d54",
            ),
            "vgg11-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "b51cea02-5515-4a33-90a8-72eea534114a",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "cec0b3620c2f6d312366267b428e893f",
            ),
            "vgg11-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "3e81c5c9-d491-4076-9a15-acdf4d91f418",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "c999a06fd0e0ec00441be63d8f4cf3f3",
            ),
            "vgg11-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "ad2ea1ed-157e-40ed-b203-284d06cf30fa",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "59474638969c7e53391ae9cbf9fb3fcc",
            ),
            "vgg11-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "da923c19-4514-403f-aecb-47bf53e7f0fc",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "fdf22d53895b944ddb2117116402e831",
            ),
            "vgg11-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "ccc4af17-a94b-44d4-bc01-888927055458",
                "https://zenodo.org/record/8285099/files/cifar10_vgg11-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "369985d6ebfebdf6505967b3abd38a34",
            ),
            "vgg13-bn_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "babfda86-0c3e-4269-bdf5-6dcde9ec0eee",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "6648c019ffefa9d5ec9ea4abdc3d5a99",
            ),
            "vgg13-bn_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "dab0738b-742a-414c-8e94-8b46b0ad977c",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "e6b64da83760104016d5becbee32a945",
            ),
            "vgg13-bn_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "02d2eb95-df04-44b6-b58f-ab0be5b6ec15",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "a13a56ac285f9c11b42c9c2d351c47fc",
            ),
            "vgg13-bn_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "66c8cde8-7673-4a1c-a8e5-b7d46e66bd05",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "ce96987bf073dd49078ea240ad96aa39",
            ),
            "vgg13-bn_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "667b3570-e25a-417e-86ca-f7fdf60a338f",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "324dd7835c8016b818554182727e6fc4",
            ),
            "vgg13-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "265f8b5d-f5c3-475a-af0f-bb4b31963530",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "3565974bc23e28376ff54c6eeb4722fb",
            ),
            "vgg13-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "9c70516c-5b70-4680-9bf9-bd4c20d2aebb",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "74952ace3a14ca066aba0ce99a10caaf",
            ),
            "vgg13-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "baa0a87c-6b8e-48f5-b69e-4ad1284f71d3",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "12fa2684a54986fc18feeb9df97dc701",
            ),
            "vgg13-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "8518d61d-16bb-4a48-931e-b4aa21c84219",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "47fe9f31e4b489037a450f7b4bbcdc8c",
            ),
            "vgg13-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "89fea259-f778-4a5f-a907-3397fb826dc2",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "c8a8dcb73a1073c3f69fb924901476c5",
            ),
            "vgg13-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "32675550-911c-425b-a4b5-cef9b531b83f",
                "https://zenodo.org/record/8285099/files/cifar10_vgg13-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "b68d0f77faf5d638cf41f4e61b71d73c",
            ),
            "vgg16-bn_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "e6c2f303-f676-4a54-b4e1-74a154bc6a31",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "180407cee45a1f3aaf7d804aa208db4c",
            ),
            "vgg16-bn_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "4750ce60-4b7b-437d-9fc5-93e8d3591c95",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "6061b51a7c1f6edd4f61ae18b1a8a403",
            ),
            "vgg16-bn_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "73443800-dd9a-45d8-bdb1-284d1a1940d0",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "6fcfe8b81422f4aa860f7ddaa7d46bb4",
            ),
            "vgg16-bn_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "6733c34f-7c84-40df-bafc-ca95e9d8cfb8",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "c121fe7db674c83347a490fb1ea889e2",
            ),
            "vgg16-bn_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "22dab17b-d9bf-434f-b1e1-2393ef934f68",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "948a35a2ea36247ad720b0f0b5ae09a2",
            ),
            "vgg16-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "0259a14e-5593-4e16-a9b8-143ab8bc6c73",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "941dafd96e3a45cbc3fdb69ff04d7c7e",
            ),
            "vgg16-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "22278dba-b2ed-49d8-8882-87b1a722870c",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "7dad0f645d0b88b32917e6f21572378c",
            ),
            "vgg16-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "734787be-d3c8-4129-8df6-cf764ea16bf5",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "fb33b51a6c77e3517be6fc76c1ccbc07",
            ),
            "vgg16-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "c84018f3-bfa3-4753-829f-f693c9067d89",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "078c4cc12901630ff1580008bd848fab",
            ),
            "vgg16-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "5161e62d-2fcd-44ca-998b-b1cf15818040",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "fe0b08208895e9cc299926ead440cea1",
            ),
            "vgg16-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "67763380-0faf-4530-95f8-d5c19beab566",
                "https://zenodo.org/record/8285099/files/cifar10_vgg16-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "d004c265514ad3ebef98acb203a3054a",
            ),
            "vgg19-bn_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "adf6e0a1-6c13-4b5a-9a8c-8fb4aba03f69",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_lse_lse_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5_orig.pt",
                "fa25d91ef0859d9652184b685e9b32fa",
            ),
            "vgg19-bn_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "759be7e2-08c5-4895-8b15-3a7b7c0f488e",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_pgd_ce_seed0_bs1024_lr0.01_wd0.0001_sgd_50ep_eps128_orig.pt",
                "051a86386b3844e6e32f093ffc99c114",
            ),
            "vgg19-bn_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "c7f0c205-f3af-45e1-8600-9bb1dfb78722",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig.pt",
                "771820aba12df8e5ccfbc5cacf6a3a8e",
            ),
            "vgg19-bn_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps0.5.pt": (
                "8285099",
                "8055634c-28bb-4020-a3b1-e8a5304163ef",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_pgd_ce_seed0_bs1024_lr0.1_wd0.0001_sgd_50ep_eps128_orig.pt",
                "b6705c615c479a99d5b3383759253974",
            ),
            "vgg19-bn_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps0.5.pt": (
                "8285099",
                "045c16c6-e9f1-4217-8a8c-34b53db2e840",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_pgd_ce_seed0_bs1024_lr0.1_wd1e-05_sgd_50ep_eps128_orig.pt",
                "17f8f1984ede19007da13a6f5a8dfea6",
            ),
            "vgg19-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "fe0d5724-7d15-4c92-b504-fd2a1a47773d",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "506b8cf90172f125833fd24c41a34ed7",
            ),
            "vgg19-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "59f5bf5c-0955-4d13-8afd-110cddf889e5",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "7bfcd152d1a4df25e3cb5cacb2f94210",
            ),
            "vgg19-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "98987bd0-3461-47a1-bd2c-8ff666c14179",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_trades-lse_trades-lse_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "7a910c2ab865f8861358aa59847203f2",
            ),
            "vgg19-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1.pt": (
                "8285099",
                "12a70a77-21c7-4207-977c-a21be339b21c",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1_orig.pt",
                "6ad83e8eae3704df403f98f5b83c51eb",
            ),
            "vgg19-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3.pt": (
                "8285099",
                "4fb8ccc0-3a69-4e5d-93b7-48c7ddf24562",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.3_orig.pt",
                "c0d3527af63cbe62d3297523b1bde90f",
            ),
            "vgg19-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0.pt": (
                "8285099",
                "3cfbc0a5-0ee4-4469-ab92-3bb5cdf1822d",
                "https://zenodo.org/record/8285099/files/cifar10_vgg19-bn_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta1.0_orig.pt",
                "81b74089643e58653d0177cd6657c795",
            ),
        },
    },
}


def build_custom_model(
    model_name: str,
    model_dir: Union[str, Path] = "./models",
    dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
    threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
) -> nn.Module:
    """Build a custom model.

    Model is downloaded from Google Drive using gdown package.

    Args:
        model_name: Model name. Example: custom_resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps0.5
        model_dir: The base directory where the models are saved. Defaults to "./models".
        dataset: The dataset on which the model is trained. Defaults to BenchmarkDataset.cifar_10.
        threat_model: The threat model for which the model is trained. Defaults to ThreatModel.Linf.

    Returns:
        Custom pre-trained model.
    """
    logger.info("Building a custom model...")
    if not isinstance(dataset, BenchmarkDataset):
        if dataset == "cifar10":
            dataset = "cifar_10"
        elif dataset == "cifar100":
            dataset = "cifar_100"
        dataset = BenchmarkDataset[dataset]
    assert isinstance(dataset, BenchmarkDataset)
    if not isinstance(threat_model, ThreatModel):
        threat_model = ThreatModel[threat_model]
    assert isinstance(threat_model, ThreatModel)
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)
    assert isinstance(model_dir, Path)
    model_dir = model_dir / dataset.value / threat_model.value
    model_dir = model_dir.expanduser()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Parse model_name
    # Example: custom_resnext29-32x4d_pgd_ce_seed0_bs1024_lr0.01_wd1e-05_sgd_50ep_eps128_orig
    name_tokens = model_name.split("_")
    assert name_tokens[0] == "custom", "Only custom models are supported."
    arch = name_tokens[1]
    pure_model_name = model_name.replace("custom_", "")

    # Find model weight in model_dir
    filename = f"{pure_model_name}.pt"
    model_path = model_dir / filename

    if filename not in _MODEL_DATA[dataset][threat_model]:
        available_models = list(_MODEL_DATA[dataset][threat_model].keys())
        raise NotImplementedError(
            f"Model {model_name} is not available. For the given dataset "
            f"({dataset.value}) and threat_model ({threat_model.value}), "
            f"available models are: {available_models}."
        )
    metadata = _MODEL_DATA[dataset][threat_model][filename]
    download_url, checksum = metadata[-2:]

    if not model_path.is_file():
        logger.info(
            "Downloading %s from Google Drive to %s", filename, model_path
        )
        logger.info("URL: %s", download_url)
        _download(download_url, model_path)
        logger.info("Download finished.")

    assert model_path.is_file()
    # Checksum
    with model_path.open("rb") as file:
        md5sum = hashlib.md5(file.read()).hexdigest()
    if md5sum != checksum:
        raise RuntimeError("Checksums do not match! Try again!")
    logger.info("Loading model from %s", str(model_path))

    return _build_model(arch, dataset, str(model_path))


def _build_model(arch: str, dataset: str, model_path: str) -> nn.Module:
    """Build one model based on config."""
    # Get dataset-specific params
    metadata = {
        BenchmarkDataset.cifar_10: {
            "num_classes": 10,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        BenchmarkDataset.cifar_100: {
            "num_classes": 100,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        BenchmarkDataset.imagenet: None,
    }[dataset]
    num_classes: int = metadata["num_classes"]
    # Define normalizing params
    normalize_params: _NormParams = {
        "mean": metadata["mean"],
        "std": metadata["std"],
    }

    logger.info("=> Building a classifier...")
    model_fn = {
        BenchmarkDataset.cifar_10: {
            "resnet18": cifar_resnet.ResNet18,
            "resnet34": cifar_resnet.ResNet34,
            "resnet50": cifar_resnet.ResNet50,
            "resnet101": cifar_resnet.ResNet101,
            "resnet152": cifar_resnet.ResNet152,
            "efficientnetb0": cifar_efficientnet.EfficientNetB0,
            "mobilenetv2": cifar_mobilenetv2.MobileNetV2,
            "wideresnet28-10": cifar_wideresnet.wideresnet28_10,
            "wideresnet34-10": cifar_wideresnet.wideresnet34_10,
            "wideresnet34-20": cifar_wideresnet.wideresnet34_20,
            "wideresnet70-16": cifar_wideresnet.wideresnet70_16,
            "dla": cifar_dla.DLA,
            "densenet121": cifar_densenet.densenet121,
            "resnext29-2x64d": cifar_resnext.ResNeXt29_2x64d,
            "resnext29-4x64d": cifar_resnext.ResNeXt29_4x64d,
            "resnext29-8x64d": cifar_resnext.ResNeXt29_8x64d,
            "resnext29-32x4d": cifar_resnext.ResNeXt29_32x4d,
            "inception-v3": cifar_inception.inception_v3,
            "senet18": cifar_senet.SENet18,
            "simplevit": cifar_simplevit.simple_vit,
            "convmixer": cifar_convmixer.build_conv_mixer,
            "googlenet": cifar_googlenet.googlenet,
            "vgg11-bn": cifar_vgg.vgg11_bn,
            "vgg13-bn": cifar_vgg.vgg13_bn,
            "vgg16-bn": cifar_vgg.vgg16_bn,
            "vgg19-bn": cifar_vgg.vgg19_bn,
        }
    }[dataset][arch]
    model = model_fn(num_classes=num_classes)
    model = nn.Sequential(
        common.Normalize(**normalize_params),
        model,
    )
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith("module."):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict, strict=True)
    return model.to("cuda")


def _download(url: str, model_path: Path, chunk_size: int = 1024) -> None:
    """Download a model from url to model_path and check its checksum.

    Args:
        url: URL to download from.
        model_path: Path to save the downloaded model.
        chunk_size: Write chunk size. Defaults to 1024.

    Raises:
        RuntimeError: Download fails.
    """
    response = requests.get(url, stream=True, timeout=1000)
    total = int(response.headers.get("content-length", 0))
    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed. Status code: {response.status_code}.\n"
            f"{response.json()}"
        )

    with model_path.open("wb") as file, tqdm(
        desc=str(model_path),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            pbar.update(size)
