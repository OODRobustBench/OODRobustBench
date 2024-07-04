# OODRobustBench: Adversarial Robustness under Distribution Shift
**Lin Li (KCL), Yifei Wang (MIT), Chawin Sitawarin (UC Berkeley), Michael Spratling (KCL)**

This is the official code of the paper "OODRobustBench: a Benchmark and Large-Scale Analysis of Adversarial Robustness under Distribution Shift". This work has been accepted by the main conference of ICML 2024 and the workshop Data-centric Machine Learning Research (DMLR) of ICLR 2024.

The leaderboard: https://oodrobustbench.github.io/

Paper: https://arxiv.org/abs/2310.12793

## 1. High-level idea and design

Existing works have made great progress in improving adversarial robustness, but typically test their method only on data from the same distribution as the training data, i.e. in-distribution (ID) testing. As a result, it is unclear how such robustness generalizes under input distribution shifts, i.e. out-of-distribution (OOD) testing. This is a concerning omission as such distribution shifts are unavoidable when methods are deployed in the wild. To address this issue we propose a benchmark named OODRobustBench to comprehensively assess OOD adversarial robustness using 23 dataset-wise shifts (i.e. naturalistic shifts in input distribution) and 6 threat-wise shifts (i.e., unforeseen adversarial threat models).

![](assets/benchmark_construction.png)

The code of OODRobustBench is built on top of [RobustBench](https://github.com/RobustBench/robustbench) to allow a unified, RobustBench-like, interface of evaluation and loading models and support loading (latest) models from RobustBench, in other words, you know how to use RobustBench then you know how to use OODRobustBench. Nevertheless, if you have not used RobustBench before, no worry! We have provided a detailed and easy-to-follow guide below for preparation and usage. 

## 2. Preparation

### 2.1. Installation

First of all, **Python 3.8 is strongly recommended** because there is a conflict of dependencies between [robustbench](https://github.com/RobustBench/robustbench) and [perceptual-advex](https://github.com/cassidylaidlaw/perceptual-advex) when a higher version like Python 3.9 is used. 

#### As a repository

clone the project and run the following command to install required packages:

```bash
pip install -r requirements.txt
```

#### As a package

```bash
pip install git+https:github.com:OODRobustBench/OODRobustBench.git
```

This enables importing the package as follows:

```python
from oodrobustbench.eval import benchmark
```

#### Resolve a Python compatibility issue

Unfortunately, the latest version of [robustbench](https://github.com/RobustBench/robustbench) has an [issue](https://github.com/RobustBench/robustbench/blob/master/robustbench/model_zoo/architectures/robustarch_wide_resnet.py) with loading models in Python 3.8. To run the code, follow these steps:

1. Locate the file `robustbench/model_zoo/architectures/robustarch_wide_resnet.py` in your Python environment. This location is provided in the error information when you attempt to import `oodrobustbench`.
2. Open the file in your preferred text editor.
3. Uncomment line 10 to import `List` from `typing`. You should see the original warning here.
4. Replace all instances of `list[]` with `List[]`. This can be quickly done by replacing `list[` with `List[`.

I will monitor updates to robustbench and remove this section once the issue is resolved.

### 2.2. Data

We suggest to put all datasets under the same directory (say $DATA) to ease data management and avoid modifying the source code of data loading. An overview of the file structure is shown below

```bash
$DATA/
|–– cifar-10-batches-py/
|–– cifar-10.1/
|–– cifar-10.2/
|–– cifar-10-r/
|–– CIFAR-10-C/
|–– CINIC-10/
|–– imagenet/
|–– imagenetv2-matched-frequency-format-val/
|–– imagenet-a/
|–– imagenet-r/
|–– objectnet/
|–– ImageNet-C/
|–– ImageNet-V/
|–– ImageNet-Sketch/
```

The above datasets are divided into two groups:

1. Automatic download: CIFAR10, CIFAR10-C
2. Manual download: [CIFAR10.1](https://github.com/modestyachts/CIFAR-10.1/tree/master/datasets), [CIFAR10.2](https://github.com/modestyachts/cifar-10.2), [CIFAR10-R](https://github.com/TreeLLi/cifar10-r), [CINIC-10](https://datashare.ed.ac.uk/handle/10283/3192), [ImageNet](https://image-net.org/download.php), [ImageNet-v2](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main), [ImageNet-A](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar), [ImageNet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar), [ImageNet-C](https://github.com/hendrycks/robustness#imagenet-c), [ObjectNet](https://www.kaggle.com/datasets/treelinli/objectnet-imagenet-overlap), [ImageNet-V](https://github.com/Heathcliff-saku/ViewFool_/tree/master), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

Datasets in Group 1 will be downloaded automatically when used. Datasets in Group 2 need to be downloaded from the given links by clicking the dataset name and structured as specified above. A guide on how to prepare ImageNet can be found [here](https://github.com/soumith/imagenet-multiGPU.torch#data-processing).

Note that the folder names should be followed strictly unless modifying our original source code. We suggest to use [soft link](https://en.wikipedia.org/wiki/Symbolic_link) to reuse the datasets that you have already had before by linking them to $DATA. 

## 3. Model Zoo

Our model zoo consists of three groups: 

1. *RobustBench models* are loaded by [robustbench](https://github.com/RobustBench/robustbench) API
2. *Public collected models* are manually collected from public adversarial ML works (not covered by RobustBench by the release time of this work). 
3. *Custom models* were trained by ourselves, see [here](#5.-Custom-Models) for details.

Noticeably, thanks to the deep integration with RobustBench, **OODRobustBench can seamlessly load the latest submissions to RobustBench** by simply upgrading the RobustBench package in the dev environment.

### 3.1. Automatic model loading

To load a model, we provide a unified API similar to RobustBench:

```python
from oodrobustbench.utils import load_model
# an example of loading a model named Wong2020Fast for CIFAR10 Linf
model = load_model(model_name='Wong2020Fast', 
                   model_dir='/root/to/models',
                   dataset='cifar10',
                   threat_model='Linf')
```

Replace `/root/to/models` with the real directory for `model_dir`, by default, `models` under the working directory will be used. 

The weights of models of RobustBench and *custom models* will be automatically downloaded and placed in the specified directory if needed, while the weights of *public collected* models currently need to be manually downloaded from this sharedpoint (TODO) and placed in the appropriate directory. 

Regarding model names, please refer to [robustbench](https://github.com/RobustBench/robustbench) for RobustBench models, [this section](#5.-Custom-Models) for custom models, and [this source code file](https://github.com/OODRobustBench/OODRobustBench/blob/main/oodrobustbench/models/__init__.py) for public collected models.

### 3.2. Contribution: submit your model to the leaderboard and / or model zoo 

Thanks for your interest! To submit your model to the leaderboard, you need to first enable the automatic loading of your model as following:

1. (Optional) add your custom model architecture file under `/oodrobustbench/models`
3. Modify `/oodrobustbench/models/__init__.py` to add a callable constructor of your model arch
4. Modify `load_model()` in `/oodrobustbench/utils.py` to load trained weights to your model

Please refer to [this doc](https://github.com/RobustBench/robustbench/tree/master?tab=readme-ov-file#model-definition) for some guidance, otherwise, contact the author if further clarification required. 

After successfully adding your model, say it can be automatically loaded by the code, you need to email the author (`linli.tree@outlook.com`) with the above modified files and your model weights for a submission to the leaderboard. 

## 4. Evaluation

We describe below the template commands we used to get the results reported in our paper. The output results are saved under the directory `model_info/$DATASET/$THREAT_MODEL`. The program automatically saves the result of each shift evaluation and load the results from the saved file if have so no need to worry about the evaluation being interrupted.

### 4.1. Dataset shift

For CIFAR10 $\ell_\infty$ models under all corruptions and natural shifts with 10k samples:

```bash
python -m oodrobustbench.eval --data_dir $DATA --threat-model Linf --adv-norm Linf -a mm5 --corruption-models corruptions --natural-shifts all -n 10000 --model_name $MODEL_NAME
```

Please refer to the code of `oodrobustbench/eval.py` or running the command `python -m oodrobustbench.eval -h` for the explanation and the candidate values of each argument. 

For CIFAR10 $\ell_2$ models:

```bash
python -m oodrobustbench.eval --data_dir $DATA --threat-model L2 --adv-norm L2 -a mm5 --corruption-models corruptions --natural-shifts all -n 10000 --model_name $MODEL_NAME
```

For ImageNet $\ell_\infty$ models:

```bash
python -m oodrobustbench.eval --data_dir $DATA --dataset imagenet --threat-model Linf --adv-norm Linf -a mm5 --eps 0.01568627 --corruption-models corruptions --natural-shifts all -n 5000 --model_name $MODEL_NAME
```

### 4.2 Threat shift

For CIFAR10 $\ell_\infty$ models against LPA threat shift:

```bash
python -m oodrobustbench.eval --data_dir $DATA --threat-model Linf -a lpa --eps 0.5 -n 10000 --model_name $MODEL_NAME
```

For CIFAR10 $\ell_\infty$ models against PPGD threat shift:

```bash
python -m oodrobustbench.eval --data_dir $DATA --threat-model Linf -a ppgd --eps 0.5 -n 10000 --model_name $MODEL_NAME
```

For CIFAR10 $\ell_\infty$ models against ReColor threat shift:

```bash
python -m oodrobustbench.eval --data_dir $DATA --threat-model Linf -a stadv --eps 0.05 -n 10000 --model_name $MODEL_NAME
```

For CIFAR10 $\ell_\infty$ models against StAdv threat shift:

```bash
python -m oodrobustbench.eval --data_dir $DATA --threat-model Linf -a recolor --eps 0.06 -n 10000 --model_name $MODEL_NAME
```

For CIFAR10 $\ell_\infty$ models against different $p$-norm threat shift:

```bash
python -m oodrobustbench.eval --data_dir $DATA --threat-model Linf --adv-norm L2 -a mm5 --eps 0.5 -n 10000 --model_name $MODEL_NAME
```

For CIFAR10 $\ell_\infty$ models against different $\epsilon$ threat shift:

```bash
python -m oodrobustbench.eval --data_dir $DATA --threat-model Linf --adv-norm Linf -a mm5 --eps 0.0470588 -n 10000 --model_name $MODEL_NAME
```

Please refer to our paper for the configuration of threat shifts for the settings other than CIFAR10 $\ell_\infty$.

### 4.3 Evaluate your own model

To evaluate your own models, you can use `benchmark()`  as exemplified below:

```python
from oodrobustbench.eval import benchmark
model = initialize your own model
model.eval()
id_acc, id_rob, ood_acc_robs = benchmark(model,
                                         n_examples=10000,
                                         dataset='cifar10',
                                         attack='mm5',
                                         threat_model='Linf',
                                         adv_norm='Linf',
                                         natural_shifts='all',
                                         corruption_models='corruptions',
                                         corruptions=None,
                                         severities=[1,2,3,4,5],
                                         to_disk=True,
                                         model_name=$MODEL_NAME,
                                         data_dir=$DATA,
                                         device='cuda',
                                         batch_size=100,
                                         eps=8/255)
```

This is actually what happens when you run `python -m oodrobustbench.eval`. Note that the results will be also saved on the disk when calling `benchmark()` to evaluate.

## 5. Custom Models

The following instructions explain how to load custom models that we have trained ourselves. Model name is a bit long and starts with `custom_`. It contains the hyperparameter choices. For example, `custom_convmixer_trades_trades_seed0_bs512_lr0.1_wd0.0001_sgd_50ep_eps0.5_beta0.1`,

The weights are hosted on Zenodo and is downloaded automatically when a model is called. Download speed from Zenodo server can be poor sometimes so if you know you want to use all the models, you can download all the weights at once with `zenodo_get` and put them at the right location:

```bash
pip install zenodo_get
cd $MODEL_PATH  # mkdir if needed
zenodo_get $DEPOSIT_ID
```

where `$DEPOSIT_ID` and `$MODEL_PATH` are the Zenodo deposit ID and the associated model path. Each deposit has a maximum size of 40 GB and contains a group of models denoted by the path. See the list of `$DEPOSIT_ID: $MODEL_PATH` below:

* `8285099`: `cifar10/L2`.

Please install the extra packages in `requirements.txt`. See `oodar.models.custom_models.utils._MODEL_DATA` for the list of available models.

## 6. Citation

```
@inproceedings{li2024oodrobustbench,
    title={OODRobustBench: a Benchmark and Large-Scale Analysis of Adversarial Robustness under Distribution Shift},
    author={Lin Li, Yifei Wang, Chawin Sitawarin, Michael Spratling},
    booktitle={International Conference on Machine Learning},
    year={2024}
}
```
