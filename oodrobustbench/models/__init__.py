from addict import Dict

from timm.models import create_model
from torchvision import models as pt_models

from robustbench.model_zoo.architectures.utils_architectures import normalize_model
from robustbench.model_zoo.architectures.wide_resnet import WideResNet

from .preact_resnet import PreActResNet
from .vit import vit_base
from .wide_resnet import Wide_ResNet as WideResNet2
# from .utils_architecture import get_new_model

mu = (0.485, 0.456, 0.406)
sigma = (0.229, 0.224, 0.225)

resnet50 = lambda: normalize_model(create_model('resnet50', num_classes=1000),
                                   mu, sigma)
prn18 = lambda: PreActResNet(18, 10)
wrn3410 = lambda: WideResNet(depth=34, widen_factor=10, sub_block1=False)
wrn2810 = lambda: WideResNet(depth=28, widen_factor=10, sub_block1=False)
wrn34102 = lambda: WideResNet2(depth=34, width=10, out_dim=10)
wrn28102 = lambda: WideResNet2(depth=28, width=10, out_dim=10)
vitb = lambda: vit_base((3, 32, 32), 10, 4)

models = Dict()


'''
CIFAR10

'''

models.prn18fgsmnadvlc.model = prn18
models.prn18fgsmnadvlcswa.model = prn18
models.prn18pgdadvlc.model = prn18
models.prn18pgdadvlcswa.model = prn18

models.prn18nfgsmaroid.model = prn18

models.prn18scoreaa.model = prn18
models.prn18scorecutout.model = prn18
models.prn18scoreta.model = prn18
models.prn18scorearoid.model = prn18
models.prn18scorercrop.model = prn18
models.prn18scoreidbh.model = prn18

models.prn18tradesaa.model = prn18
models.prn18tradescutout.model = prn18
models.prn18tradesta.model = prn18
models.prn18tradesaroid.model = prn18
models.prn18tradesrcrop.model = prn18
models.prn18tradesidbh.model = prn18

models.wrn3410pgdaroid.model = wrn34102
models.wrn3410pgdswaaroid.model = wrn34102
models.wrn3410iseataroid.model = wrn34102
models.wrn3410awparoid.model = wrn34102
models.wrn3410awpswaaroid.model = wrn34102
models.wrn3410pgdaa.model = wrn34102
models.wrn3410pgdta.model = wrn34102
models.wrn3410pgdcutout.model = wrn34102
models.wrn3410pgdidbh.model = wrn34102
models.wrn3410pgdcutmix.model = wrn34102
models.wrn3410pgdrcrop.model = wrn34102
models.wrn3410pgdswaaa.model = wrn34102
models.wrn3410pgdswata.model = wrn34102
models.wrn3410pgdswacutout.model = wrn34102
models.wrn3410pgdswaidbh.model = wrn34102
models.wrn3410pgdswacutmix.model = wrn34102
models.wrn3410pgdswarcrop.model = wrn34102

models.vitbpgdaa.model = vitb
models.vitbpgdcutout.model = vitb
models.vitbpgdta.model = vitb
models.vitbpgdaroid.model = vitb
models.vitbpgdcutmix.model = vitb
models.vitbpgdrcrop.model = vitb
models.vitbpgdidbh.model = vitb

models.wrn3410iseatrcrop.model = wrn34102
models.wrn3410iseatswarcrop.model = wrn34102
models.wrn3410iseatidbh.model = wrn34102
models.wrn2810iseatidbh.model = wrn28102
models.wrn2810iseatextra.model = wrn28102

'''
ImageNet

'''

models.rn50nfgsmbest.model = resnet50
models.rn50nfgsmbest.prepr = 'Res256Crop224'
models.rn50nfgsmend.model = resnet50
models.rn50nfgsmend.prepr = 'Res256Crop224'

models.rn50trades5best.model = resnet50
models.rn50trades5best.prepr = 'Res256Crop224'
models.rn50trades5end.model = resnet50
models.rn50trades5end.prepr = 'Res256Crop224'

models.rn50trades6best.model = resnet50
models.rn50trades6best.prepr = 'Res256Crop224'
models.rn50trades6end.model = resnet50
models.rn50trades6end.prepr = 'Res256Crop224'

models.rn50trades7best.model = resnet50
models.rn50trades7best.prepr = 'Res256Crop224'
models.rn50trades7end.model = resnet50
models.rn50trades7end.prepr = 'Res256Crop224'

models.advtrain_vgg16_ep4.model = lambda: normalize_model(create_model('vgg16', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_vgg16_ep4.prepr = 'Res256Crop224'

models.advtrain_resnest50d_ep4.model = lambda: normalize_model(create_model('resnest50d', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_resnest50d_ep4.prepr = 'Res256Crop224'

models.advtrain_resnet101_ep4.model = lambda: normalize_model(create_model('resnet101', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_resnet101_ep4.prepr = 'Res256Crop224'

models.advtrain_swin_small_patch4_window7_224_ep4.model = lambda: normalize_model(create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_swin_small_patch4_window7_224_ep4.prepr = 'Res256Crop224'

models.advtrain_swin_base_patch4_window7_224_ep4.model = lambda: normalize_model(create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_swin_base_patch4_window7_224_ep4.prepr = 'Res256Crop224'

models.advtrain_densenet121_ep4.model = lambda: normalize_model(create_model('densenet121', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_densenet121_ep4.prepr = 'Res256Crop224'

models.advtrain_seresnet50_ep4.model = lambda: normalize_model(create_model('seresnet50', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_seresnet50_ep4.prepr = 'Res256Crop224'

models.advtrain_efficientnet_b0_ep4.model = lambda: normalize_model(create_model('efficientnet_b0', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_efficientnet_b0_ep4.prepr = 'Res256Crop224'

models.advtrain_efficientnet_b1_ep4.model = lambda: normalize_model(create_model('efficientnet_b1', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_efficientnet_b1_ep4.prepr = 'Res256Crop224'

models.advtrain_efficientnet_b2_ep4.model = lambda: normalize_model(create_model('efficientnet_b2', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_efficientnet_b2_ep4.prepr = 'Res256Crop224'

models.advtrain_efficientnet_b3_ep4.model = lambda: normalize_model(create_model('efficientnet_b3', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_efficientnet_b3_ep4.prepr = 'Res256Crop224'

models.advtrain_resnext50_32x4d_ep4.model = lambda: normalize_model(create_model('resnext50_32x4d', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_resnext50_32x4d_ep4.prepr = 'Res256Crop224'

models.advtrain_seresnet10_ep4.model = lambda: normalize_model(create_model('seresnet10', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_seresnet10_ep4.prepr = 'Res256Crop224'

models.advtrain_vit_base_patch32_224_ep4.model = lambda: normalize_model(create_model('vit_base_patch32_224', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_vit_base_patch32_224_ep4.prepr = 'Res256Crop224'

models.advtrain_vit_small_patch16_224_ep4.model = lambda: normalize_model(create_model('vit_small_patch16_224', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_vit_small_patch16_224_ep4.prepr = 'Res256Crop224'

models.advtrain_vit_base_patch16_224_ep4.model = lambda: normalize_model(create_model('vit_base_patch16_224', pretrained=False, num_classes=1000),mu, sigma)
models.advtrain_vit_base_patch16_224_ep4.prepr = 'Res256Crop224'

models.rn50free.model = resnet50
models.rn50free.prepr = 'Res256Crop224'

# models.revisiting_convnext_base.model = lambda: normalize_model(create_model('convnext_base', pretrained=False, num_classes=1000),mu, sigma)
models.revisiting_convnext_base.model = lambda: create_model('convnext_base', pretrained=False, num_classes=1000)
models.revisiting_convnext_base.prepr = 'Res256Crop224'

models.revisiting_convnext_t.model = lambda: normalize_model(create_model('convnext_tiny'), mu, sigma)
models.revisiting_convnext_t.prepr = 'Res256Crop224'

models.revisiting_vit_s.model = lambda: normalize_model(create_model('vit_small_patch16_224'),mu, sigma)
models.revisiting_vit_s.prepr = 'Res256Crop224'

models.revisiting_vit_b.model = lambda: create_model('vit_base_patch16_224')
models.revisiting_vit_b.prepr = 'Res256Crop224'

# models.revisiting_convnext_iso_cvst.model = lambda: normalize_model(get_new_model('convnext_iso', not_original=True), mu, sigma)
# models.revisiting_convnext_iso_cvst.prepr = 'Res256Crop224'

# models.revisiting_vit_m_cvst.model = lambda: get_new_model('vit_m', not_original=True)
# models.revisiting_vit_m_cvst.prepr = 'Res256Crop224'


'''
NonLp

'''

from perceptual_advex.utilities import get_dataset_model

models.pat_alexnet.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]
models.pat_self.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]
models.pat_recolor.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]
models.pat_stadv.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]
models.pat_average.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]
models.pat_max.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]
models.pat_random.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]

from torch import nn
from .vr.resnet_cifar import resnet18, resnet50
from .vr.wide_resnet import wrn_28_10

mu, sigma = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)

models.vr_linf_resnet18.model = lambda: normalize_model(resnet18(num_classes=10, activation=nn.ReLU()), mu, sigma)
models.vr_linf_wrn2810.model = lambda: normalize_model(wrn_28_10(num_classes=10, activation=nn.ReLU()), mu, sigma)
models.vr_l2_resnet18.model = lambda: normalize_model(resnet18(num_classes=10, activation=nn.ReLU()), mu, sigma)
models.vr_pat0500.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]
models.vr_pat0501.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]
models.vr_pat1001.model = lambda: get_dataset_model(dataset='cifar', dataset_path='data', arch='resnet50')[1]

# SSL
from .dynacl.resnet import resnet18

models.acl.model = lambda: resnet18(num_classes=10)

# composite adv
from .composite.resnet import ResNet50
from .composite.wideresnet import WideResNet
mu, sigma = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

models.gatfresnet50.model = lambda: ResNet50()
models.gatfsresnet50.model = lambda: ResNet50()

models.gatfwrn3410.model = lambda: WideResNet()
models.gatfswrn3410.model = lambda: WideResNet()

# REx
from .rex import ResNet18

models.msd_rex_105.model = lambda: ResNet18()
models.msd_rex_99.model = lambda: ResNet18()

