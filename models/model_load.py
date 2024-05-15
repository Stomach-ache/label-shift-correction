import torchvision.models as models
import torch
from collections import OrderedDict
import os
import os
from os import path as osp
import sys
from contextlib import contextmanager

from transformers import ViTForImageClassification

this_dir = osp.dirname(__file__)


def add_pythonpath(path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            sys.path.insert(0, path)
            original_pythonpath = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = f"{path}:{original_pythonpath}"
            try:
                result = func(*args, **kwargs)
            finally:
                os.environ["PYTHONPATH"] = original_pythonpath
                sys.path.pop(0)
            return result

        return wrapper

    return decorator


@contextmanager
def add_path_context(path):
    original_pythonpath = os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, path)
    os.environ["PYTHONPATH"] = f"{path}:{original_pythonpath}"
    try:
        yield
    finally:
        os.environ["PYTHONPATH"] = original_pythonpath
        sys.path.pop(0)


def remove_item_in_key(state_dict, item="module"):
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(item):
            k = k[(len(item) + 1):]
        new_dict[k] = v
    return new_dict


def load_resx50_org():
    from torchvision.models import resnext50_32x4d as x50
    model = x50(weights='DEFAULT')
    return model


def load_paco_res50_imagenetlt(model_path='pre_trained/gpaco_r50_imagenetlt.pth.tar'):
    from .paco_resnet_imagenet import MoCo
    state = torch.load(model_path)['state_dict']
    new_dict = remove_item_in_key(state)

    model = MoCo(# getattr(resnet_imagenet, args.arch),
        models.__dict__['resnet50'], 128, 8192, 0.999, 0.2, True)
    model.load_state_dict(state_dict=new_dict)
    return model


def load_paco_res32_cifar100(model_path='pre_trained/gpaco_x50_imagenetlt.pth.tar'):
    from .paco_model.moco.builder import MoCo
    from .paco_model.resnet_cifar import resnet32
    state = torch.load(model_path)['state_dict']
    new_dict = remove_item_in_key(state)

    model = MoCo(resnet32, dim=32, K=1024, m=0.999, T=0.05, mlp=True, feat_dim=64, num_classes=100, normalize=False)
    model.load_state_dict(state_dict=new_dict)
    return model


def load_paco_resx50_imagenetlt(model_path='pre_trained/gpaco_x50_imagenetlt.pth.tar'):
    from .paco_resnet_imagenet import MoCo
    state = torch.load(model_path)['state_dict']
    new_dict = remove_item_in_key(state)

    model = MoCo(# getattr(resnet_imagenet, args.arch),
        models.__dict__['resnext50_32x4d'], dim=128, K=8192, m=0.999, T=0.2, mlp=True)
    model.load_state_dict(state_dict=new_dict)
    return model


def load_ride_res32_cifar100(model_path='pre_trained/ride_cifar100/checkpoint-epoch5.pth'):
    from .Ride_model import ride_net_wrapped
    from .Ride_model.parse_config import ConfigParser
    from .Ride_model import utils
    # state = torch.load(model_path)['state_dict']
    # new_dict = remove_item_in_key(state, 'backbone')

    model = ride_net_wrapped('build_ride_res32_cifar100', 'pre_trained/ride_cifar100/checkpoint-epoch5.pth')
    # model.model.load_state_dict(state)
    return model


def load_ride_res50_imagenetlt(model_path='pre_trained/imagenet/RIDE_imagenet_x50/checkpoint-epoch5.pth'):
    from .Ride_model import ride_net_wrapped
    state = torch.load(model_path)['state_dict']
    # new_dict = remove_item_in_key(state, 'backbone')

    model = ride_net_wrapped('build_ride_resx50_imagenet_lt')
    model.model.load_state_dict(state)
    return model


class ViTHug_wrap(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(model_path)

    def forward(self, x, target=None):
        logits = self.model(x, target).logits
        return logits


def load_ft_vit_cifar100(model_path='pre_trained/vit_cifar100'):
    model = ViTHug_wrap(model_path)
    return model


def load_ncl_cifar100(model_path='pre_trained/NCL_cifar100/best_ensemble_model.pth'):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper()
    model.build_ncl_w_con_cifar100()

    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')

    model.model.load_state_dict(state_dict)
    return model


def load_ncl_cifar100_wo_con(model_path='pre_trained/augmix/cifar100_ncl.pth', div_out=False, c_r_dir=None):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_wo_con_cifar100(c_r_dir=c_r_dir)
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model


def load_ncl_imagenet(model_path='pre_trained/augmix/imagenet_ncl_augmix.pth', div_out=False):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_imagenet()
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model


def load_ncl_imagenet_x50(model_path='pre_trained/imagenet/imagenet_ncl_x50.pth', div_out=False):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_imagenet_x50()
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model


def load_ncl_places_wo_con(model_path='pre_trained/places/NCL_places_wo_moco_hcm.pth', div_out=False):
    from .NCL_model.build_model import NCLModelWrapper
    pth_dict = torch.load(model_path)
    model = NCLModelWrapper(div_out=div_out)
    model.build_ncl_places_wo_con()
    # model.model = torch.nn.DataParallel(model.model).cuda()
    # state_dict = pth_dict['state_dict']
    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model


def load_shike_cifar100(model_path='pre_trained/cifar100_im100/shike.pth.tar'):
    from .SHIKE_model.build_model import SHIKEWrapper
    model = SHIKEWrapper()
    model.build_cifar()
    pth_dict = torch.load(model_path)

    state_dict = remove_item_in_key(pth_dict['state_dict'], 'module')
    model.model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    # load_adv_wres34_cifar10()
    # load_paco_res50_imagenetlt()
    # load_ride_res32_cifar100()
    pass
