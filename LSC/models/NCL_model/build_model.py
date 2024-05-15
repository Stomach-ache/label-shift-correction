import torch.nn

from . import init_path
from .lib.utils.utils import get_multi_model_final
from .lib.config import cfg, update_config

import os
import yaml
import copy
class Dummy_Arg:
    def __init__(self, cfg=None, auto_resume=None, local_rank=None, model_dir=None, opts=''):
        self.cfg, self.auto_resume, self.local_rank, self.model_dir, self.opts = \
            cfg, auto_resume, local_rank, model_dir, opts

    def parse_args(self, **kwargs):
        return self

class Placeholder:
    def __getattr__(self, name):
        def placeholder(*args, **kwargs):
            pass
        return placeholder


def get_cfg_logger(c_r_dir):
    logger = Placeholder
    current_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(current_path)
    c_a_dir = os.path.join(directory_path, c_r_dir)
    args = Dummy_Arg(cfg=c_a_dir)
    update_config(cfg, args=args)

    return cfg, logger

class NCLModelWrapper(torch.nn.Module):
    def __init__(self, div_out=False):
        super().__init__()
        self.forward_func = None
        self.div_out = div_out
        self.featurizer, self.classifier = None, None

    def build_ncl_w_con_cifar100(self, c_r_dir=None):
        if c_r_dir is None:
            c_r_dir = 'configs/CIFAR/CIFAR100/cifar100_im100_NCL_with_contrastive.yaml'
        cfg, logger = get_cfg_logger(c_r_dir)
        self.model = get_multi_model_final(cfg, num_classes=100, num_class_list=None, device='cuda', logger=logger)
        self.forward = self.forward_w_con
        return self

    def build_ncl_wo_con_cifar100(self, c_r_dir=None):
        if c_r_dir is None:
            c_r_dir = 'configs/CIFAR/CIFAR100/cifar100_im100_NCL_augmix_randaug_wo_hcm.yaml'

        cfg, logger = get_cfg_logger(c_r_dir)
        self.model = get_multi_model_final(cfg, num_classes=100, num_class_list=None, device='cuda', logger=logger)
        self.forward =self.forward_wo_con
        return self

    def build_ncl_imagenet(self):
        c_r_dir = 'configs/ImageNet_LT/ImageNet_LT.yaml'
        cfg, logger = get_cfg_logger(c_r_dir)
        self.model = get_multi_model_final(cfg, num_classes=1000, num_class_list=None, device='cuda', logger=logger)
        self.forward = self.forward_wo_con
        return self

    def build_ncl_imagenet_x50(self):
        c_r_dir = 'configs/ImageNet_LT/ImageNet_LT_x50.yaml'
        cfg, logger = get_cfg_logger(c_r_dir)
        self.model = get_multi_model_final(cfg, num_classes=1000, num_class_list=None, device='cuda', logger=logger)
        self.forward = self.forward_wo_con
        return self

    def build_ncl_places_wo_con(self):
        c_r_dir = 'configs/Places_LT/Places_LT_NCL_wo_moco_hcm.yaml'
        cfg, logger = get_cfg_logger(c_r_dir)
        self.model = get_multi_model_final(cfg, num_classes=365, num_class_list=None, device='cuda', logger=logger)
        self.forward = self.forward_wo_con
        return self

    def build_ncl_inat_wo_con(self):
        c_r_dir = 'configs/iNat18/inat18_NCL_wo_hcm.yaml'
        cfg, logger = get_cfg_logger(c_r_dir)
        self.model = get_multi_model_final(cfg, num_classes=8142, num_class_list=None, device='cuda', logger=logger)
        self.forward = self.forward_wo_con
        return self

    def forward_w_con(self, input, **kwargs):
        if not isinstance(input, (list, tuple)):
            ncl_input = [input, input, input]
        else:
            ncl_input = input

        feature, feature_MA = self.model((ncl_input, ncl_input), feature_flag=True)
        if 'feature_flag' in kwargs:
            return feature
        logits_ce, logits, logits_MA = self.model((feature, feature_MA), classifier_flag=True)
        if self.div_out:
            return logits_ce
        mean_logits = sum(logits_ce) / len(logits_ce)
        return mean_logits


    def forward_wo_con(self, input, **kwargs):
        if not isinstance(input, (list, tuple)):
            ncl_input = [input, input, input]
        else:
            ncl_input = input
        model = self.model
        network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)

        feature = model(ncl_input, feature_flag=True)
        if 'feature_flag' in kwargs:
            return feature
        logits_ce = model(feature, classifier_flag=True)

        if self.div_out:
            return logits_ce
        mean_logits = sum(logits_ce) / len(logits_ce)
        return mean_logits
        # return logits_ce[0]



