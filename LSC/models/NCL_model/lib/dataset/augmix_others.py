from .baseset import BaseSet
import random, cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter, Image
from .randaugment import rand_augment_transform
from .augmix.augment_and_mix import aug_mix_torch, aug_mix_cuda, aug_mix_torch224
import os

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def aug_plus(dataset='ImageNet_LT', aug_type='randcls_sim', mode='train', randaug_n=2, randaug_m=10, plus_plus='False'):
    # PaCo's aug: https://github.com/jiequancui/ Parametric-Contrastive-Learning

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if dataset == 'inat' \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if plus_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            # transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_regular = [
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_sim = [
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_sim02 = [
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randnclsstack = [
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_randncls = [
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    val_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    if aug_type == 'regular_regular':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif aug_type == 'mocov2_mocov2':
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif aug_type == 'sim_sim':
        transform_train = [transforms.Compose(augmentation_sim), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randcls_sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randclsstack_sim':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randclsstack_sim02':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim02)]

    if mode == 'train':
        return transform_train
    else:
        return val_transform


class ImageNet_LT_AUGMIX(BaseSet):
    NORM = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def __init__(self, mode='train', cfg=None, sample_id = 0, transform=None):
        super(ImageNet_LT_AUGMIX, self).__init__(mode, cfg, transform)
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()
        self.use_randaug = cfg.DATASET.AUGMIX.randaug
        self.randaug = aug_plus(dataset='ImageNet_LT', aug_type='randcls_sim', mode=mode, plus_plus='False')
        self.severity = self.cfg.DATASET.AUGMIX.aug_severity
        self.all_ops = self.cfg.DATASET.AUGMIX.all_ops
        self.width = self.cfg.DATASET.AUGMIX.width
        self.depth = self.cfg.DATASET.AUGMIX.depth
        self.mix_alpha = self.cfg.DATASET.AUGMIX.alpha
        self.train = mode == 'train'
        self.pre_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0)])
        self.progress = transforms.Compose([transforms.ToTensor(), self.normalize])

    def _get_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = Image.open(fpath).convert('RGB')
        return img


    def __getitem__(self, index):

        now_info = self.data[index]
        meta = dict()
        img_org = self._get_image(now_info)
        img_trans = self.pre_transform(img_org)
        if self.train:
            meta['augmix'] = [
                aug_mix_torch224(img_trans, self.progress, aug_severity=self.severity, mixture_width=self.width,
                                 mixture_depth=self.depth, all_ops=self.all_ops, alpha=self.mix_alpha),
                aug_mix_torch224(img_trans, self.progress, aug_severity=self.severity, mixture_width=self.width,
                                 mixture_depth=self.depth, all_ops=self.all_ops, alpha=self.mix_alpha), ]
            if self.use_randaug:
                img = self.randaug[0](img_org)
            else:
                img = self.progress(img_trans)
        else:
            img = self.randaug(img_org)
        image_label = now_info['category_id']  # 0-index
        return img, image_label, meta