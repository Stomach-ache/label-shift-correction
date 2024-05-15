import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset

NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
NORM_VIT = ((.5, .5, .5), (.5, .5, .5))
SIZE_VIT = 224
vit_te_transforms = transforms.Compose([transforms.Resize(SIZE_VIT),
                                        transforms.CenterCrop(SIZE_VIT),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*NORM_VIT)])

mnist_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

distrb_dict = {
    'uniform': (1, False),
    'forward50': (0.02, False),
    'forward25': (0.04, False),
    'forward10': (0.1, False),
    'forward5': (0.2, False),
    'forward2': (0.5, False),
    'backward50': (0.02, True),
    'backward25': (0.04, True),
    'backward10': (0.1, True),
    'backward5': (0.2, True),
    'backward2': (0.5, True),
}

def wrapper_cifar10_c(args):
    corruption, level, dataroot, batch_size, workers = args.corruption, args.level, args.dataroot, args.test_batch_size, args.workers
    # reverse,  imb_factor = args.reverse, args.imb_factor  # label imbalance cls_num, imb_type, imb_factor, reverse
    distrb = args.distrb
    if distrb not in distrb_dict:
        raise NotImplementedError(f"{distrb} not implemented")
    else:
        imb_factor, reverse = distrb_dict[distrb]
    if '100' in args.dataset.lower():
        if 'vit' in args.model.lower():
            test_set, test_loader = get_cifar100_c_testset_and_loader(corruption, level, dataroot, batch_size, workers, transform=vit_te_transforms)
        else:
            test_set, test_loader = get_cifar100_c_testset_and_loader(corruption, level, dataroot, batch_size, workers, reverse=reverse, imb_factor=imb_factor)
    else:
        if 'vit' in args.model.lower():
            test_set, test_loader = get_cifar10_c_testset_and_loader(corruption, level, dataroot, batch_size, workers, transform=vit_te_transforms)
        else:
            test_set, test_loader = get_cifar10_c_testset_and_loader(corruption, level, dataroot, batch_size, workers)

    return test_set, test_loader


# TODO implement imbalance
def get_cifar10_c_testset_and_loader(corruption='original', level=5, dataroot='../data', batch_size=256, workers=8, transform=te_transforms):
    tesize = 10000
    teset = cifar10_c_testset(dataroot=dataroot, transform=transform)
    if corruption in common_corruptions:
        print('Test on %s level %d' % (corruption, level))
        teset_raw = np.load(dataroot + '/CIFAR-10-C/%s.npy' % (corruption))
        teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
        teset.samples = teset_raw
    else:
        print("Use Common original Dataset")

    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return teset, teloader


def get_cifar100_c_testset_and_loader(corruption=None, level=5, dataroot='../data', batch_size=256, workers=8, transform=te_transforms, reverse=False, imb_factor=1):
    teset = cifar100_c_testset(dataroot=dataroot, transform=transform, reverse=reverse, imb_factor=imb_factor)
    tesize = len(teset.targets)  # 10000
    if corruption in common_corruptions:
        print('Test on %s level %d' % (corruption, level))
        teset_raw = np.load(dataroot + '/CIFAR-100-C/%s.npy' % (corruption))
        teset_raw = teset_raw[(level - 1) * tesize: level * tesize]
        teset.samples = teset_raw
    else:
        print("Use Common original Dataset")

    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return teset, teloader


class cifar100_c_testset(torchvision.datasets.CIFAR100):
    def __init__(self, dataroot, transform, original=True, rotation=True, rotation_transform=None, reverse=False, imb_factor=1):
        super(cifar100_c_testset, self).__init__(root=dataroot, train=False, download=True, transform=transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform
        self.samples = self.data
        self.original_samples = self.samples
        self.reverse = reverse
        self.imb_factor = imb_factor

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        img, target = self.samples[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = self.samples[:subset_size]
        self.targets = self.targets[:subset_size]
        return len(self.targets)


    def set_specific_subset(self, indices):
        self.samples = self.samples[indices]
        self.targets = [self.targets[index] for index in indices]

    def set_imbalance_data(self):
        cls_num, imb_type, imb_factor, reverse = 100, 'exp', self.imb_factor, self.reverse
        img_max = len(self.data) / cls_num
        if imb_factor == 1:
            self.num_per_cls_dict = {}
            img_num_per_cls = img_max
            for the_class in range(100):
                self.num_per_cls_dict[the_class] = img_num_per_cls
            return
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                else:
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        self.gen_imbalanced_data(img_num_per_cls)

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.samples[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.samples = new_data
        self.targets = new_targets


class cifar10_c_testset(torchvision.datasets.CIFAR10):
    def __init__(self, dataroot, transform, original=True, rotation=True, rotation_transform=None):
        super(cifar10_c_testset, self).__init__(root=dataroot, train=False, download=True, transform=transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform
        self.samples = self.data
        self.original_samples = self.samples

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def __getitem__(self, index):

        img, target = self.samples[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = self.samples[:subset_size]
        self.targets = self.targets[:subset_size]
        return len(self.targets)

    def set_specific_subset(self, indices):
        self.samples = self.samples[indices]
        self.targets = [self.targets[index] for index in indices]


def prepare_train_data(args):
    print('Preparing data...')
    if args.dataset == 'cifar10':
        trset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                             train=True, download=True, transform=tr_transforms)
    else:
        raise Exception('Dataset not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.workers)
    return trset, trloader
