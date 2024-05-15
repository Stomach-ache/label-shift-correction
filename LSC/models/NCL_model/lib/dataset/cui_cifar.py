from dataset.baseset import BaseSet
import numpy as np
import random
from utils.utils import get_category_list
import math
import torchvision.transforms as transforms
from PIL import ImageFilter
from dataset.autoaug import CIFAR10Policy, Cutout
import torchvision
from PIL import Image
import random
import os
import cv2
import time
import json
import copy
import math
from .augmix.augment_and_mix import aug_mix_torch, aug_mix_cuda

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def aug_plus(aug_comb='cifar100', mode='train', plus_plus='False'):
    # PaCo's aug: https://github.com/jiequancui/Parametric-Contrastive-Learning

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if plus_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.ToPILImage(),
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
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_regular = [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    augmentation_sim_cifar = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]


    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]
    if aug_comb == 'regular_regular':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif aug_comb == 'mocov2_mocov2':
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif aug_comb == 'cifar100':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]

    if mode == 'train':
        return transform_train
    else:
        return val_transform

class MULTI_NETWORK_CIFAR_AUGPLIS(BaseSet):
    def __init__(self, mode = 'train', cfg = None, sample_id = 0, transform = None):
        super().__init__(mode, cfg, transform)
        self.sample_id = sample_id
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()

        self.transform = aug_plus(aug_comb='cifar100', mode=mode, plus_plus='False')


        if mode == 'train':
            if 'weighted' in self.sample_type:
                self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
                print('-' * 20 + ' dataset' + '-' * 20)
                print('multi_network: %d class_weight is (the first 10 classes): '%sample_id)
                print(self.class_weight[:10])

                num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

                self.instance_p = np.array([num / sum(num_list) for num in num_list])
                self.class_p = np.array([1 / self.num_classes for _ in num_list])
                num_list = [math.sqrt(num) for num in num_list]

                self.square_p = np.array([pow(num, 0.5) / sum(pow(np.array(num_list), 0.5)) for num in num_list])

                self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        if self.sample_type == "weighted_progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        if 'weighted' in self.sample_type \
                and self.mode == 'train':
            assert self.sample_type in ["weighted_balance", 'weighted_square', 'weighted_progressive']
            if self.sample_type == "weighted_balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.sample_type == "weighted_square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        if self.mode != 'train':
            image = self.transform(img)
        else:
            image = self.transform[0](img)
        meta = dict({'image_id': index})
        image_label = now_info['category_id']
        return image, image_label, meta




class CIFAR100_BASE(BaseSet):
    def __init__(self, mode = 'train', cfg = None, sample_id = 0, transform = None):
        super().__init__(mode, cfg, transform)
        self.sample_id = sample_id
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()

        self.transform = aug_plus(aug_comb='cifar100', mode='test', plus_plus='False')


        if mode == 'train':
            if 'weighted' in self.sample_type:
                self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
                print('-' * 20 + ' dataset' + '-' * 20)
                print('multi_network: %d class_weight is (the first 10 classes): '%sample_id)
                print(self.class_weight[:10])

                num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

                self.instance_p = np.array([num / sum(num_list) for num in num_list])
                self.class_p = np.array([1 / self.num_classes for _ in num_list])
                num_list = [math.sqrt(num) for num in num_list]

                self.square_p = np.array([pow(num, 0.5) / sum(pow(np.array(num_list), 0.5)) for num in num_list])

                self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        if self.sample_type == "weighted_progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        if 'weighted' in self.sample_type \
                and self.mode == 'train':
            assert self.sample_type in ["weighted_balance", 'weighted_square', 'weighted_progressive']
            if self.sample_type == "weighted_balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.sample_type == "weighted_square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        if self.mode != 'train':
            image = self.transform(img)
        else:
            image = self.transform(img)
        meta = dict({'image_id': index})
        image_label = now_info['category_id']
        return image, image_label, meta


class MULTI_NETWORK_CIFAR_MOCO_AUGPLIS(BaseSet):
    def __init__(self, mode = 'train', cfg = None, sample_id = 0, transform = None):
        super().__init__(mode, cfg, transform)
        self.sample_id = sample_id
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)
        self.mode = mode

        # strong augmentation. remove it you will get weak augmentation which is defined in BaseSet.update_transform() .
        self.transform = aug_plus(aug_comb='cifar100', mode=mode, plus_plus='False')

        self.class_dict = self._get_class_dict()
        if mode == 'train':
            if 'weighted' in self.sample_type:
                self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
                print('-' * 20 + ' dataset' + '-' * 20)
                print('multi_network: %d class_weight is (the first 10 classes): '%sample_id)
                print(self.class_weight[:10])

                num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

                self.instance_p = np.array([num / sum(num_list) for num in num_list])
                self.class_p = np.array([1 / self.num_classes for _ in num_list])
                num_list = [math.sqrt(num) for num in num_list]

                self.square_p = np.array([pow(num, 0.5) / sum(pow(np.array(num_list), 0.5)) for num in num_list])

                self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        if self.sample_type == "weighted_progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            #print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        if 'weighted' in self.sample_type \
                and self.mode == 'train':
            assert self.sample_type in ["weighted_balance", 'weighted_square', 'weighted_progressive']
            if self.sample_type == "weighted_balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.sample_type == "weighted_square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        meta = dict({'image_id': index})
        image_label = now_info['category_id']

        if self.mode != 'train':
            image1 = self.transform(img)
            return image1, image_label, meta

        image1 = self.transform[0](img)
        image2 = self.transform[1](img)

        return (image1, image2), image_label, meta



class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10



    def __init__(self, mode, cfg, root = './dataset/cifar', imb_type='exp',
                 transform=None, target_transform=None, download=True):
        train = True if mode == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.cfg = cfg
        self.train = train
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE

        rand_number = cfg.DATASET.IMBALANCECIFAR.RANDOM_SEED
        if self.train:
            np.random.seed(rand_number)
            random.seed(rand_number)
            imb_factor = self.cfg.DATASET.IMBALANCECIFAR.RATIO
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.data_format_transform()
            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])

        self.data = self.all_info
        '''
            load the generated CAM-based dataset
        '''


        if self.cfg.DATASET.USE_CAM_BASED_DATASET and mode == 'train':
            assert os.path.isfile(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH), \
                'the CAM-based generated json file does not exist!'
            self.data = self.data + json.load(open(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH))
            new_data = []
            for info in self.data:
                if 'fpath' not in info:
                    new_data.append(copy.deepcopy(info))
                    continue
                img = self._load_image(info)
                new_data.append({
                    'image': img,
                    'category_id': info['category_id']
                })
            self.data = new_data

        self.class_dict = self._get_class_dict()

        print("{} Mode: Contain {} images".format(mode, len(self.data)))
        self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
            print('-'*20+'in imbalance cifar dataset'+'-'*20)
            print('class_weight is: ')
            print(self.class_weight)

            num_list, cat_list = get_category_list(self.get_annotations(), self.cls_num, self.cfg)
            self.instance_p = np.array([num / sum(num_list) for num in num_list])
            self.class_p = np.array([1/self.cls_num for _ in num_list])
            num_list = [math.sqrt(num) for num in num_list]
            self.square_p = np.array([num / sum(num_list) for num in num_list])


    def update(self, epoch):
        self.epoch = max(0, epoch-self.cfg.TRAIN.TWO_STAGE.START_EPOCH) if self.cfg.TRAIN.TWO_STAGE.DRS else epoch
        if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train\
            and (not self.cfg.TRAIN.TWO_STAGE.DRS or (self.cfg.TRAIN.TWO_STAGE.DRS and self.epoch)):
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", 'square', 'progressive']
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "square":
                sample_class = np.random.choice(np.arange(self.cls_num), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.cls_num), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img, target = self.data[index]['image'], self.data[index]['category_id']
        meta = dict()
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.cfg.TRAIN.SAMPLER.TYPE == "bbn sampler" and self.cfg.TRAIN.SAMPLER.BBN_SAMPLER.TYPE == "reverse":
            sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
            sample_img, sample_label = self.data[sample_index]['image'], self.data[sample_index]['category_id']
            sample_img = Image.fromarray(sample_img)
            sample_img = self.transform(sample_img)
            if self.target_transform is not None:
                sample_label = self.target_transform(sample_label)
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label
        return img, target, meta

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print(fpath)
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "pillow open {} failed".format(fpath)
                time.sleep(0.1)

    def _load_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight


    def _get_trans_image(self, img_idx):
        now_info = self.data[img_idx]
        img = now_info['image']
        img = Image.fromarray(img)
        return self.transform(img)[None, :, :, :]

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for d in self.all_info:
            annos.append({'category_id': int(d['category_id'])})
        return annos

    def _get_image(self, now_info):
        img = now_info['image']
        return copy.deepcopy(img)

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            for img in self.data[selec_idx, ...]:
                new_data.append({
                    'image': img,
                    'category_id': the_class
                })
        self.all_info = new_data

    def data_format_transform(self):
        new_data = []
        targets_np = np.array(self.targets, dtype=np.int64)
        assert len(targets_np) == len(self.data)
        for i in range(len(self.data)):
            new_data.append({
                'image': self.data[i],
                'category_id': targets_np[i],
            })
        self.all_info = new_data


    def __len__(self):
        return len(self.all_info)



class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class MULTI_NETWORK_CIFAR_AUGMIX(BaseSet):
    def __init__(self, mode = 'train', cfg = None, sample_id = 0, transform = None):
        super().__init__(mode, cfg, transform)
        self.sample_id = sample_id
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()

        # self.transform = aug_plus(aug_comb='cifar100', mode=mode, plus_plus='False')
        self.train = mode == 'train'
        self.progress = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
            self.use_randaug = cfg.DATASET.AUGMIX.randaug
            self.randaug = aug_plus(aug_comb='cifar100', mode=mode, plus_plus='False')[0]
            self.severity = self.cfg.DATASET.AUGMIX.aug_severity
            self.all_ops = self.cfg.DATASET.AUGMIX.all_ops
            self.width = self.cfg.DATASET.AUGMIX.width
            self.depth = self.cfg.DATASET.AUGMIX.depth
            self.mix_alpha = self.cfg.DATASET.AUGMIX.alpha
        else:
            self.transform = self.progress
            self.use_randaug = False

        if mode == 'train':
            if 'weighted' in self.sample_type:
                self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
                print('-' * 20 + ' dataset' + '-' * 20)
                print('multi_network: %d class_weight is (the first 10 classes): '%sample_id)
                print(self.class_weight[:10])

                num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

                self.instance_p = np.array([num / sum(num_list) for num in num_list])
                self.class_p = np.array([1 / self.num_classes for _ in num_list])
                num_list = [math.sqrt(num) for num in num_list]

                self.square_p = np.array([pow(num, 0.5) / sum(pow(np.array(num_list), 0.5)) for num in num_list])

                self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        if self.sample_type == "weighted_progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)

        # temp use
        s_adapt = False
        if s_adapt:
            if epoch < 100:
                self.severity = 1
            elif epoch < 200:
                self.severity = 3
            elif epoch < 300:
                self.severity = 5
            else:
                self.severity = 7

    def __getitem__(self, index):
        if 'weighted' in self.sample_type and self.mode == 'train':
            assert self.sample_type in ["weighted_balance", 'weighted_square', 'weighted_progressive']
            if self.sample_type == "weighted_balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.sample_type == "weighted_square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]

        img_array = self._get_image(now_info)
        img_org = Image.fromarray(img_array)
        img_trans = self.transform(img_org)
        meta = dict({'image_id': index})
        image_label = now_info['category_id']
        if self.train:
            meta['augmix'] = [aug_mix_torch(img_trans, self.progress, aug_severity=self.severity, mixture_width=self.width, mixture_depth=self.depth, all_ops=self.all_ops, alpha=self.mix_alpha),
                              aug_mix_torch(img_trans, self.progress, aug_severity=self.severity, mixture_width=self.width, mixture_depth=self.depth, all_ops=self.all_ops, alpha=self.mix_alpha),]
            if self.use_randaug:
                img = self.randaug(img_array)
            else:
                img = self.progress(img_trans)
        else:
            img = self.transform(img_org)

        return img, image_label, meta

    # def aug_mix_wrap(self, img_trans):
    #     return aug_mix_torch(img_trans, self.progress, aug_severity=self.severity, mixture_width=self.width, mixture_depth=self.depth, all_ops=self.all_ops, alpha=self.mix_alpha)