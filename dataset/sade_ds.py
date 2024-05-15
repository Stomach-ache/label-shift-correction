import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision
import random
from PIL import Image
from PIL import ImageFilter


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, reverse=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        img_max = len(self.data) / cls_num
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
        return img_num_per_cls

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
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


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



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class TestAgnosticImbalanceCIFAR100DataLoader(DataLoader):
    """
    Imbalance Cifar100 Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
                 retain_epoch_size=True, imb_type='exp', imb_factor=0.01, test_imb_factor=0, reverse=False):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_trsfm)  # test set

        if training:
            dataset = IMBALANCECIFAR100(data_dir, train=True, download=True, transform=train_trsfm, imb_type=imb_type,
                                        imb_factor=imb_factor)
            val_dataset = test_dataset
        else:
            dataset = IMBALANCECIFAR100(data_dir, train=True, download=True, transform=train_trsfm, imb_type=imb_type,
                                        imb_factor=test_imb_factor, reverse=reverse)
            train_dataset = IMBALANCECIFAR100(data_dir, train=False, download=True,
                                              transform=TwoCropsTransform(train_trsfm), imb_type=imb_type,
                                              imb_factor=test_imb_factor, reverse=reverse)
            val_dataset = IMBALANCECIFAR100(data_dir, train=False, download=True, transform=test_trsfm,
                                            imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse)

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 100

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs)  # Note that sampler does not apply to validation set

    def train_set(self):
        return DataLoader(dataset=self.train_dataset, shuffle=True, **self.init_kwargs)

    def test_set(self):
        return DataLoader(dataset=self.val_dataset, shuffle=False, **self.init_kwargs)


class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num


class LT_Dataset(Dataset):

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label


class ImageNetLTDataLoader(DataLoader):
    """
    ImageNetLT Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False,
                 retain_epoch_size=True,
                 train_txt='./data_txt/ImageNet_LT/ImageNet_LT_train.txt',
                 val_txt='./data_txt/ImageNet_LT/ImageNet_LT_val.txt',
                 test_txt='./data_txt/ImageNet_LT/ImageNet_LT_test.txt',
                 train_trans=None):
        # train_trsfm = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        if train_trans is None:
            train_trsfm = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_trsfm = train_trans
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if training:
            dataset = LT_Dataset(data_dir, train_txt, train_trsfm)
            train_dataset = dataset
            val_dataset = LT_Dataset(data_dir, val_txt, test_trsfm)
        else:  # test
            dataset = LT_Dataset(data_dir, test_txt, train_trsfm)
            train_dataset = LT_Dataset(data_dir, test_txt, TwoCropsTransform(train_trsfm))
            val_dataset = LT_Dataset(data_dir, test_txt, test_trsfm)
            self.trained_dataset = LT_Dataset(data_dir, train_txt, train_trsfm)

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 1000

        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs,
                         sampler=sampler)  # Note that sampler does not apply to validation set

    def train_set(self):
        return DataLoader(dataset=self.train_dataset, shuffle=True, **self.init_kwargs)

    def test_set(self):
        return DataLoader(dataset=self.val_dataset, shuffle=False, **self.init_kwargs)



class LT_Dataset_Eval(Dataset):
    num_classes = 365

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()
        self.labels = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class LT365_Dataset(Dataset):
    num_classes = 365

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [i for i in range(self.num_classes)]
        # for i in range(self.num_classes):
        #     self.class_map[sorted_classes[i]] = i

        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
        self.labels = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class Places_LT(DataLoader):
    def __init__(self, data_dir="", batch_size=60, num_workers=40, training=True,
                 train_txt="./data_txt/Places_LT_v2/Places_LT_train.txt",
                 eval_txt="./data_txt/Places_LT_v2/Places_LT_val.txt",
                 test_txt="./data_txt/Places_LT_v2/Places_LT_test.txt"):
        self.num_workers = num_workers
        self.batch_size = batch_size

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = LT365_Dataset(data_dir, train_txt, transform=transform_train)

        if training:
            dataset = LT365_Dataset(data_dir, train_txt, transform=transform_train)
            val_dataset = LT_Dataset_Eval(data_dir, eval_txt, transform=transform_test,
                                           class_map=train_dataset.class_map)
        else:  # test
            dataset = LT365_Dataset(data_dir, test_txt, transform_train)
            train_dataset = LT365_Dataset(data_dir, test_txt, TwoCropsTransform(transform_train))
            val_dataset = LT365_Dataset(data_dir, test_txt, transform_test)
            self.trained_dataset = LT365_Dataset(data_dir, train_txt, transform_train)

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.cls_num_list = train_dataset.cls_num_list
        """
        self.data = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        """

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': num_workers
        }

        # super().__init__(dataset=self.data)
        super().__init__(dataset=self.dataset, **self.init_kwargs)

    def split_validation(self):
        # If you do not want to validate:
        #     return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)

    def train_set(self):
        return DataLoader(dataset=self.train_dataset, **self.init_kwargs)

    def test_set(self):
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)


