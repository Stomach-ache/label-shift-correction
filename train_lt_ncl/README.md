# Code for pretrained model in LSC

## Setup

### Main requirements

```bash
torch >= 1.7.1 #This is the version I am using, other versions may be accteptable, if there is any problem, go to https://pytorch.org/get-started/previous-versions/ to get right version(espicially CUDA) for your machine.
tensorboardX >= 2.1 #Visualization of the training process.
tensorflow >= 1.14.0 #convert long-tailed cifar datasets from tfrecords to jpgs.
Python 3.6 #This is the version I am using, other versions(python 3.x) may be accteptable.
```

### Detailed requirement

```bash
pip install -r requirements.txt
```

### Prepare datasets

**This part is mainly based on https://github.com/Bazinga699/NCL**

- #### Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `fpath`, `im_height`, `im_width` and `category_id`.

Here is an example.

```
{
    'annotations': [
                    {
                        'image_id': 1,
                        'fpath': '/data/iNat18/images/train_val2018/Plantae/7477/3b60c9486db1d2ee875f11a669fbde4a.jpg',
                        'im_height': 600,
                        'im_width': 800,
                        'category_id': 7477
                    },
                    ...
                   ]
    'num_classes': 8142
}
```

- #### CIFAR-LT

  [Cui et al., CVPR 2019](https://arxiv.org/abs/1901.05555) firstly proposed the CIFAR-LT. They provided the [download link](https://github.com/richardaecn/class-balanced-loss/blob/master/README.md#datasets) of CIFAR-LT, and also the [codes](https://github.com/richardaecn/class-balanced-loss/blob/master/README.md#datasets) to generate the data, which are in TensorFlow. 

     You can follow the steps below to get this version of  CIFAR-LT:

        1. Download the Cui's CIFAR-LT in [GoogleDrive](https://drive.google.com/file/d/1NY3lWYRfsTWfsjFPxJUlPumy-WFeD7zK/edit) or [Baidu Netdisk ](https://pan.baidu.com/s/1rhTPUawY3Sky6obDM4Tczg) (password: 5rsq). Suppose you download the data and unzip them at path `/downloaded/data/`.
        2. Run tools/convert_from_tfrecords, and the converted CIFAR-LT and corresponding jsons will be generated at `/downloaded/converted/`.

  ```bash
  # Convert from the original format of CIFAR-LT
  python tools/convert_from_tfrecords.py  --input_path /downloaded/data/ --output_path /downloaded/converted/
  ```

- #### ImageNet-LT

  You can use the following steps to convert from the original images of ImageNet-LT.

  1. Download the original [ILSVRC-2012](http://www.image-net.org/). Suppose you have downloaded and reorgnized them at path `/downloaded/ImageNet/`, which should contain two sub-directories: `/downloaded/ImageNet/train` and `/downloaded/ImageNet/val`.
  2. Directly replace the data root directory in the file `dataset_json/ImageNet_LT_train.json`, `dataset_json/ImageNet_LT_val.json`,You can handle this with any editor, or use the following command.

  ```bash
  # replace data root
  python tools/replace_path.py --json_file dataset_json/ImageNet_LT_train.json --find_root /media/ssd1/lijun/ImageNet_LT --replaces_to /downloaded/ImageNet
  
  python tools/replace_path.py --json_file dataset_json/ImageNet_LT_val.json --find_root /media/ssd1/lijun/ImageNet_LT --replaces_to /downloaded/ImageNet
  
  ```

- #### Places_LT

  You can use the following steps to convert from the original format of Places365-Standard.

  1. The images and annotations should be downloaded at [Places365-Standard](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar) firstly. Suppose you have downloaded them at  path `/downloaded/Places365/`.
  2. Directly replace the data root directory in the file `dataset_json/Places_LT_train.json`, `dataset_json/Places_LT_val.json`,You can handle this with any editor, or use the following command.

  ```bash
  # replace data root
  python tools/replace_path.py --json_file dataset_json/Places_LT_train.json --find_root /media/ssd1/lijun/data/places365_standard --replaces_to /downloaded/Places365
  
  python tools/replace_path.py --json_file dataset_json/Places_LT_val.json --find_root /media/ssd1/lijun/data/places365_standard --replaces_to /downloaded/Places365
  
  ```

## Training

 **CIFAR100 im100**

```
bash train_cifar100_im100_NCL_200epoch.sh 
```

**ImageNet-LT**

```
bash train_ImageNet_LT_x50.sh
```

## Generate logits and label for LSA

To generate logits and label for LSA, please modify commands below.

**CIFAR100 im100**

```
python main/gen_logits_label.py --cfg config/CIFAR/CIFAR100/cifar100_im100_NCL_200epoch.yaml --modelpath path/to/model --distpath path/to/save
```

**ImageNet-LT**

```
python main/gen_logits_label.py --cfg config/ImageNet-LT/ImageNet_LT_x50.yaml --modelpath path/to/model --distpath path/to/save
```



## Acknowledgements
This project is forked from  [NCL](https://github.com/Bazinga699/NCL).
