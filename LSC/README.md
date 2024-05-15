#  LEARNING LABEL SHIFT CORRECTION FOR TEST- AGNOSTIC LONG-TAILED RECOGNITION

This is the source code for the paper: LEARNING LABEL SHIFT CORRECTION FOR TEST- AGNOSTIC LONG-TAILED RECOGNITION

## Requirements

* python3.9

```bash
pip install -r requirements.txt
```



## Preparation 

### Datasets

Download the dataset [Places](http://places2.csail.mit.edu/download.html), [ImageNet](http://image-net.org/index), and Cifar dateset will be downloaded automatically. 

See `Code for pretrained model in LSA` for more detail.

**NOTE**: For a fixed cifar dataset which will be used both in training long-tailed model and LSA, you need to use dataset provide by [NCL](https://github.com/Bazinga699/NCL) or remove the shuffle step when creating a long-tail CIFAR data set.

```python
    def gen_imbalanced_data(self, img_num_per_cls):
        print("gen fixd imbalanced data")
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx) <----- **remove this step**
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
```



### Long-tailed model

You can choose to train a model or use the pre-trained models we will provide soon.

Besides, you also need to generate logits and labels for LSA. There are 2 choice for you to do that. 

1) if you use our code to train your model , you can directly generate them with the help of that README. 
2) if you train your own model, you can generate them by using the function in tool/gen_feat.py.



## Usage

To reproduce the main result in the paper, please run

```bash
# run LSA on Cifar100-im100
python lsa_cifar100_im100.py --model_dir path/to/model --feat_dir path/to/logits --data_dir path/to/dataset

# run LSA on ImageNet-LT
python lsa_imagenet_x50.py --model_dir path/to/model --feat_dir path/to/logits --data_dir path/to/dataset

# run LSA on Places-LT
python lsa_places.py --model_dir path/to/model --feat_dir path/to/logits --data_dir path/to/dataset
```

Here is a more specified version of command, take ImageNet-Lt as example.

```bash
python lsa_imagenet_x50.py 
  -h, --help            show this help message and exit
  --dataset DATASET     Name of dataset
  --force_train_one FORCE_TRAIN_ONE
                        Force train a new classifier
  --force_gen_data FORCE_GEN_DATA
                        Force generate a new dataset
  --epoch EPOCH         epochs of training LDE
  --step STEP           step of the LDDataset
  --topk TOPK           train topk, -1 denotes adaptive threshold
  --tol TOL             tolerance of the threshold
  --test_topk TEST_TOPK
                        test topk
  --feat_dir FEAT_DIR   dir of logtis and label
  --model_dir MODEL_DIR
                        dir of pre-trained model
  --data_dir DATA_DIR   path of dataset
  --used_distri USED_DISTRI [USED_DISTRI ...]
                        distribution of testset, list
```



## Acknowledgment

We thank the authors for the following repositories for code reference:

**[NCL](https://github.com/Bazinga699/NCL)**, **[SADE-AgnosticLT](https://github.com/Vanint/SADE-AgnosticLT)**, [**SAR**](https://github.com/mr-eggplant/SAR)
