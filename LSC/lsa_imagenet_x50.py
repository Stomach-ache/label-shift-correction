import os
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
import os.path as osp
import matplotlib.pyplot as plt
import argparse

from dataset.sade_ds import ImageNetLTDataLoader
from methods.pda import PDA
import models.model_load as model_load
from utils.valid_func import validate, pse_label_distri_dist
from utils.cli_utils import AverageMeter
from utils.utils import plot_bar_chart_onehot, plot_bar_chart,  get_num_cls_list_from_label, time_tester

FIG_OUTPUT_PATH = 'output/fig/'
FIG_CNT = 0
lbl_freq_list = []
num_per_cls_list = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def gen_fig_path():
    out_path = os.path.join(FIG_OUTPUT_PATH, FIG_CNT.__str__())
    return out_path


def gen_output(model, train_loader, feat_out=True, out_dir=None):
    # gen feature
    train_output_list = [[], [], []]
    with torch.no_grad():
        for images, targets in train_loader:
            images = images.cuda()
            # targets = targets.cuda()
            if feat_out:
                output = model(images, feature_flag=True)
            else:
                output = model(images)
            for f, l in zip(output, train_output_list):
                l.append(f.cpu())

    train_feat = torch.stack([torch.cat(l) for l in train_output_list])
    if out_dir is not None:
        torch.save(train_feat, out_dir)
    return train_feat


class LDE(nn.Module):
    def __init__(self, input_dim, output_dim, ensemble=1):
        super().__init__()
        self.hidden_dim = output_dim*3
        self.mlp = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(input_dim, self.hidden_dim)),
            ('act1', nn.ReLU()),
            ('linear2', nn.Linear(self.hidden_dim, output_dim)),
        ]))
        self.ensemble = ensemble
        self.in_dim = input_dim
        self.out_dim = output_dim

    def forward(self, input):
        output = self.mlp(input)
        return output


def gen_imbalanced_data(feat, img_num_per_cls, targets, target_one_hot, idx_each_class, classes=None):
    if classes is None:
        classes = torch.unique(targets)
    distri_target_sorted = img_num_per_cls / img_num_per_cls.sum()
    distri_target = torch.zeros_like(distri_target_sorted)
    distri_target[classes] = distri_target_sorted
    feat_each_cls = torch.zeros((img_num_per_cls.shape[0], feat.shape[1]))

    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = idx_each_class[the_class]
        if idx.shape[0] <= the_img_num:
            feat_each_cls[the_class] = torch.mean(feat[idx], dim=0)
        else:
            selec_idx = torch.multinomial(torch.ones(idx.size(0)), the_img_num, replacement=True)
            feat_each_cls[the_class] = torch.mean(feat[selec_idx], dim=0)

    feat_mean = (feat_each_cls * distri_target.reshape(-1, 1)).sum(dim=0)

    return feat_mean, distri_target


class LDDateset(Dataset):
    def __init__(self, feat: torch.Tensor, label: torch.Tensor, imb_ub=50.0, step=.1, load_path=None, clc_num=100):
        super().__init__()
        self.feat = feat
        self.label = label
        self.imb_ub = imb_ub
        self.step = step
        self.distri_list = []
        self.distri_targets = []

        cur_imb = 1.0
        self.clc_num = clc_num
        self.label_one_hot = F.one_hot(label).float()
        self.num_cls_list = []

        time_start = time.time()
        if load_path is None or not osp.exists(osp.join(load_path, 'distri_list')):
            classes = torch.unique(label)
            idx_each_class = []
            img_max = 0
            for the_class in classes:
                idx = torch.nonzero(label == the_class).squeeze(1)
                idx_each_class.append(idx)
                idx_len = len(idx)
                self.num_cls_list.append(idx_len)
                img_max = max(img_max, idx_len)

            sorted_num_cls, ind = torch.sort(torch.tensor(self.num_cls_list), stable=True)
            f1_flag = True
            f2_flag = True
            while cur_imb < imb_ub:

                img_num_per_cls1 = self.set_imbalance_data(img_max, cur_imb, False, self.clc_num)
                img_num_per_cls2 = [i for i in reversed(img_num_per_cls1)]
                img_num_per_cls1_tensor = torch.tensor(img_num_per_cls1)
                img_num_per_cls2_tensor = torch.tensor(img_num_per_cls2)

                if f1_flag:
                    feat1, distri1 = gen_imbalanced_data(self.feat, img_num_per_cls1_tensor, label, self.label_one_hot,
                                                         classes=ind, idx_each_class=idx_each_class)
                    self.distri_list.append(feat1)
                    self.distri_targets.append(distri1)
                if f2_flag:
                    feat2, distri2 = gen_imbalanced_data(self.feat, img_num_per_cls2_tensor, label, self.label_one_hot,
                                                         classes=ind, idx_each_class=idx_each_class)
                    self.distri_list.append(feat2)
                    self.distri_targets.append(distri2)

                cur_imb += step
                # cur_imb *= step
            time_cost = time.time() - time_start
            print(f'It takes {time_cost} sec to complete dataset')
        else:
            self.load_param(load_path)
        return

    def __len__(self):
        return len(self.distri_list)

    def __getitem__(self, item):
        return self.distri_list[item], self.distri_targets[item]

    def save_param(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.distri_list, osp.join(file_path, 'distri_list'))
        torch.save(self.distri_targets, osp.join(file_path, 'distri_targets'))

    def load_param(self, file_path):
        self.distri_list = torch.load(osp.join(file_path, 'distri_list'))
        self.distri_targets = torch.load(osp.join(file_path, 'distri_targets'))

    @staticmethod
    def set_imbalance_data(img_max, imb_factor, reverse, cls_num):
        img_num_per_cls = []
        if imb_factor > 1.0:
            reverse = not reverse
            imb_factor = 1. / imb_factor
        for cls_idx in range(cls_num):
            if reverse:
                num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
            else:
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        # random.shuffle(img_num_per_cls)
        # img_num_per_cls_torch = torch.tensor(img_num_per_cls)
        return img_num_per_cls

    @staticmethod
    def get_topK(logits, label, tol=.9):
        num_sample, num_label = logits.shape
        label_expanded = label.unsqueeze(1).expand_as(logits)
        top_k_values, top_indices = torch.topk(logits, num_label, dim=1)
        k = 1
        for k in range(1, num_label + 1):
            comparison = label_expanded[:, :k] == top_indices[:, :k]
            pos_ratio = (comparison.any(dim=1)).sum() / num_sample
            if pos_ratio > tol:
                break
        return k

    @staticmethod
    def topk_mask(tensor, k=5, dim=1, mask_val=0):
        _, indices = tensor.topk(k, dim=dim)
        mask = torch.full_like(tensor, mask_val)
        mask.scatter_(dim, indices, tensor.gather(dim, indices))
        return mask

    def draw_distribution(self, ind):
        pred = self.distri_list[ind].reshape(3, -1).sum(dim=0)
        y_gt = self .distri_targets[ind]
        plot_bar_chart(y_gt[lbl_freq_list] * 100, pred[lbl_freq_list], f'No. {ind} Distribution')


def count_masked_columns(mask, title='Count of Masked Columns'):
    count = torch.sum(mask, dim=0)
    x = range(count.size(0))
    plt.bar(x, count)
    plt.xlabel('Column')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()
    return count


def filter_pred_logits_entropy(pred_logits, threshold):
    entropy = -torch.sum(torch.softmax(pred_logits, dim=1) * torch.log_softmax(pred_logits, dim=1), dim=1)
    filtered_samples = pred_logits[entropy > threshold]
    mean_value = torch.mean(filtered_samples)
    num_samples = filtered_samples.shape[0]

    return mean_value, num_samples


def mask_topk(x, label, tol=0.9):
    chosen_k = LDDateset.get_topK(x, label=label, tol=tol)
    masked_x = LDDateset.topk_mask(x, k=chosen_k, dim=1)
    return masked_x, chosen_k


def proc_ide_logits_list(train_logits, label, top_k='ada', tol=0.9):
    ide_logits_list = []
    for i in range(3):
        if top_k == 'ada':
            chosen_k = LDDateset.get_topK(train_logits[i], label=label, tol=tol)
        else:
            chosen_k = top_k
        print(f"chosen_k: {chosen_k}", end='\t')
        logits_item = LDDateset.topk_mask(train_logits[i], k=chosen_k, dim=1)
        ide_logits_list.append(logits_item)
    print("")
    ide_logits = torch.hstack(ide_logits_list)
    return ide_logits


def proc_logtis_and_label_list(load_root_dir_list, top_k, tol):
    l_list, t_list = [], []
    for root_dir in load_root_dir_list:
        l = torch.load(os.path.join(root_dir, 'logits'), 'cpu')
        t = torch.load(os.path.join(root_dir, 'label'), 'cpu')
        l_list.append(l)
        t_list.append(t)

    ide_logits_list = []
    for l, t in zip(l_list, t_list):
        ide_logits_list.append(proc_ide_logits_list(l, t, top_k, tol=tol))

    label = torch.cat(t_list)
    ide_logits = torch.cat(ide_logits_list)
    return ide_logits, label


def draw_train_logtis(ide_logits, label):
    pred = ide_logits.reshape(-1, 3, 1000).sum(dim=1)
    ygt = F.one_hot(label, 1000).float().mean(dim=0)
    plot_bar_chart(ygt[lbl_freq_list] * 1000, pred.mean(dim=0)[lbl_freq_list])
    plot_bar_chart_onehot(ygt[lbl_freq_list], pred[:, lbl_freq_list])


def get_dataset_wrap(feat_dir, top_k, save_data_path=None, tol=.95, force_gen_data=True, step=.1):
    load_root_dir_list = [
        feat_dir,
        ]

    ide_logits, label = proc_logtis_and_label_list(load_root_dir_list, top_k, tol)
    if osp.exists(osp.join(save_data_path, 'distri_list')) and not force_gen_data:
        ide_train_set = LDDateset(ide_logits, label, imb_ub=100.0, step=step, load_path=save_data_path, clc_num=1000)
    else:
        ide_train_set = LDDateset(ide_logits, label, imb_ub=100.0, step=step, clc_num=1000)
        ide_train_set.save_param(save_data_path)
    return ide_train_set


def run_one(seed, feat_dir, top_k='ada', num_epoch=100, tol=0.95, force_gen_data=True, step=.1):
    set_seed(seed)
    save_path = f'{feat_dir}/distri/_seed{seed}_k_{top_k}_tol{tol}_step{step}'
    ide_train_set = get_dataset_wrap(feat_dir, top_k, save_path, tol, force_gen_data, step=step)
    # global lbl_freq_list
    ide_loader = DataLoader(ide_train_set, batch_size=128, shuffle=True)

    ide_model = LDE(ide_train_set.feat.shape[1], ide_train_set.clc_num, ensemble=3)
    optimizer = torch.optim.Adam(ide_model.parameters(), lr=1e-5, betas=(.9, .999))

    tot_loss = 0
    for epoch in range(num_epoch):
        # print(f"EPOCH [{epoch}]")
        tot_loss = 0
        for feat, label in ide_loader:
            output = ide_model(feat)
            loss = F.kl_div(F.log_softmax(output, dim=1), label, reduction='batchmean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss
    print(f"tot_loss : {tot_loss}")
    train_feat_mean = ide_train_set.feat.mean(0)
    train_distri = ide_train_set.label_one_hot.float().mean(0)
    pred_train_distri = ide_model(train_feat_mean)
    train_dist = pse_label_distri_dist(F.softmax(pred_train_distri, dim=0), train_distri)
    print(f"train pse_label_distri_dist: {train_dist}")
    return ide_model

def mask_logits_tensor(all_3_feat, final_tensor, testk, class_num=1000):
    for i in range(3):
        one_masked_logits = LDDateset.topk_mask(all_3_feat[:, i], k=testk[i])
        final_tensor[:, i*class_num:(i+1)*class_num] = one_masked_logits
    return final_tensor


def check_imb(head_prob, med_prob, tail_prob):
    """
    return 1: forward; 0:uniform; -1: backward
    """
    # return forward
    if head_prob > 1.5 * tail_prob and head_prob > med_prob:
        return 1
    if tail_prob > 1.5 * head_prob and tail_prob > med_prob:
        return -1
    return 0



draw_sum = 0
draw_dist_list = [[], []]
draw_dist_mean_list = [[], []]

def LDE_adapt(net, lde_model, val_dataset, val_loader, org_logits_path, force_pred_logits=False):
    param, param_names = PDA.collect_params(net)
    lr = 1e-3  # args.lr
    optimizer = torch.optim.SGD(param, lr, momentum=0.9)
    PDA_model = PDA(net, optimizer=optimizer, class_num=1000)
    PDA_model = PDA_model.cuda()
    PDA_model.temper = 1.0
    lde_model = lde_model.cuda()
    net.div_out = True
    overall_targets = []
    feat_list = []
    PDA_model.eval()
    if not os.path.exists(org_logits_path) or force_pred_logits:
        with torch.no_grad():
            for _, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                images = images.cuda()
                target = target.cuda()
                output = PDA_model(images, update_weight=False, consistency_reg=False)
                feat2 = output
                feat_all = torch.hstack([feat2[j] for j in range(3)])
                feat_list.append(feat_all)
                overall_targets.append(target.cpu().detach())
        all_feat = torch.cat(feat_list)
        os.makedirs(os.path.dirname(org_logits_path), exist_ok=True)
        torch.save(all_feat, org_logits_path)
    else:
        all_feat = torch.load(org_logits_path)
    all_3_feat = all_feat.reshape(-1, 3, 1000)
    final_tensor = torch.zeros_like(all_feat)
    global num_per_cls_list
    tail_idx, head_idx = num_per_cls_list < 20, num_per_cls_list > 100
    med_idx = torch.logical_not(torch.logical_or(tail_idx, head_idx))

    final_tensor_list = []
    infer_distri_list = []
    train_topk = 220
    final_testk = train_topk
    testK_list = [i for i in range(100, 1100, 100)] + [train_topk]
    class_num = 1000
    for i in testK_list:
        ft = mask_logits_tensor(all_3_feat, torch.zeros_like(all_feat), [i] * 3, class_num=class_num)
        final_tensor_list.append(ft.cpu())
        infer_distri_list.append(F.softmax(lde_model(ft.mean(dim=0)), dim=0))

    all_ft = torch.stack(final_tensor_list).reshape(len(final_tensor_list), -1, 3, class_num)
    all_dist = torch.stack(infer_distri_list)
    mean_ft = all_ft.mean(dim=1).mean(1)
    head_cmp = mean_ft[:, head_idx].sum(dim=-1) / mean_ft.sum(dim=1)
    med_cmp = mean_ft[:, med_idx].sum(dim=-1) / mean_ft.sum(dim=1)
    tail_cmp = mean_ft[:, tail_idx].sum(dim=-1) / mean_ft.sum(dim=1)
    mean_dist = all_dist.mean(dim=0)
    head_prob = mean_dist[head_idx].mean()
    med_prob = mean_dist[med_idx].mean()
    tail_prob = mean_dist[tail_idx].mean()

    dist_flag = check_imb(head_prob, med_prob, tail_prob)
    print(f"head prob: {head_prob}\nmed prb: {med_prob}\ntail prob : {tail_prob}")
    if dist_flag == 1:
        head_k = 3
        _, idx = torch.sort(head_cmp)
        final_tensor = 0
        final_testk = []
        for i in range(1, head_k + 1):
            final_testk.append(testK_list[idx[-i]])
            final_tensor = final_tensor_list[idx[-i]] + final_tensor
        final_tensor = final_tensor / head_k
        infer_distri = F.softmax(lde_model(final_tensor.mean(dim=0).cuda()), dim=0)
    elif dist_flag == 0:
        final_tensor = torch.ones_like(all_feat)
        infer_distri = torch.ones(class_num, device='cuda')
    else:
        tail_k = 1
        _, idx = torch.sort(tail_cmp)
        final_tensor = 0
        final_testk = []
        for i in range(1, tail_k + 1):
            final_testk.append(testK_list[idx[-i]])

            final_tensor = final_tensor_list[idx[-i]] + final_tensor
        final_tensor = final_tensor / tail_k
        infer_distri = F.softmax(lde_model(final_tensor.mean(dim=0).cuda()), dim=0)
    print(f"dist flag {dist_flag}")
    print(f"test top k {final_testk}")

    PDA_model.bsce_weight = infer_distri.flatten().detach()
    return PDA_model


def get_imagenet_lt_args():
    parser = argparse.ArgumentParser(description='LSA exps')
    parser.add_argument('--dataset', default='ImageNet', help='Name of dataset')
    parser.add_argument('--force_train_one', default=False, type=bool, help='Force train a new classifier')
    parser.add_argument('--force_gen_data', default=False, type=bool, help='Force generate a new dataset')
    parser.add_argument('--epoch', default=100, type=int, help='epochs of training LDE')
    parser.add_argument('--step', default=0.1, type=float, help='step of the LDDataset')
    parser.add_argument('--topk', default=-1, type=int, help='train topk, -1 denotes adaptive threshold')
    parser.add_argument('--tol', default=0.99, type=float, help='tolerance of the threshold ')
    parser.add_argument('--test_topk', default=120, type=int, help='test topk')
    parser.add_argument('--feat_dir', default='output/feat/imagenet_ncl_x50/rangaug_seed22', help='dir of logtis and label')
    parser.add_argument('--model_dir', default='pre_trained/imagenet/ncl_x50_imagenet.pth', help='dir of pre-trained model')
    parser.add_argument('--data_dir', default='../data/ImageNet', help='path of dataset')
    parser.add_argument('--used_distri', default=['forward50', 'forward25', 'forward10', 'forward5', 'uniform', 'backward5', 'backward10', 'backward25', 'backward50'], nargs='+', help='distribution of testset, list')
    # ['forward50', 'forward25', 'forward10', 'forward5', 'uniform', 'backward5', 'backward10', 'backward25', 'backward50'],
    return parser.parse_args()



def main():
    args = get_imagenet_lt_args()

    used_distri = args.used_distri
    seedlist = [0]
    batch_size, num_workers = 64, 16
    name_dataset = args.dataset
    num_epoch = args.epoch
    force_train_one = args.force_train_one
    force_gen_data = args.force_gen_data
    if args.topk < 0:
        topk = 'ada'
    else:
        topk = args.topk
    tol = args.tol
    step = args.step
    feat_dir = args.feat_dir
    model_dir = args.model_dir
    data_dir = args.data_dir

    # create imb Validation Dataset
    net = model_load.load_ncl_imagenet_x50(model_dir)
    net = net.cuda()
    result = []
    record_str = ""

    for distri in used_distri:
        top1_meter = AverageMeter('Acc@1', ':6.2f')
        top5_meter = AverageMeter('Acc@5', ':6.2f')
        head_meter = AverageMeter('Head', ':6.2f')
        med_meter = AverageMeter('med', ':6.2f')
        tail_meter = AverageMeter('tail', ':6.2f')
        test_txt = f'./data_txt/ImageNet_LT/ImageNet_LT_{distri}.txt'

        for s in seedlist:
            imagenet_data_loader = ImageNetLTDataLoader(
                data_dir,
                batch_size=128,
                shuffle=False,
                training=False,
                num_workers=8,
                test_txt=test_txt,
            )
            val_set = imagenet_data_loader.val_dataset
            val_set.num_per_cls_list = get_num_cls_list_from_label(val_set.labels)  #
            val_loader = imagenet_data_loader.test_set()
            train_set = imagenet_data_loader.trained_dataset
            train_set.num_per_cls_list = get_num_cls_list_from_label(train_set.labels)
            global num_per_cls_list
            num_per_cls_list = torch.tensor(train_set.num_per_cls_list)
            global lbl_freq_list
            _, lbl_freq_list = torch.sort(num_per_cls_list)
            model_dir = os.path.join(feat_dir, f'model/lde_{topk}_{tol}')
            if osp.exists(model_dir) and not force_train_one:
                LDE_model = torch.load(model_dir)
            else:
                LDE_model = run_one(s, feat_dir=feat_dir, top_k=topk, num_epoch=num_epoch, tol=tol,
                                    force_gen_data=force_gen_data, step=step)
                os.makedirs(osp.dirname(model_dir), exist_ok=True)
                torch.save(LDE_model, model_dir)
            set_seed(s)

            org_logits_path = feat_dir + f'/org_test_logits_{distri}'
            PDA_model = LDE_adapt(net, LDE_model, val_set, val_loader, org_logits_path)
            if os.path.exists(org_logits_path):
                org_logits = torch.load(feat_dir+f'/org_test_logits_{distri}', 'cpu')
            else:
                org_logits = None
            net.div_out = False
            sum_org_logits = org_logits.reshape((-1, 3, 1000)).sum(dim=1).cuda()
            use_logits = sum_org_logits + PDA_model.bsce_weight.log()

            top1, top5, *freq = validate(val_loader, net, None, name_dataset, gpu=0, print_freq=10, debug=False,
                                         mode='eval', num_per_cls_list=num_per_cls_list, use_logits=use_logits)
            top1_meter.update(top1)
            top5_meter.update(top5)
            head_meter.update(freq[0])
            med_meter.update(freq[1])
            tail_meter.update(freq[2])

            print(f"now status")
            print(f"top1 {top1} | top5 {top5}")
            print("-"*20)
            print(f"avg status")
            print(f"top1 {top1_meter.avg} | top5 {top5_meter.avg}")
            if distri == 'uniform':
                break

        result_str = f"Status: distribution: {distri} | epoch {num_epoch} \n" \
                     f"top1 {top1_meter.avg} | top5 {top5_meter.avg}"
        record_str += f"{top1_meter.avg:.2f}\t"
        result.append(result_str)

    print('_'*40)
    for r in result:
        print(r, end='\n\n')
    print(record_str)


if __name__ == '__main__':
    main()
