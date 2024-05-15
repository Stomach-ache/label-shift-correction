import time

import os
import sys
import logging
import random

import numpy as np
import torch

import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def mean(items):
    return sum(items)/len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger
    

def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_kendall_tau(a, b):
    '''
    Kendall Tau is a metric to measure the ordinal association between two measured quantities.
    Refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    '''
    assert len(a) == len(b), "Sequence a and b should have the same length while computing kendall tau."
    length = len(a)
    count = 0
    total = 0
    for i in range(length-1):
        for j in range(i+1, length):
            count += _sign(a[i] - a[j]) * _sign(b[i] - b[j])
            total += 1
    Ktau = count / total
    return Ktau


def infer(buffer, ranker):
    matrix0 = []
    ops0 = []
    matrix1 = []
    ops1 = []
    for ((matrix0_, ops0_), (matrix1_, ops1_)) in buffer:
        matrix0.append(matrix0_)
        ops0.append(ops0_)

        matrix1.append(matrix1_)
        ops1.append(ops1_)

    matrix0 = torch.stack(matrix0, dim=0)
    ops0 = torch.stack(ops0, dim=0)

    matrix1 = torch.stack(matrix1, dim=0)
    ops1 = torch.stack(ops1, dim=0)

    with torch.no_grad():
        outputs = ranker((matrix0, ops0), (matrix1, ops1))

    return _sign((outputs-0.5).cpu().tolist())


def select(items, index):
    return [items[i] for i in index]


def list_select(items, index):
    return [item[index] for item in items]


def transpose_l(items):
    return list(map(list, zip(*items)))


def index_generate(m, n, up_triangular=False, max_batch_size=1024):
    if up_triangular:
        indexs = []
        for i in range(m-1):
            for j in range(i+1, n):
                indexs.append((i, j))
                if len(indexs) == max_batch_size:
                    yield indexs
                    indexs = []
        if indexs:
            yield indexs
    else:
        indexs = []
        for i in range(m):
            for j in range(n):
                indexs.append((i, j))
                if len(indexs) == max_batch_size:
                    yield indexs
                    indexs = []
        if indexs:
            yield indexs


def batchify(items):
    if isinstance(items[0], (list, tuple)):
        transposed_items = transpose_l(items)
        return [torch.stack(item, dim=0) for item in transposed_items]
    else:
        return torch.stack(items, dim=0) 
    


def cartesian_traverse(arch0, arch1, ranker, up_triangular=False):
    m, n = len(arch0), len(arch1)
    outputs = []
    with torch.no_grad():
        for index in index_generate(m, n, up_triangular):
            i, j = transpose_l(index)
            a = batchify(select(arch0, i))
            b = batchify(select(arch1, j))
            output = ranker(a, b)
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    if up_triangular:
        return outputs
    else:
        return outputs.view(m, n)


def compute_kendall_tau_AR(ranker, archs, performances):
    '''
    Kendall Tau is a metric to measure the ordinal association between two measured quantities.
    Refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    '''
    # assert len(matrix) == len(ops) == len(performances), "Sequence a and b should have the same length while computing kendall tau."
    length = len(performances)
    count = 0
    total = 0

    archs = transpose_l([torch.unbind(item) for item in archs])
    outputs = cartesian_traverse(archs, archs, ranker, up_triangular=True)

    p_combination = _sign((outputs-0.5).cpu().tolist())

    for i in range(length-1):
        for j in range(i+1, length):
            count += p_combination[total] * _sign(performances[i]-performances[j])
            total += 1

    assert len(p_combination) == total
    Ktau = count / total
    return Ktau


def concat(a, b):
    return [torch.cat([item0, item1]) for item0, item1 in zip(a, b)]


def list_concat(a, b):
    if len(a) == 3:
        a0, a1, a2 = a
        b0, b1, b2 = b
        rev0 = concat(a0, b0)
        rev1 = concat(a1, b1)
        rev2 = torch.cat([a2, b2], dim=0)
        return rev0, rev1, rev2
    else:
        a0, a1 = a
        b0, b1 = b
        rev0 = concat(a0, b0)
        rev1 = concat(a1, b1)
        return rev0, rev1

def compute_flops(module: nn.Module, size, skip_pattern, device):
    # print(module._auxiliary)
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)
    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("init hool for", name)
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        # print(f"training={training}")
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops

def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param




def plot_distribution_diff(y_gt, infer_distri):
    # 将pytorch tensor转换为numpy数组
    y_gt_np = y_gt.cpu().numpy().ravel()
    infer_distri_np = infer_distri.detach().cpu().numpy().ravel()

    # 计算直方图或密度估计
    bins = np.linspace(np.min(np.concatenate([y_gt_np, infer_distri_np])), np.max(np.concatenate([y_gt_np, infer_distri_np])), 50)
    y_gt_hist, _ = np.histogram(y_gt_np, bins=bins, density=True)
    infer_distri_hist, _ = np.histogram(infer_distri_np, bins=bins, density=True)

    # 绘制直方图或密度估计
    plt.figure()
    plt.plot(bins[:-1], y_gt_hist, label='y_gt')
    plt.plot(bins[:-1], infer_distri_hist, label='infer_distri')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.show()


def plot_histogram(y_gt, infer_distri):
    # 将pytorch tensor转换为numpy数组
    y_gt_np = y_gt.cpu().numpy().ravel()
    infer_distri_np = infer_distri.detach().cpu().numpy().ravel()

    # 绘制直方图
    plt.figure()
    plt.hist(y_gt_np, bins=len(y_gt_np), alpha=0.5, label='y_gt')
    plt.hist(infer_distri_np, bins=len(infer_distri_np), alpha=0.5, label='infer_distri')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison')
    plt.show()

def plot_bar_chart(y_gt, infer_distri, title='Bar Chart Comparison', save_path=None):
    # 将pytorch tensor转换为numpy数组
    y_gt_np = y_gt.cpu().numpy().ravel()
    infer_distri_np = infer_distri.detach().cpu().numpy().ravel()

    # 创建x轴的下标
    x = np.arange(len(y_gt_np))

    # 绘制柱状图
    plt.figure()
    plt.bar(x, y_gt_np, alpha=0.5, label='y_gt')
    plt.bar(x, infer_distri_np, alpha=0.5, label='infer_distri')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    if not save_path is None:
        plt.savefig(save_path)
    plt.show()

def plot_bar_chart_onehot(y_gt, overall_logits_tensor, title='Prediction Distribution', save_path=None):
    # 将pytorch tensor转换为numpy数组
    y_gt_np = y_gt.cpu().numpy().ravel()
    overall_logits_np = overall_logits_tensor.detach().cpu().numpy()
    all_labels = y_gt_np.shape[0]

    # 计算最终预测标签
    predicted_labels = np.argmax(overall_logits_np, axis=1)

    # 统计预测标签和y_gt的数量分布
    unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
    rped_label_ditri = np.zeros(all_labels)
    rped_label_ditri[unique_labels] = label_counts
    rped_label_ditri = rped_label_ditri / np.sum(rped_label_ditri)
    # 创建x轴的下标
    x = np.arange(all_labels)

    # 绘制柱状图
    plt.figure()
    plt.bar(x, y_gt_np, alpha=0.5, label='y_gt')
    plt.bar(x, rped_label_ditri, alpha=0.5, label='predicted')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    if not save_path is None:
        plt.savefig(save_path)
    plt.show()


def plot_heatmap(matrix, title='Heatmap', cmap='hot'):
    """
    绘制矩阵的热力图

    参数：
    matrix: 要绘制热力图的矩阵
    title: 图表标题
    cmap: 颜色映射名称
    """
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()


def get_num_cls_list_from_label(label):
    if isinstance(label, list):
        label = torch.tensor(label).int()
    label_set,  cnt = torch.unique(label, return_counts=True)
    num_cls_list = cnt.tolist()
    return num_cls_list


class time_tester:
    def __init__(self):
        self.chk_pt = None
        self.notion = ''

    def set(self, notion='', reset=False):
        if self.chk_pt is not None and not reset:
            duration = time.time() - self.chk_pt
            print(f"[notion] {self.notion}: {duration}s")
        self.chk_pt = time.time()
        self.notion = notion