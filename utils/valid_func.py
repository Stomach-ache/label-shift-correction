
import time
import torch
import transformers
import numpy as np

from .cli_utils import AverageMeter, accuracy, ProgressMeter

data_split_threshold = {'cifar-100': {
 'low': 69,
 'med': 35,
 'high': 0,
}}


def accuracy_each_cls(output, label, low=69, med=35, high=0):
    pred = torch.argmax(output.cpu(), dim=1).numpy()
    label = label.cpu().numpy()
    ind_h = label < med
    cnt_h = ind_h.sum()
    acc_h = (pred[ind_h] == label[ind_h]).sum() / cnt_h if cnt_h > 0 else 0

    ind_t = label > low
    cnt_t = ind_t.sum()
    acc_t = (pred[ind_t] == label[ind_t]).sum() / cnt_t if cnt_t > 0 else 0

    ind_m = np.logical_not(np.logical_or(ind_t, ind_h))
    cnt_m = ind_m.sum()
    acc_m = (pred[ind_m] == label[ind_m]).sum() / cnt_m if cnt_m > 0 else 0
    return acc_h * 100, acc_m * 100, acc_t * 100, cnt_h * 100, cnt_m * 100, cnt_t * 100


def valid_cifar100(val_loader, model, gpu, print_freq, debug=False, use_logits=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    head = AverageMeter('Head', ':6.2f')
    med = AverageMeter('med', ':6.2f')
    tail = AverageMeter('tail', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    if use_logits is None:
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                # compute output
                output = model(images)
                if isinstance(output, (tuple, list)):
                    # output = output[0]
                    output = sum(output)
                elif isinstance(output, transformers.modeling_outputs.ImageClassifierOutput):
                    output = output.logits
                # _, targets = output.max(1)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                acc_h, acc_m, acc_t, cnt_h, cnt_m, cnt_t = accuracy_each_cls(output, target)
                head.update(acc_h, cnt_h)
                med.update(acc_m, cnt_m)
                tail.update(acc_t, cnt_t)

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)
                if i > 10 and debug:
                    break

            print(f"Head are {head.avg.item()}")
            print(f"Med are {med.avg.item()}")
            print(f"Tail are {tail.avg.item()}")
    else:
        output = use_logits.cuda()
        target = torch.tensor(val_loader.dataset.targets).cuda()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc_h, acc_m, acc_t, cnt_h, cnt_m, cnt_t = accuracy_each_cls(output, target)

        head.update(acc_h, cnt_h)
        med.update(acc_m, cnt_m)
        tail.update(acc_t, cnt_t)

        top1.update(acc1[0], output.shape[0])
        top5.update(acc5[0], output.shape[0])
        print(f"Head are {head.avg.item()}")
        print(f"Med are {med.avg.item()}")
        print(f"Tail are {tail.avg.item()}")
    return top1.avg, top5.avg, head.avg, med.avg, tail.avg


def acc_ind(output, target, ind):
    pred = torch.argmax(output.cpu(), dim=1).numpy()
    label = target.cpu().numpy()
    idx = ind[label]  # ind 表示1000 个类别，有那几个属于head, tail 或则 med
    cnt = idx.sum()
    if idx.shape[0] == 1:
        idx = idx.tolist()
    acc = (pred[idx] == label[idx]).sum() / cnt
    return acc, cnt


def valid_imagenet(val_loader, model, gpu, print_freq, debug, num_per_cls_list, use_logits=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    head = AverageMeter('Head', ':6.2f')
    med = AverageMeter('med', ':6.2f')
    tail = AverageMeter('tail', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()
    head_ind = num_per_cls_list > 100
    tail_ind = num_per_cls_list < 20
    med_ind = torch.logical_not(torch.logical_or(head_ind, tail_ind))
    head_label_idx = torch.tensor([head_ind[i]for i in range(1000)])
    med_label_idx = torch.tensor([med_ind[i]for i in range(1000)])
    tail_label_idx = torch.tensor([tail_ind[i]for i in range(1000)])

    if use_logits is None:
        with torch.no_grad():
            end = time.time()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                # compute output
                output = model(images)
                if isinstance(output, (tuple, list)):
                    # output = output[0]
                    output = sum(output)
                elif isinstance(output, transformers.modeling_outputs.ImageClassifierOutput):
                    output = output.logits
                # _, targets = output.max(1)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                acc_h, cnt_h = acc_ind(output, target, head_label_idx)
                acc_m, cnt_m = acc_ind(output, target, med_label_idx)
                acc_t, cnt_t = acc_ind(output, target, tail_label_idx)

                head.update(acc_h, cnt_h)
                med.update(acc_m, cnt_m)
                tail.update(acc_t, cnt_t)

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)
                if i > 10 and debug:
                    break

            print(f"Head are {head.avg.item() * 100}")
            print(f"Med are {med.avg.item() * 100}")
            print(f"Tail are {tail.avg.item() * 100}")
    else:
        output = use_logits.cuda()
        target = torch.tensor(val_loader.dataset.labels).cuda()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc_h, cnt_h = acc_ind(output, target, head_label_idx)
        acc_m, cnt_m = acc_ind(output, target, med_label_idx)
        acc_t, cnt_t = acc_ind(output, target, tail_label_idx)

        head.update(acc_h, cnt_h)
        med.update(acc_m, cnt_m)
        tail.update(acc_t, cnt_t)

        top1.update(acc1[0], output.shape[0])
        top5.update(acc5[0], output.shape[0])
        print(f"Head are {head.avg.item() * 100:.1f}")
        print(f"Med are {med.avg.item() * 100:.1f}")
        print(f"Tail are {tail.avg.item() * 100:.1f}")

    return top1.avg, top5.avg, head.avg, med.avg, tail.avg


@torch.no_grad()
def valid_places365(val_loader, model, gpu, print_freq, debug, num_per_cls_list, use_logits=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    head = AverageMeter('Head', ':6.2f')
    med = AverageMeter('med', ':6.2f')
    tail = AverageMeter('tail', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()
    head_ind = num_per_cls_list > 100
    tail_ind = num_per_cls_list < 20
    med_ind = torch.logical_not(torch.logical_and(head_ind, tail_ind))
    head_label_idx = torch.tensor([head_ind[i]for i in range(365)])
    med_label_idx = torch.tensor([med_ind[i]for i in range(365)])
    tail_label_idx = torch.tensor([tail_ind[i]for i in range(365)])

    if use_logits is None:
        with torch.no_grad():
            end = time.time()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                # compute output
                output = model(images)
                if isinstance(output, (tuple, list)):
                    # output = output[0]
                    output = sum(output)
                elif isinstance(output, transformers.modeling_outputs.ImageClassifierOutput):
                    output = output.logits
                # _, targets = output.max(1)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                acc_h, cnt_h = acc_ind(output, target, head_label_idx)
                acc_m, cnt_m = acc_ind(output, target, med_label_idx)
                acc_t, cnt_t = acc_ind(output, target, tail_label_idx)

                head.update(acc_h, cnt_h)
                med.update(acc_m, cnt_m)
                tail.update(acc_t, cnt_t)

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)
                if i > 10 and debug:
                    break

            print(f"Head are {head.avg.item()}")
            print(f"Med are {med.avg.item()}")
            print(f"Tail are {tail.avg.item()}")
    else:
        output = use_logits.cuda()
        target = torch.tensor(val_loader.dataset.labels).cuda()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc_h, cnt_h = acc_ind(output, target, head_label_idx)
        acc_m, cnt_m = acc_ind(output, target, med_label_idx)
        acc_t, cnt_t = acc_ind(output, target, tail_label_idx)

        head.update(acc_h, cnt_h)
        med.update(acc_m, cnt_m)
        tail.update(acc_t, cnt_t)

        top1.update(acc1[0], output.shape[0])
        top5.update(acc5[0], output.shape[0])
        print(f"Head are {head.avg.item()}")
        print(f"Med are {med.avg.item()}")
        print(f"Tail are {tail.avg.item()}")

    return top1.avg, top5.avg, head.avg, med.avg, tail.avg



@torch.no_grad()
def valid_places365(val_loader, model, gpu, print_freq, debug, num_per_cls_list, use_logits=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    head = AverageMeter('Head', ':6.2f')
    med = AverageMeter('med', ':6.2f')
    tail = AverageMeter('tail', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()
    head_ind = num_per_cls_list > 100
    tail_ind = num_per_cls_list < 20
    med_ind = torch.logical_not(torch.logical_and(head_ind, tail_ind))
    head_label_idx = torch.tensor([head_ind[i]for i in range(365)])
    med_label_idx = torch.tensor([med_ind[i]for i in range(365)])
    tail_label_idx = torch.tensor([tail_ind[i]for i in range(365)])

    if use_logits is None:
        with torch.no_grad():
            end = time.time()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                # compute output
                output = model(images)
                if isinstance(output, (tuple, list)):
                    # output = output[0]
                    output = sum(output)
                elif isinstance(output, transformers.modeling_outputs.ImageClassifierOutput):
                    output = output.logits
                # _, targets = output.max(1)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                acc_h, cnt_h = acc_ind(output, target, head_label_idx)
                acc_m, cnt_m = acc_ind(output, target, med_label_idx)
                acc_t, cnt_t = acc_ind(output, target, tail_label_idx)

                head.update(acc_h, cnt_h)
                med.update(acc_m, cnt_m)
                tail.update(acc_t, cnt_t)

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)
                if i > 10 and debug:
                    break

            print(f"Head are {head.avg.item()}")
            print(f"Med are {med.avg.item()}")
            print(f"Tail are {tail.avg.item()}")
    else:
        output = use_logits.cuda()
        target = torch.tensor(val_loader.dataset.labels).cuda()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc_h, cnt_h = acc_ind(output, target, head_label_idx)
        acc_m, cnt_m = acc_ind(output, target, med_label_idx)
        acc_t, cnt_t = acc_ind(output, target, tail_label_idx)

        head.update(acc_h, cnt_h)
        med.update(acc_m, cnt_m)
        tail.update(acc_t, cnt_t)

        top1.update(acc1[0], output.shape[0])
        top5.update(acc5[0], output.shape[0])
        print(f"Head are {head.avg.item()}")
        print(f"Med are {med.avg.item()}")
        print(f"Tail are {tail.avg.item()}")

    return top1.avg, top5.avg, head.avg, med.avg, tail.avg

@torch.no_grad()
def valid_inat(val_loader, model, gpu, print_freq, debug, num_per_cls_list, use_logits=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    head = AverageMeter('Head', ':6.2f')
    med = AverageMeter('med', ':6.2f')
    tail = AverageMeter('tail', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()
    head_ind = num_per_cls_list > 100
    tail_ind = num_per_cls_list < 20
    med_ind = torch.logical_not(torch.logical_and(head_ind, tail_ind))
    head_label_idx = torch.tensor([head_ind[i]for i in range(8142)])
    med_label_idx = torch.tensor([med_ind[i]for i in range(8142)])
    tail_label_idx = torch.tensor([tail_ind[i]for i in range(8142)])

    if use_logits is None:
        with torch.no_grad():
            end = time.time()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                # compute output
                output = model(images)
                if isinstance(output, (tuple, list)):
                    # output = output[0]
                    output = sum(output)
                elif isinstance(output, transformers.modeling_outputs.ImageClassifierOutput):
                    output = output.logits
                # _, targets = output.max(1)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                acc_h, cnt_h = acc_ind(output, target, head_label_idx)
                acc_m, cnt_m = acc_ind(output, target, med_label_idx)
                acc_t, cnt_t = acc_ind(output, target, tail_label_idx)

                head.update(acc_h, cnt_h)
                med.update(acc_m, cnt_m)
                tail.update(acc_t, cnt_t)

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)
                if i > 10 and debug:
                    break

            print(f"Head are {head.avg.item()}")
            print(f"Med are {med.avg.item()}")
            print(f"Tail are {tail.avg.item()}")
    else:
        output = use_logits.cuda()
        target = torch.tensor(val_loader.dataset.labels).cuda()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc_h, cnt_h = acc_ind(output, target, head_label_idx)
        acc_m, cnt_m = acc_ind(output, target, med_label_idx)
        acc_t, cnt_t = acc_ind(output, target, tail_label_idx)

        head.update(acc_h, cnt_h)
        med.update(acc_m, cnt_m)
        tail.update(acc_t, cnt_t)

        top1.update(acc1[0], output.shape[0])
        top5.update(acc5[0], output.shape[0])
        print(f"Head are {head.avg.item()}")
        print(f"Med are {med.avg.item()}")
        print(f"Tail are {tail.avg.item()}")

    return top1.avg, top5.avg, head.avg, med.avg, tail.avg

def validate(val_loader, model, criterion, dataset, gpu, print_freq, debug, mode='eval', num_per_cls_list=None, use_logits=None):
    if 'cifar-100' in dataset.lower():
        return valid_cifar100(val_loader, model, gpu, print_freq, debug, use_logits)

    if 'imagenet' in dataset.lower():
        return valid_imagenet(val_loader, model, gpu, print_freq, debug, num_per_cls_list, use_logits)

    if 'places' in dataset.lower():
        return valid_places365(val_loader, model, gpu, print_freq, debug, num_per_cls_list, use_logits)

    if 'inat' in dataset.lower():
        return valid_inat(val_loader, model, gpu, print_freq, debug, num_per_cls_list, use_logits)

    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            # compute output
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, transformers.modeling_outputs.ImageClassifierOutput):
                output = output.logits
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)
            if i > 10 and debug:
                break
    return top1.avg, top5.avg

# pseudo_label_distribution_discrepancy
def pse_label_distri_dist(y_pl, y_gt):
    y_pl_n = y_pl / y_pl.sum()
    y_gt_n = y_gt / y_gt.sum()
    return ((y_pl_n - y_gt_n).abs() / y_gt_n).sum()



