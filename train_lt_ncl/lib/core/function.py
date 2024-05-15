
from core.evaluate import accuracy, AverageMeter
import torch
import time

from dataset import test_CIFAR100
import random

def multi_networks_train_model(
        trainLoader, model, epoch, epoch_number, optimizer, combiner, criterion, cfg, logger, rank=0, **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)
    trainLoader.dataset.update(epoch)
    combiner.update(epoch)
    criterion.update(epoch)

    if cfg.DATASET.use_cuda and epoch % cfg.DATASET.CUDA.update_epoch == 0:
        n_samples_per_class = kwargs['num_class_list']
        curr_state, label = update_score_base(trainLoader, model, n_samples_per_class,
            posthoc_la=False, num_test=cfg.DATASET.CUDA.num_test, accept_rate=cfg.DATASET.CUDA.accept_rate)
    
    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):

        if 'AUGMIX' in cfg.DATASET.DATASET:
            image_list = [image, *meta['augmix']]
        else:
            image_list = [image] * network_num
        label_list = [label] * network_num
        meta_list = [meta] * network_num

        cnt = label_list[0].shape[0]

        optimizer.zero_grad()

        loss, now_acc = combiner.forward(model, criterion, image_list, label_list, meta_list, now_epoch=epoch,
                                         train=True, cfg=cfg, iteration=i, log=logger,
                                         class_list=criterion.num_class_list)


        if cfg.NETWORK.MOCO:
            alpha = cfg.NETWORK.MA_MODEL_ALPHA
            for net_id in range(network_num):
                net = ['backbone', 'module']
                for name in net:
                    for ema_param, param in zip(eval('model.module.' + name + '_MA').parameters(),
                                                eval('model.module.' + name).parameters()):
                        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


        loss.backward()
        optimizer.step()
        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0 and rank == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc.avg, all_loss.avg

def multi_network_valid_model_final(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, rank, **kwargs
):
    model.eval()
    network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)
    cnt_all = 0
    every_network_result = [0 for _ in range(network_num)]

    with torch.no_grad():
        all_loss = AverageMeter()
        acc_avg = AverageMeter()

        for i, (image, label, meta) in enumerate(dataLoader):

            image, label = image.to(device), label.to(device)
            image_list = [image for i in range(network_num)]

            if cfg.NETWORK.MOCO:
                feature = model((image_list,image_list), label=label, feature_flag=True)
                output_ce, output, output_MA = model(feature, classifier_flag=True)
            else:
                feature = model(image_list, label=label, feature_flag=True)
                output_ce = model(feature, classifier_flag=True)

            loss = criterion(output_ce, (label,))

            for j, logit in enumerate(output_ce):
                every_network_result[j] += torch.sum(torch.argmax(logit, dim=1).cpu() == label.cpu())

            average_result = torch.mean(torch.stack(output_ce), dim=0)
            now_result = torch.argmax(average_result, 1)

            acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            cnt_all += cnt
            all_loss.update(loss.data.item(), cnt)
            acc_avg.update(acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_ensemble_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc_avg.avg * 100
        )
        if rank == 0:
            for i, result in enumerate(every_network_result):
                logger.info("network {} Valid_single_Acc: {:>5.2f}%".format(i, float(result) / cnt_all * 100))
            logger.info(pbar_str)
        best_single_acc = max(every_network_result) / cnt_all
    return acc_avg.avg, all_loss.avg, best_single_acc



def update_score_base(loader, model, n_samples_per_class, posthoc_la, num_test, accept_rate):
    model.eval()

    if posthoc_la:
        dist = torch.tensor(n_samples_per_class)
        prob = dist / dist.sum()


    with torch.no_grad():
        n = num_test
        pos, state = [], []
        for cidx in range(len(n_samples_per_class)):
            class_pos = torch.where(torch.tensor(loader.dataset.targets) == cidx)[0]
            max_state = loader.dataset.curr_state[class_pos[0]].int()
            for s in range(max_state + 1):
                _pos = random.choices(class_pos.tolist(), k=n * (s + 1))
                pos += _pos
                state += [s] * len(_pos)

        tmp_dataset = test_CIFAR100(pos, state, loader.dataset)
        tmp_loader = torch.utils.data.DataLoader(tmp_dataset, batch_size=128, shuffle=False, num_workers=8)

        for batch_idx, data_tuple in enumerate(tmp_loader):
            data = data_tuple[0].cuda()
            label = data_tuple[1]
            idx = data_tuple[2]
            state = data_tuple[3]

            logit = model(data, output_type=None)

            if isinstance(logit, list):
                logits = logit[1:]  # remove first original image
                logit = (sum(logits) / len(logits)).cpu()
            else:
                logit = logit.cpu()

            if posthoc_la:
                logit = logit.cpu() - torch.log(prob.view(1, -1).expand(logit.shape[0], -1))

            correct = (logit.max(dim=1)[1] == label).int().detach().cpu()
            loader.dataset.update_scores(correct, idx, state)

    # loader.dataset.update()
    correct_sum_per_class = torch.zeros(len(n_samples_per_class))
    trial_sum_per_class = torch.zeros(len(n_samples_per_class))
    for cidx in range(len(n_samples_per_class)):
        class_pos = torch.where(torch.tensor(loader.dataset.targets) == cidx)[0]

        correct_sum_row = torch.sum(loader.dataset.score_tmp[class_pos], dim=0)
        trial_sum_row = torch.sum(loader.dataset.num_test[class_pos], dim=0)

        ratio = correct_sum_row / trial_sum_row
        idx = loader.dataset.curr_state[class_pos][0].int() + 1
        condition = torch.sum((ratio[:idx] > accept_rate)) == idx

        # if correct_sum == trial_sum:
        # if float(correct_sum) >= float(trial_sum * 0.6):
        if condition:
            loader.dataset.curr_state[class_pos] += 1
        else:
            loader.dataset.curr_state[class_pos] -= 1

    loader.dataset.curr_state = loader.dataset.curr_state.clamp(loader.dataset.min_state, loader.dataset.max_state - 1)
    loader.dataset.score_tmp *= 0
    loader.dataset.num_test *= 0

    # print(f'Max correct: {int(torch.max(correct_sum_per_class))} Max trial: {int(torch.max(trial_sum_per_class))}')

    # loader.dataset.update()
    model.train()

    # Debug
    curr_state = loader.dataset.curr_state
    label = loader.dataset.targets
    print(f'Max state: {int(torch.max(curr_state))} // Min state: {int(torch.min(curr_state))}')

    return curr_state, label
