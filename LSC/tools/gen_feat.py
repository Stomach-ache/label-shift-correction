import torch
from dataset.sade_data_loader.cifar_data_loaders import ImbalanceCIFAR100DataLoader
from dataset.sade_ds import ImageNetLTDataLoader
import os
import copy


def gen_feat_cifar100(model, save_dir, data_dir='../data/cifar-100-python'):
    model.eval()
    network_num = 3
    # every_network_predict = [[] for _ in range(network_num)]
    every_network_logits = [[] for _ in range(network_num)]
    every_network_feature = [[] for _ in range(network_num)]

    train_loader = ImbalanceCIFAR100DataLoader(data_dir, batch_size=128, shuffle=True, num_workers=16, training=True)
    average_predict = []
    all_label = []
    with torch.no_grad():
        for i, (image, label) in enumerate(train_loader):
            all_label.append(label.cpu())
            # label = label.cuda()
            image = image.cuda()
            output_ce = model(image)
            output_ce = [output_ce.cpu() for _ in range(3)]
            sum_result = copy.deepcopy(sum(output_ce) / len(output_ce))
            for j in range(network_num):
                if j > 0:
                    sum_result += output_ce[j]
            average_predict.append(sum_result.argmax(dim=1).cpu())

            for every_list, logit in zip(every_network_logits, output_ce):
                every_list.append(logit)
                # every_network_predict[j].append(torch.argmax(logit, dim=1).cpu())

    # 3xNxC
    all_logits = torch.stack([torch.cat(l)for l in every_network_logits])
    # all_feat = torch.stack([torch.cat(f)for f in every_network_feature])
    all_label= torch.cat(all_label)
    # all_label = torch.argmax(all_label_onehot, dim=1)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(all_logits, os.path.join(save_dir, 'logits'))
    # torch.save(all_feat, os.path.join(save_dir, 'feat'))
    torch.save(all_label, os.path.join(save_dir, 'label'))
    print(f'File saved in {save_dir}')
    return


def gen_feat_imagenet(model, save_dir, data_dir='../data/ImageNet'):
    import copy
    model.eval()
    network_num = 3
    # every_network_predict = [[] for _ in range(network_num)]
    every_network_logits = [[] for _ in range(network_num)]
    every_network_feature = [[] for _ in range(network_num)]

    train_loader = ImageNetLTDataLoader(data_dir, batch_size=128, shuffle=True, num_workers=16, training=True)
    average_predict = []
    all_label = []
    with torch.no_grad():
        for i, (image, label) in enumerate(train_loader):
            all_label.append(label.cpu())
            # label = label.cuda()
            image = image.cuda()
            output_ce = model(image)
            output_ce = [output_ce.detach().cpu() for _ in range(3)]
            # sum_result = copy.deepcopy(output_ce[0])
            # for j in range(network_num):
            #     if j > 0:
            #         sum_result += output_ce[j]
            # average_predict.append(sum_result.argmax(dim=1).cpu())

            for every_list, logit in zip(every_network_logits, output_ce):
                every_list.append(logit)
                # every_network_predict[j].append(torch.argmax(logit, dim=1).cpu())

    # 3xNxC
    all_logits = torch.stack([torch.cat(l)for l in every_network_logits])
    # all_feat = torch.stack([torch.cat(f)for f in every_network_feature])
    all_label = torch.cat(all_label)
    # all_label = torch.argmax(all_label_onehot, dim=1)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(all_logits, os.path.join(save_dir, 'logits'))
    # torch.save(all_feat, os.path.join(save_dir, 'feat'))
    torch.save(all_label, os.path.join(save_dir, 'label'))
    print(f'File saved in {save_dir}')
    return


if __name__ == '__main__':
    pass