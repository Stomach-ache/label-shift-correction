import torch
import torch.nn as nn
from torch.nn import functional as F


def NBOD(inputs, factor):
    classifier_num = len(inputs)
    if classifier_num == 1:
        return 0
    logits_softmax = []
    logits_logsoftmax = []
    for i in range(classifier_num):
        logits_softmax.append(F.softmax(inputs[i], dim=1))
        logits_logsoftmax.append(torch.log(logits_softmax[i] + 1e-9))

    loss_mutual = 0
    for i in range(classifier_num):
        for j in range(classifier_num):
            if i == j:
                continue
            loss_mutual += factor * F.kl_div(logits_logsoftmax[i], logits_softmax[j], reduction='batchmean')
    loss_mutual /= (classifier_num - 1)
    return loss_mutual

def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


class NIL_NBOD(nn.Module):
    def __init__(self, para_dict=None):
        super(NIL_NBOD, self).__init__()
        self.para_dict = para_dict
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']
        self.bsce_weight = torch.FloatTensor(self.num_class_list).to(self.device)

        self.multi_classifier_diversity_factor = self.para_dict['cfg'].LOSS.MULTI_CLASIIFIER_LOSS.DIVERSITY_FACTOR
        self.multi_classifier_diversity_factor_hcm = self.para_dict[
            'cfg'].LOSS.MULTI_CLASIIFIER_LOSS.DIVERSITY_FACTOR_HCM
        self.hcm_N = self.para_dict['cfg'].LOSS.HCM_N
        self.hcm_ratio = self.para_dict['cfg'].LOSS.HCM_RATIO
        self.ce_ratio = self.para_dict['cfg'].LOSS.CE_RATIO
        self.forward_func_name = self.para_dict['cfg'].LOSS.forward
        if self.forward_func_name == 'forward':
            self.forward_func_name = 'forward_org'
        self.forward_func = getattr(self, self.forward_func_name)
        self.has_set_ride = False
        self.bsce_temper = self.para_dict['cfg'].LOSS.BSCE_TEMPER

    def set_ride(self):
        import numpy as np
        self.reweight_factor = 0.05
        cls_num_list = np.array(self.num_class_list) / np.sum(self.num_class_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * self.reweight_factor + 1 - self.reweight_factor

        per_cls_weights = per_cls_weights / np.max(per_cls_weights)
        self.diversity_temperature = torch.FloatTensor(per_cls_weights).view((1, -1)).cuda()
        self.temperature_mean = self.diversity_temperature.mean().item()
        self.has_set_ride = True

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        loss = self.forward_func(inputs, targets, **kwargs)
        return loss

    def forward_wo_hcm_temp_div(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        if self.has_set_ride is False:
            self.set_ride()
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0
        los_div = 0

        output_logits = sum(inputs) / len(inputs)

        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            los_ce += F.cross_entropy(logits, targets[0])

            output_dist = F.log_softmax(logits / self.diversity_temperature, dim=1)
            with torch.no_grad():
                mean_output_dist = F.softmax(output_logits / self.diversity_temperature, dim=1)
            los_div += self.temperature_mean * self.temperature_mean * F.kl_div(
                output_dist,
                mean_output_dist,
                reduction='batchmean')

        loss += los_ce * self.ce_ratio
        loss += los_div * self.multi_classifier_diversity_factor
        return loss

    def naked_ce(self, inputs, targets, **kwargs):
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0

        for i in range(classifier_num):
            logits = inputs[i]

            los_ce += F.cross_entropy(logits, targets[0])

        loss += los_ce * self.ce_ratio
        return loss

    def forward_wo_hcm_w_unbalance_div(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0

        inputs_balance = []
        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)

            los_ce += F.cross_entropy(logits, targets[0])

        # loss += NBOD(inputs_balance, factor=self.multi_classifier_diversity_factor)
        loss += NBOD(inputs, factor=self.multi_classifier_diversity_factor)
        loss += los_ce * self.ce_ratio
        return loss

    def forward_wo_hcm_div(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0

        inputs_balance = []
        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)

            los_ce += F.cross_entropy(logits, targets[0])

        # loss += NBOD(inputs_balance, factor=self.multi_classifier_diversity_factor)
        loss += los_ce * self.ce_ratio
        return loss


    def forward_wo_hcm_ce(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0
        if 'div_factor' in kwargs:
            div_factor = kwargs['div_factor']
        else:
            div_factor = self.multi_classifier_diversity_factor
        if len(targets) == 1:
            targets = targets * 3

        inputs_balance = []
        for i in range(classifier_num):
            logits = inputs[i]
            inputs_balance.append(logits)

            los_ce += F.cross_entropy(logits, targets[i])

        loss += NBOD(inputs_balance, factor=div_factor)
        loss += los_ce * self.ce_ratio
        return loss

    def forward_wo_hcm(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0
        if 'div_factor' in kwargs:
            div_factor = kwargs['div_factor']
        else:
            div_factor = self.multi_classifier_diversity_factor
        if len(targets) == 1:
            targets = targets * classifier_num

        inputs_balance = []
        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log() * self.bsce_temper
            inputs_balance.append(logits)

            los_ce += F.cross_entropy(logits, targets[i])

        loss += NBOD(inputs_balance, factor=div_factor)
        loss += los_ce * self.ce_ratio
        return loss

    def forward_wo_hcm_temp(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0
        temper = 1.2

        inputs_balance = []
        for i in range(classifier_num):
            logits = (inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log())
            los_ce += F.cross_entropy(logits, targets[0])
            logits = logits / temper
            inputs_balance.append(logits)


        loss += NBOD(inputs_balance, factor=self.multi_classifier_diversity_factor)
        loss += los_ce * self.ce_ratio
        return loss

    def forward_half_bal_div(self, inputs, targets, **kwargs):
        """
        idle
        """
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0

        inputs_balance = []
        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)

            los_ce += F.cross_entropy(logits, targets[0])

        loss += (NBOD(inputs_balance, factor=self.multi_classifier_diversity_factor) + NBOD(inputs, factor=self.multi_classifier_diversity_factor)) / 2.0
        loss += los_ce * self.ce_ratio
        return loss

    def forward_unbal_div(self, inputs, targets, **kwargs):
        """
        idle
        """
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0

        inputs_balance = []
        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)

            los_ce += F.cross_entropy(logits, targets[0])

        loss += NBOD(inputs, factor=self.multi_classifier_diversity_factor)
        loss += los_ce * self.ce_ratio
        return loss

    def forward_org(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_N]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets[0])

        loss += NBOD(inputs_balance, factor=self.multi_classifier_diversity_factor)
        loss += NBOD(inputs_HCM_balance, factor=self.multi_classifier_diversity_factor_hcm)
        loss += los_ce * self.ce_ratio + loss_HCM * self.hcm_ratio
        return loss

    def forward_wo_hcm_neg_bsce(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss = 0
        los_ce = 0

        inputs_balance = []
        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            los_ce += F.cross_entropy(logits, targets[0])
            logits = inputs[i] - self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)

        loss += NBOD(inputs_balance, factor=self.multi_classifier_diversity_factor)
        loss += los_ce * self.ce_ratio
        return loss

    def update(self, epoch):
        """
        Args:
           code can be added for progressive loss.
        """
        pass


if __name__ == '__main__':
    pass
