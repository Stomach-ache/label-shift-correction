
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def get_logits_w_bsce(inputs, bsce_weight, temper, update=False):
    # BSCE
    logits = inputs + (bsce_weight.unsqueeze(0).expand(inputs.shape[0], -1).log()) / temper
    # logits = inputs
    return logits

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


class PDA(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, class_num=100):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.bsce_weight = torch.ones(class_num).cuda()
        self.temper = 3.

        self.tot_probs = torch.zeros(100).cuda()
        self.sample_cnt = torch.zeros(1).cuda()

    def update_bsce_weight(self, weight):
        pass

    def forward(self, x, update_weight=False, consistency_reg=False):
        if self.episodic:
            self.reset()
        self.eval()
        b_logits=None
        for _ in range(self.steps):
            logits = self.model(x)
            if isinstance(logits, (list, tuple)):
                b_logits = [get_logits_w_bsce(l, self.bsce_weight, self.temper) for l in logits]
                final_b_logits = sum(b_logits) / len(b_logits)
                final_logits = sum(logits) / len(logits)
            else:
                b_logits = get_logits_w_bsce(logits, self.bsce_weight, self.temper)
                final_logits = logits
                # final_logits = b_logits

            if consistency_reg and isinstance(logits, (list, tuple)):
                self.optimizer.zero_grad()
                loss_con = NBOD(b_logits[1:], 1.0)
                loss_con.backward()
                self.optimizer.step()

            if update_weight:
                p_0 = .0
                probs = F.softmax(final_logits, 1).detach()
                conf_index = torch.max(probs, dim=1).values > p_0
                self.tot_probs += probs[conf_index, ...].sum(dim=0)
                self.sample_cnt += conf_index.sum()
                self.bsce_weight = self.tot_probs
        return b_logits

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    @staticmethod
    def collect_params(model):
        """Collect all trainable parameters.

        Walk the model's modules and collect all parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model.eval()
        # disable grad, to (re-)enable only what tent updates
        model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)
        return model


def jsd(p, q, log_target=True):
    div = (F.kl_div(p, q, reduction='batchmean', log_target=log_target) +
           F.kl_div(q, p, reduction='batchmean', log_target=log_target)) / 2
    return div


def rmt_loss(logits):
    mean_logits = 0
    mean_logits_log = 0
    log_prob = []
    for l in logits:
        p = F.softmax(l, dim=1)
        mean_logits = p
        log_prob.append(p.log())
    with torch.no_grad():
        mean_logits_log = (mean_logits / len(logits)).log()
    loss = 0
    for lp in log_prob:
        loss_item = jsd(lp, mean_logits_log)
        loss += loss_item

    return loss






