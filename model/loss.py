import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, config, task_num=0):
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.task_num = task_num
        self.criterion = []
        for a in range(0, self.task_num):
            try:
                ratio = config.getfloat("train", "loss_weight_%d" % a)
                self.criterion.append(
                    nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, ratio], dtype=np.float32)).cuda()))
                # print_info("Task %d with weight %.3lf" % (task, ratio))
            except Exception as e:
                self.criterion.append(nn.CrossEntropyLoss())

    def forward(self, outputs, labels):
        loss = 0
        for a in range(0, len(outputs[0])):
            o = outputs[:, a, :].view(outputs.size()[0], -1)
            loss += self.criterion[a](o, labels[:, a])

        return loss


def log_sum_exp(prediction, target, mask=None):
    """
    Expand softmax to multi-label classification

    :param prediction:
        Torch Float Tensor with shape of [batch_size * sequence_length * N]
            don't use sigmoid or softmax

    :param target:
        Torch Long Tensor with shape of [batch_size * sequence_length * N]
            one-hot representation for the label

    :param mask:
        Torch Long Tensor with shape of [batch_size * sequence_length * N]
            attention mask, mask out the padded token.
            (padded token should not be count as negative token)

    :return:
        log sum exp loss with shape of [batch_size * sequence_length]
    """
    if mask is None:
        mask = torch.ones_like(prediction).long()

    prediction_pos = prediction.masked_fill((1-target).bool(), 1e12)

    prediction_neg = prediction.masked_fill((target | (1-mask)).bool(), -1e12)

    zeros = torch.zeros_like(prediction[..., :1])

    prediction_pos = torch.cat((-prediction_pos, zeros), dim=-1)
    prediction_neg = torch.cat((prediction_neg, zeros), dim=-1)

    pos_loss = torch.logsumexp(prediction_pos, dim=-1)
    neg_loss = torch.logsumexp(prediction_neg, dim=-1)

    return (pos_loss + neg_loss).mean()


def multi_label_cross_entropy_loss(outputs, labels):
    labels = labels.float()
    temp = outputs
    res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
    res = torch.mean(torch.sum(res, dim=1))

    return res


def cross_entropy_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


def log_square_loss(outputs, labels):
    return torch.mean((torch.log(torch.clamp(outputs, 0, 450) + 1) - torch.log(torch.clamp(labels, 0, 450) + 1)) ** 2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
