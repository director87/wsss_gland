import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'sce':
            return self.SCELoss
        elif mode == 'ns':
            return self.ns
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target, gama):
        # print(logit.size())
        n, c, h, w = logit.size()
        # n, c = logit.size()
        # print(n, c, h, w)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())
        # print(loss)

        if self.batch_average:
            loss /= n
        # print(loss.shape)
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        # print(logit)
        # n, h, w = logit.size()
        n, c, h, w = logit.size()
        # loss = F.cross_entropy(logit, target.long(), weight=self.weight, ignore_index=self.ignore_index)
        # print(loss.shape[0])
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        # logpt = -criterion(logit, target)
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def SCELoss(self, logit, target, alpha=0.2, beta=0.8, num_classes=4):
        # print(logit)
        n, c, h, w = logit.size()
        # print(n, c, h, w)
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        ce = criterion(logit, target.long())
        logit = F.softmax(logit, dim=1)
        logit = torch.clamp(logit, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(target.long(), num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        label_one_hot = label_one_hot.view(n, num_classes, 224, 224)
        # print(logit.shape)
        # print(label_one_hot.shape)
        rce = (-1 * torch.sum(logit * torch.log(label_one_hot), dim=1))
        loss = alpha * ce + beta * rce.mean()

        if self.batch_average:
            loss /= n

        return loss

    def weight_loss(self, loss):
        n = loss.shape[0]
        loss = loss.view(n, -1)
        loss_weight = F.softmax(loss.clone().detach(), dim=1) / torch.mean(
            F.softmax(loss.clone().detach(), dim=1), dim=1, keepdim=True
        )
        loss = torch.sum(loss * loss_weight) / (n * loss.shape[1])
        return loss

    def reduce_loss(self, loss, reduction):
        """Reduce loss as specified.

        Args:
            loss (Tensor): Elementwise loss tensor.
            reduction (str): Options are "none", "mean" and "sum".

        Return:
            Tensor: Reduced loss tensor.
        """
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()

    def weight_reduce_loss(self, loss, weight=None, reduction='mean', avg_factor=None):
        """Apply element-wise weight and reduce loss.

        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Avarage factor when computing the mean of losses.

        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            assert weight.dim() == loss.dim()
            if weight.dim() > 1:
                assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            loss = self.reduce_loss(loss, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                loss = loss.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def ns(self, pred, label, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=255):
        loss = F.cross_entropy(pred, label.long(), weight=class_weight, reduction='none', ignore_index=ignore_index)
        # print(loss.mean())

        # if loss.mean() <= 1.0:

        weight = torch.ones_like(loss)
        metric = -loss.detach().reshape((loss.shape[0], loss.shape[1] * loss.shape[2]))
        weight = F.softmax(metric, 1)  # sm(-L)
        # print(torch.max(weight), torch.min(weight), torch.mean(weight))
        # print(torch.max(weight, dim=1, keepdim=False)[0])
        weight = weight / (torch.max(weight, dim=1, keepdim=False)[0] - torch.min(weight, dim=1, keepdim=False)[0]).reshape(-1, 1)
        # weight = weight / weight.mean(1).reshape((-1, 1))  # sm(-L)/mean(sm(-L))
        weight = weight.reshape((loss.shape[0], loss.shape[1], loss.shape[2]))
        sm_x = F.softmax(pred.detach().reshape((pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])), 1)
        max_x = torch.max(sm_x, dim=1, keepdim=False)
        weight = torch.mul(weight, max_x[0])

        # apply onss on images of multiple labels
        for i in range(label.shape[0]):
            tag = set(label[i].reshape(label.shape[1] * label.shape[2]).tolist()) - {255}
            if len(tag) <= 1:
                weight[i] = 1
        # else:
        #     weight = class_weight

        # apply weights and reduction
        if weight is not None:
            weight = weight.float()
        loss = self.weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss
