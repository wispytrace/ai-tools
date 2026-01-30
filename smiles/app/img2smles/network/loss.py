import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        """
        Args:
            gamma (float): 聚焦参数，越大越关注难样本 (推荐 2.0)
            alpha (list/tensor): 类别权重，用于解决样本不平衡 (推荐用倒数频率)
            ignore_index (int): 忽略的 padding index
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [Batch, Seq, Vocab] -> [N, Vocab]
        # targets: [Batch, Seq] -> [N]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # 过滤 ignore_index (padding)
        valid_mask = targets != self.ignore_index
        logits = logits[valid_mask]
        targets = targets[valid_mask]

        if logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 计算标准 CE Loss (不归约)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 计算概率 p_t
        p_t = torch.exp(-ce_loss)
        
        # 计算 Focal Term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # 如果有类别权重 alpha
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss