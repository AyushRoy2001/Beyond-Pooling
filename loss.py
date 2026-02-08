import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import exp

from parse_options import parse_option

args = parse_option()

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss
    

class CosineSimilarityLoss(nn.Module):
    def __init__(self, num_axes=1, eps=1e-8):
        super().__init__()
        self.num_axes = num_axes   # Number of dominant axes (K) to consider
        self.eps = eps             # For numerical stability

    def forward(self, patch_tokens, text_features_inv, text_features_dom):
        """
        patch_tokens:        [1, 768]  (features for a patch or image)
        text_features_inv:   [768, 2]  (e.g., "photo of a good/bad object")
        text_features_dom:   [768, 2]  (e.g., "photo of a good/bad chestxray")
        """
        patch_tokens = torch.nan_to_num(patch_tokens, nan=0.0)
        text_features_inv = torch.nan_to_num(text_features_inv, nan=0.0)
        text_features_dom = torch.nan_to_num(text_features_dom, nan=0.0)
        # Stack and normalize text features: [4, 768]
        txt = torch.cat([text_features_dom.T, text_features_inv.T], dim=0)
        txt = F.normalize(txt, dim=1, eps=self.eps)
        # Covariance on hypersphere
        txt_centered = txt - txt.mean(dim=0, keepdim=True)
        cov = txt_centered.t() @ txt_centered / (txt_centered.shape[0] - 1) # [768,768]
        # Eigendecomposition: extract axes of highest variance
        eigvals, eigvecs = torch.linalg.eigh(cov.float())
        idx = torch.argsort(eigvals, descending=True)
        main_axes = eigvecs[:, idx[:self.num_axes]]  # Shape: [768, num_axes]

        patch_norm = F.normalize(patch_tokens, dim=-1, eps=self.eps)
        # Compute |cosine similarity| between patch and dominant eigenvectors
        # Result: [1, num_axes]
        similarity = torch.abs(torch.mm(patch_norm, main_axes))
        loss = similarity.mean()
        return loss

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, module_list):
        loss = 0.0
        count = 0
        for module in module_list:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() == 2:  # Only 2D weights (linear layers)
                    W = param
                    # Compute Gram matrix: W^T * W
                    WT_W = torch.matmul(W.t(), W)  
                    I = torch.eye(WT_W.size(0), device=W.device, dtype=W.dtype)
                    loss += torch.norm(WT_W - I, p='fro')  # Frobenius norm
                    count += 1
        if count > 0:
            loss = loss / count
        return loss
