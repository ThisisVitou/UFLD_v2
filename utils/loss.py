# Create utils/loss.py - Copy from cfzd's repo
import torch
import torch.nn as nn
import torch.nn. functional as F

def soft_nll(pred, target, ignore_index=-1):
    """Label smoothing for grid-based targets"""
    C = pred.shape[1]
    invalid_target_index = target == ignore_index

    ttarget = target.clone()
    ttarget[invalid_target_index] = C

    target_l = target - 1
    target_r = target + 1

    invalid_part_l = target_l == -1
    invalid_part_r = target_r == C

    invalid_target_l_index = torch.logical_or(invalid_target_index, invalid_part_l)
    target_l[invalid_target_l_index] = C

    invalid_target_r_index = torch.logical_or(invalid_target_index, invalid_part_r)
    target_r[invalid_target_r_index] = C

    supp_part_l = target. clone()
    supp_part_r = target.clone()
    supp_part_l[target != 0] = C
    supp_part_r[target != C-1] = C

    target_onehot = torch.nn.functional.one_hot(ttarget, num_classes=C+1)
    target_onehot = target_onehot[..., :-1].permute(0, 3, 1, 2)

    target_l_onehot = torch.nn.functional.one_hot(target_l, num_classes=C+1)
    target_l_onehot = target_l_onehot[..., :-1].permute(0, 3, 1, 2)

    target_r_onehot = torch.nn.functional.one_hot(target_r, num_classes=C+1)
    target_r_onehot = target_r_onehot[..., :-1].permute(0, 3, 1, 2)

    supp_part_l_onehot = torch.nn.functional. one_hot(supp_part_l, num_classes=C+1)
    supp_part_l_onehot = supp_part_l_onehot[..., :-1].permute(0, 3, 1, 2)

    supp_part_r_onehot = torch.nn. functional.one_hot(supp_part_r, num_classes=C+1)
    supp_part_r_onehot = supp_part_r_onehot[..., :-1].permute(0, 3, 1, 2)

    # Smooth labels:  0. 9 main + 0.05 left + 0.05 right + 0.05 boundary smoothing
    target_fusion = 0.9 * target_onehot + 0.05 * target_l_onehot + 0.05 * target_r_onehot + 0.05 * supp_part_l_onehot + 0.05 * supp_part_r_onehot
    
    return -(target_fusion * pred).sum() / (target != ignore_index).sum()


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2, ignore_lb=-1, soft_loss=True):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_lb = ignore_lb
        self. soft_loss = soft_loss
        if not self.soft_loss:
            self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        # Compute softmax probabilities
        scores = F.softmax(logits, dim=1)
        
        # Focal weight:  (1 - p)^gamma
        factor = torch.pow(1. - scores, self.gamma)
        
        # Log softmax
        log_score = F.log_softmax(logits, dim=1)
        
        # Apply focal weight
        log_score = factor * log_score
        
        # Use soft NLL with label smoothing
        if self.soft_loss:
            loss = soft_nll(log_score, labels, ignore_index=self. ignore_lb)
        else:
            loss = self.nll(log_score, labels)
        
        return loss


class MeanLoss(nn.Module):
    """Regression loss - converts softmax to expected position"""
    def __init__(self):
        super(MeanLoss, self).__init__()
        self.l1 = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, logits, label):
        n, c, h, w = logits.shape
        # Create grid:  [0, 1, 2, ..., 99] for 100 grid cells
        grid = torch.arange(c, device=logits. device).view(1, c, 1, 1)
        
        # Compute expected position:  sum(p_i * i)
        logits = (logits.softmax(1) * grid).sum(1)
        
        # Compute loss only on valid targets
        loss = self.l1(logits, label.float())[label != -1]
        return loss.mean()
    

class ParsingRelationLoss(nn.Module):
    """Enforce smoothness between adjacent row predictions"""
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
        
    def forward(self, logits):
        """
        Args:
            logits: [B, num_grid, num_cls, num_lanes]
        """
        n, c, h, w = logits.shape
        loss_all = []
        
        for i in range(h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i+1, :])
        
        loss = torch.cat(loss_all)
        return F.smooth_l1_loss(loss, torch.zeros_like(loss))


class ParsingRelationDis(nn.Module):
    """Enforce consistent spacing between lanes"""
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, x):
        """
        Args:
            x: [B, num_grid, num_cls, num_lanes]
        """
        n, dim, num_rows, num_cols = x.shape
        
        x_soft = F.softmax(x, dim=1)  # Softmax over all num_grid classes
        embedding = torch.arange(dim, dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
        pos = torch.sum(x_soft * embedding, dim=1)
        
        diff_list = []
        for i in range(num_rows // 2):
            diff_list.append(pos[:, i, :] - pos[:, i+1, :])
        
        if len(diff_list) <= 1:
            return x.sum() * 0
        
        loss = 0
        for i in range(len(diff_list) - 1):
            loss += self.l1(diff_list[i], diff_list[i+1])
        
        return loss / (len(diff_list) - 1)