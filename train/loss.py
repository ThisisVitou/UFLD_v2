"""
Loss Functions for Ultra-Fast Lane Detection
Combines location, existence, and segmentation losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UFLDLoss(nn.Module):
    """
    Composite loss for UFLD model
    
    Combines:
    1. Location loss (SmoothL1) for lane position prediction
    2. Existence loss (CrossEntropy) for lane presence detection
    3. Segmentation loss (CrossEntropy) for auxiliary supervision (optional)
    """
    
    def __init__(self, cfg):
        super(UFLDLoss, self).__init__()
        self.cfg = cfg
        
        # Loss weights
        self.loc_weight = cfg.loss_weights['loc']
        self.exist_weight = cfg.loss_weights['exist']
        self.seg_weight = cfg.loss_weights['seg']
        
        # Use auxiliary segmentation
        self.use_aux = cfg.use_aux
        
        # Loss functions
        self.smoothl1_loss = nn.SmoothL1Loss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        """
        Compute composite loss
        
        Args:
            predictions: Dictionary containing model outputs
                - loc_row: [B, num_cell_row, num_row, num_lanes]
                - loc_col: [B, num_cell_col, num_col, num_lanes]
                - exist_row: [B, 2, num_row, num_lanes]
                - exist_col: [B, 2, num_col, num_lanes]
                - seg_out: [B, num_lanes+1, H/8, W/8] (optional)
            
            targets: Dictionary containing ground truth
                - loc_row: [B, num_cell_row, num_row, num_lanes]
                - loc_col: [B, num_cell_col, num_col, num_lanes]
                - exist_row: [B, num_row, num_lanes]
                - exist_col: [B, num_col, num_lanes]
                - seg_mask: [B, H/8, W/8] (optional)
        
        Returns:
            Dictionary containing:
                - total_loss: Weighted sum of all losses
                - loc_loss: Location loss
                - exist_loss: Existence loss
                - seg_loss: Segmentation loss (if use_aux=True)
        """
        
        # Location loss (row)
        loc_row_loss = self._compute_location_loss(
            predictions['loc_row'],
            targets['loc_row'],
            targets['exist_row']
        )
        
        # Location loss (col)
        loc_col_loss = self._compute_location_loss(
            predictions['loc_col'],
            targets['loc_col'],
            targets['exist_col']
        )
        
        # Combined location loss
        loc_loss = (loc_row_loss + loc_col_loss) / 2.0
        
        # Existence loss (row)
        exist_row_loss = self._compute_existence_loss(
            predictions['exist_row'],
            targets['exist_row']
        )
        
        # Existence loss (col)
        exist_col_loss = self._compute_existence_loss(
            predictions['exist_col'],
            targets['exist_col']
        )
        
        # Combined existence loss
        exist_loss = (exist_row_loss + exist_col_loss) / 2.0
        
        # Total loss
        total_loss = (self.loc_weight * loc_loss + 
                     self.exist_weight * exist_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'loc_loss': loc_loss,
            'exist_loss': exist_loss
        }
        
        # Segmentation loss (if auxiliary head is used)
        if self.use_aux and 'seg_out' in predictions:
            seg_loss = self._compute_segmentation_loss(
                predictions['seg_out'],
                targets['seg_mask']
            )
            total_loss = total_loss + self.seg_weight * seg_loss
            loss_dict['seg_loss'] = seg_loss
            loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def _compute_location_loss(self, pred_loc, target_loc, target_exist):
        """
        Compute location regression loss
        Only compute loss for existing lanes (where target_exist == 1)
        
        Args:
            pred_loc: [B, num_cell, num_cls, num_lanes]
            target_loc: [B, num_cell, num_cls, num_lanes]
            target_exist: [B, num_cls, num_lanes]
        
        Returns:
            Scalar loss value
        """
        B, num_cell, num_cls, num_lanes = pred_loc.shape
        
        # Compute SmoothL1 loss
        loss = self.smoothl1_loss(pred_loc, target_loc)  # [B, num_cell, num_cls, num_lanes]
        
        # Create mask for valid locations (where lane exists)
        # Expand target_exist to match loss shape
        exist_mask = target_exist.unsqueeze(1)  # [B, 1, num_cls, num_lanes]
        exist_mask = exist_mask.expand_as(loss)  # [B, num_cell, num_cls, num_lanes]
        
        # Mask for valid targets (not -1e5)
        valid_mask = (target_loc > -1e4).float()
        
        # Combined mask
        mask = exist_mask.float() * valid_mask
        
        # Apply mask and compute mean
        masked_loss = loss * mask
        
        # Avoid division by zero
        num_valid = mask.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum()
    
    def _compute_existence_loss(self, pred_exist, target_exist):
        """
        Compute existence classification loss
        
        Args:
            pred_exist: [B, 2, num_cls, num_lanes] (logits for binary classification)
            target_exist: [B, num_cls, num_lanes] (0 or 1)
        
        Returns:
            Scalar loss value
        """
        B, _, num_cls, num_lanes = pred_exist.shape
        
        # Reshape for CrossEntropyLoss
        # pred: [B*num_cls*num_lanes, 2]
        # target: [B*num_cls*num_lanes]
        pred_exist = pred_exist.permute(0, 2, 3, 1).contiguous()  # [B, num_cls, num_lanes, 2]
        pred_exist = pred_exist.view(-1, 2)  # [B*num_cls*num_lanes, 2]
        
        target_exist = target_exist.contiguous().view(-1)  # [B*num_cls*num_lanes]
        
        # Compute CrossEntropy loss
        loss = self.ce_loss(pred_exist, target_exist)
        
        return loss
    
    def _compute_segmentation_loss(self, pred_seg, target_seg):
        """
        Compute pixel-wise segmentation loss
        
        Args:
            pred_seg: [B, num_lanes+1, H, W] (logits for each lane + background)
            target_seg: [B, H, W] (lane labels 0-4)
        
        Returns:
            Scalar loss value
        """
        # Compute CrossEntropy loss
        loss = self.ce_loss(pred_seg, target_seg.long())
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Optional alternative to standard CrossEntropy
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == '__main__':
    # Test loss functions
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from configs.tusimple_config import Config
    
    cfg = Config()
    loss_fn = UFLDLoss(cfg)
    
    # Create dummy predictions and targets
    B = 4
    predictions = {
        'loc_row': torch.randn(B, cfg.num_cell_row, cfg.num_row, cfg.num_lanes),
        'loc_col': torch.randn(B, cfg.num_cell_col, cfg.num_col, cfg.num_lanes),
        'exist_row': torch.randn(B, 2, cfg.num_row, cfg.num_lanes),
        'exist_col': torch.randn(B, 2, cfg.num_col, cfg.num_lanes),
        'seg_out': torch.randn(B, cfg.num_lanes + 1, cfg.train_height // 8, cfg.train_width // 8)
    }
    
    targets = {
        'loc_row': torch.randn(B, cfg.num_cell_row, cfg.num_row, cfg.num_lanes),
        'loc_col': torch.randn(B, cfg.num_cell_col, cfg.num_col, cfg.num_lanes),
        'exist_row': torch.randint(0, 2, (B, cfg.num_row, cfg.num_lanes)),
        'exist_col': torch.randint(0, 2, (B, cfg.num_col, cfg.num_lanes)),
        'seg_mask': torch.randint(0, cfg.num_lanes + 1, (B, cfg.train_height // 8, cfg.train_width // 8))
    }
    
    # Compute loss
    loss_dict = loss_fn(predictions, targets)
    
    print("Loss computation test:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
