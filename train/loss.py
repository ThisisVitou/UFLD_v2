"""
Loss Functions for Ultra-Fast Lane Detection
Combines location, existence, and segmentation losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.loc_weight = cfg.loss_weights.get('loc', 1.0)
        self.exist_weight = cfg.loss_weights.get('exist', 0.1)
        self.seg_weight = cfg.loss_weights.get('seg', 1.0)
        self.relation_weight = cfg.loss_weights.get('relation', 0.0)
        self.relation_dis_weight = cfg.loss_weights.get('relation_dis', 0.0)
        
        self.use_aux = cfg.use_aux
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Structural losses
        if self.relation_weight > 0:
            self.relation_loss = ParsingRelationLoss()
        if self.relation_dis_weight > 0:
            self.relation_dis_loss = ParsingRelationDis()
        
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
        
        # Add structural losses if enabled
        if self.relation_weight > 0:
            relation_loss_row = self.relation_loss(predictions['loc_row'])
            relation_loss_col = self.relation_loss(predictions['loc_col'])
            relation_loss = (relation_loss_row + relation_loss_col) / 2.0
            total_loss = total_loss + self.relation_weight * relation_loss
            loss_dict['relation_loss'] = relation_loss
        
        if self.relation_dis_weight > 0:
            relation_dis_loss_row = self.relation_dis_loss(predictions['loc_row'])
            relation_dis_loss_col = self.relation_dis_loss(predictions['loc_col'])
            relation_dis_loss = (relation_dis_loss_row + relation_dis_loss_col) / 2.0
            total_loss = total_loss + self.relation_dis_weight * relation_dis_loss
            loss_dict['relation_dis_loss'] = relation_dis_loss
        
        loss_dict['total_loss'] = total_loss
        
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
        Compute location classification loss using CrossEntropy
        
        Args:
            pred_loc: [B, num_grid=100, num_cls=56, num_lanes=4] - logits for 100 classes
            target_loc: [B, num_cls=56, num_lanes=4] - ground truth grid cell indices (0-99)
            target_exist: [B, num_cls=56, num_lanes=4] - lane existence (0 or 1)
        """
        B, num_grid, num_cls, num_lanes = pred_loc.shape
        
        # Reshape for CrossEntropyLoss
        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous()  # [B, num_cls, num_lanes, num_grid]
        pred_loc = pred_loc.view(-1, num_grid)  # [B*num_cls*num_lanes, num_grid]
        
        target_loc = target_loc.contiguous().view(-1)  # [B*num_cls*num_lanes]
        
        # Mask for valid targets (where lane exists and target >= 0)
        valid_mask = (target_loc >= 0) & (target_exist.view(-1) == 1)
        
        if valid_mask.sum() > 0:
            loss = F.cross_entropy(pred_loc[valid_mask], target_loc[valid_mask].long(), reduction='mean')
        else:
            loss = pred_loc.sum() * 0
        
        return loss
    
    def _compute_existence_loss(self, pred_exist, target_exist, mask=None):
        """
        pred_exist: [B, 2, num_row, num_lanes] logits (class=0/1)
        target_exist: [B, num_row, num_lanes] class indices {0,1}
        """
        # Flatten to [N, 2] and [N]
        B, C, R, K = pred_exist.shape
        pred_exist_flat = pred_exist.permute(0, 2, 3, 1).reshape(-1, C)   # [B*R*K, 2]
        target_exist_flat = target_exist.reshape(-1).contiguous()         # [B*R*K]

        # Cast target to Long for CrossEntropyLoss
        target_exist_flat = target_exist_flat.long()

        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            pred_exist_flat = pred_exist_flat[mask_flat]
            target_exist_flat = target_exist_flat[mask_flat]

        loss = self.ce_loss(pred_exist_flat, target_exist_flat)
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
