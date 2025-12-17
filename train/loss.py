import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import SoftmaxFocalLoss, MeanLoss, ParsingRelationLoss, ParsingRelationDis


class UFLDLoss(nn.Module):
    def __init__(self, cfg):
        super(UFLDLoss, self).__init__()
        self.cfg = cfg
        
        # Initialize all loss functions
        self.cls_loss_fn = SoftmaxFocalLoss(gamma=2, ignore_lb=-1, soft_loss=True)
        self.mean_loss_fn = MeanLoss()
        self.exist_loss_fn = nn.CrossEntropyLoss()
        
        # Structural losses
        self.relation_loss_fn = ParsingRelationLoss()
        self.relation_dis_fn = ParsingRelationDis()
        
        # Loss weights from config
        self.mean_loss_w = cfg.loss_weights.get('mean_loss', 0.05)
        self.sim_loss_w = cfg.loss_weights.get('relation', 0.0)
        self.shp_loss_w = cfg.loss_weights.get('relation_dis', 0.0)
        
        self.use_aux = cfg.use_aux
        
    def forward(self, predictions, targets):
        """
        predictions: dict with keys: 
            - loc_row: [B, 100, 56, 4] - classification logits
            - loc_col:  [B, 100, 41, 4]
            - exist_row: [B, 2, 56, 4] - binary classification
            - exist_col: [B, 2, 41, 4]
        
        targets: dict with keys:
            - loc_row: [B, 56, 4] - class indices 0-99
            - loc_col: [B, 41, 4]
            - exist_row: [B, 56, 4] - 0 or 1
            - exist_col: [B, 41, 4]
        """
        
        # === ROW LOSSES ===
        # 1. Classification loss (SoftmaxFocalLoss)
        pred_loc_row = predictions['loc_row']  # [B, 100, 56, 4]
        target_loc_row = targets['loc_row']    # [B, 56, 4]
        
        # Reshape:  [B, 100, 56, 4] -> [B*56*4, 100]
        B, C, H, W = pred_loc_row.shape
        pred_loc_row_flat = pred_loc_row.permute(0, 2, 3, 1).reshape(-1, C)
        target_loc_row_flat = target_loc_row. reshape(-1)
        
        cls_loss_row = self.cls_loss_fn(pred_loc_row_flat, target_loc_row_flat)
        
        # 2. Mean loss (regression on expected position)
        mean_loss_row = self.mean_loss_fn(pred_loc_row, target_loc_row)
        
        # 3. Existence loss
        pred_exist_row = predictions['exist_row']  # [B, 2, 56, 4]
        target_exist_row = targets['exist_row']     # [B, 56, 4]
        
        pred_exist_row_flat = pred_exist_row.permute(0, 2, 3, 1).reshape(-1, 2)
        target_exist_row_flat = target_exist_row.reshape(-1).long()
        
        exist_loss_row = self.exist_loss_fn(pred_exist_row_flat, target_exist_row_flat)
        
        # === COLUMN LOSSES (same structure) ===
        pred_loc_col = predictions['loc_col']
        target_loc_col = targets['loc_col']
        
        B, C, H, W = pred_loc_col.shape
        pred_loc_col_flat = pred_loc_col.permute(0, 2, 3, 1).reshape(-1, C)
        target_loc_col_flat = target_loc_col.reshape(-1)
        
        cls_loss_col = self.cls_loss_fn(pred_loc_col_flat, target_loc_col_flat)
        mean_loss_col = self.mean_loss_fn(pred_loc_col, target_loc_col)
        
        pred_exist_col = predictions['exist_col']
        target_exist_col = targets['exist_col']
        
        pred_exist_col_flat = pred_exist_col.permute(0, 2, 3, 1).reshape(-1, 2)
        target_exist_col_flat = target_exist_col. reshape(-1).long()
        
        exist_loss_col = self.exist_loss_fn(pred_exist_col_flat, target_exist_col_flat)
        
        # === STRUCTURAL LOSSES ===
        relation_loss_row = self.relation_loss_fn(predictions['loc_row'])
        relation_loss_col = self.relation_loss_fn(predictions['loc_col'])
        relation_loss = (relation_loss_row + relation_loss_col) / 2.0
        
        relation_dis_row = self.relation_dis_fn(predictions['loc_row'])
        relation_dis_col = self.relation_dis_fn(predictions['loc_col'])
        relation_dis = (relation_dis_row + relation_dis_col) / 2.0
        
        # === TOTAL LOSS ===
        total_loss = (
            cls_loss_row + cls_loss_col +                    # Classification:  weight=1.0
            exist_loss_row + exist_loss_col +                 # Existence: weight=1.0
            self.mean_loss_w * (mean_loss_row + mean_loss_col) +  # Mean: weight=0.05
            self.sim_loss_w * relation_loss +                 # Relation: weight=0.0
            self.shp_loss_w * relation_dis                    # Relation dis: weight=0.0
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss':  (cls_loss_row + cls_loss_col) / 2,
            'mean_loss': (mean_loss_row + mean_loss_col) / 2,
            'exist_loss': (exist_loss_row + exist_loss_col) / 2,
            'relation_loss': relation_loss,
            'relation_dis': relation_dis
        }
        
        # Segmentation loss if auxiliary head is used
        if self.use_aux and 'seg_out' in predictions:
            seg_loss = nn.CrossEntropyLoss()(
                predictions['seg_out'],
                targets['seg_mask']. long()
            )
            total_loss = total_loss + seg_loss
            loss_dict['seg_loss'] = seg_loss
            loss_dict['total_loss'] = total_loss
        
        return loss_dict