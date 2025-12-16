"""
TuSimple Evaluation Metrics
Implements official TuSimple benchmark metrics
"""

import numpy as np
import torch
import json
from collections import defaultdict


class TuSimpleEvaluator:
    """
    TuSimple Lane Detection Evaluator
    
    Metrics:
    1. Accuracy: Correctly predicted lane points / Total ground truth points
    2. False Positive (FP): Predicted lanes that don't match any ground truth
    3. False Negative (FN): Ground truth lanes that aren't matched by predictions
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.match_threshold = cfg.match_threshold
        self.reset()
    
    def reset(self):
        """Reset evaluation metrics"""
        self.total_tp = 0  # True positives (correct lane points)
        self.total_gt = 0  # Total ground truth points
        self.total_fp = 0  # False positives (extra lanes)
        self.total_fn = 0  # False negatives (missed lanes)
        self.total_samples = 0
    
    def update(self, predictions, targets):
        pred_loc = predictions['loc_row']            # [B, G, R, K]
        pred_exist = predictions['exist_row']        # [B, 2, R, K] or [B, R, K]
        tgt_loc = targets['loc_row']                 # [B, R, K] (class indices) OR [B, G, R, K]
        tgt_exist = targets['exist_row']             # [B, R, K]

        # Normalize pred_exist to [B, R, K] (prob of class=1)
        if pred_exist.dim() == 4 and pred_exist.size(1) == 2:
            pred_exist = torch.softmax(pred_exist, dim=1)[:, 1]  # [B, R, K]

        # Shapes
        B, G, R, K = pred_loc.shape

        # Decode predictions (logits)
        pred_lanes = self._decode_lanes(pred_loc, pred_exist, G, R, K)

        # Decode GT:
        # If provided as logits [B, G, R, K], convert to class indices
        if tgt_loc.dim() == 4 and tgt_loc.shape[1] == G:
            # argmax over grid dim -> [B, R, K]
            tgt_loc = torch.argmax(tgt_loc, dim=1)
        # Now tgt_loc must be [B, R, K] class indices
        gt_lanes = self._decode_lanes(tgt_loc, tgt_exist, G, R, K)

        # Accumulate metrics per sample
        for i in range(B):
            self._evaluate_sample(pred_lanes[i], gt_lanes[i])
            self.total_samples += 1
    
    def _decode_lanes(self, loc, exist, num_grid, num_row, num_lanes):
        """
        loc: either [B, num_grid, num_row, num_lanes] logits (pred) 
             or [B, num_row, num_lanes] class indices (target)
        exist: [B, num_row, num_lanes] existence (0/1 or prob)
        """
        import torch
        lanes_batch = []
        B = exist.shape[0]

        is_pred_logits = (loc.dim() == 4 and loc.shape[1] == num_grid)
        for b in range(B):
            lanes = []
            for k in range(num_lanes):
                pts = []
                for r in range(num_row):
                    # existence check
                    ex = exist[b, r, k]
                    if isinstance(ex, torch.Tensor):
                        ex_val = ex.item() if ex.dim() == 0 else float(ex)
                    else:
                        ex_val = float(ex)
                    if ex_val < 0.5:
                        continue

                    if is_pred_logits:
                        # pick grid cell by argmax from logits
                        col = loc[b, :, r, k]  # [num_grid]
                        ci = torch.argmax(col).item()
                        off = 0.5  # if you have offsets, decode them; else center of cell
                    else:
                        # class index target already provided
                        ci = int(loc[b, r, k])
                        if ci < 0 or ci >= num_grid:
                            continue
                        off = 0.5

                    cell_w = 1.0 / num_grid
                    x_norm = (ci * cell_w) + (off * cell_w)
                    y_norm = r / (num_row - 1) if num_row > 1 else 0.5
                    pts.append((x_norm, y_norm))
                if len(pts) >= 2:
                    lanes.append(pts)
            lanes_batch.append(lanes)
        return lanes_batch
    
    def _evaluate_sample(self, pred_lanes, gt_lanes):
        """
        Evaluate single sample
        
        Computes:
        - True positives: Matching lane points
        - False positives: Extra predicted lanes
        - False negatives: Missed ground truth lanes
        """
        if len(gt_lanes) == 0:
            self.total_fp += len(pred_lanes)
            return
        
        if len(pred_lanes) == 0:
            self.total_fn += len(gt_lanes)
            # Count all GT points
            for gt_lane in gt_lanes:
                self.total_gt += len(gt_lane)
            return
        
        # Match predicted lanes to ground truth
        matched_gt = set()
        matched_pred = set()
        
        # Compute distance matrix
        dist_matrix = np.zeros((len(pred_lanes), len(gt_lanes)))
        
        for i, pred_lane in enumerate(pred_lanes):
            for j, gt_lane in enumerate(gt_lanes):
                dist = self._compute_lane_distance(pred_lane, gt_lane)
                dist_matrix[i, j] = dist
        
        # Greedy matching: match closest pairs first
        while True:
            if len(matched_pred) == len(pred_lanes) or len(matched_gt) == len(gt_lanes):
                break
            
            # Find minimum distance
            min_dist = float('inf')
            min_i, min_j = -1, -1
            
            for i in range(len(pred_lanes)):
                if i in matched_pred:
                    continue
                for j in range(len(gt_lanes)):
                    if j in matched_gt:
                        continue
                    if dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1 or min_dist > (1 - self.match_threshold):
                break
            
            # Match this pair
            matched_pred.add(min_i)
            matched_gt.add(min_j)
            
            # Count true positives for this match
            tp_points = self._count_matching_points(
                pred_lanes[min_i],
                gt_lanes[min_j]
            )
            self.total_tp += tp_points
            self.total_gt += len(gt_lanes[min_j])
        
        # Count unmatched predictions as FP
        self.total_fp += (len(pred_lanes) - len(matched_pred))
        
        # Count unmatched GT as FN and add their points to total
        for j in range(len(gt_lanes)):
            if j not in matched_gt:
                self.total_fn += 1
                self.total_gt += len(gt_lanes[j])
    
    def _compute_lane_distance(self, lane1, lane2):
        """
        Compute distance between two lanes
        Uses average point-to-point distance
        """
        # Find common y range
        y1 = set([y for _, y in lane1])
        y2 = set([y for _, y in lane2])
        common_y = y1.intersection(y2)
        
        if len(common_y) == 0:
            return 1.0  # Maximum distance
        
        # Interpolate x values at common y positions
        distances = []
        for y in common_y:
            x1 = [x for x, y_val in lane1 if y_val == y][0]
            x2 = [x for x, y_val in lane2 if y_val == y][0]
            dist = abs(x1 - x2) / self.cfg.train_width  # Normalized distance
            distances.append(dist)
        
        return np.mean(distances)
    
    def _count_matching_points(self, pred_lane, gt_lane):
        """
        Count correctly predicted points
        A point is correct if within threshold of GT
        """
        threshold = 20 / self.cfg.train_width  # pixels (TuSimple standard)
        
        # Find common y values
        gt_dict = {y: x for x, y in gt_lane}
        
        correct = 0
        for x_pred_norm, y_norm in pred_lane:
            if y_norm in gt_dict:
                x_gt_norm = gt_dict[y_norm]

                #convert to pixel for comparison
                x_pred_norm = x_pred_norm * self.cfg.train_width
                x_gt_norm = x_gt_norm * self.cfg.train_width

                if abs(x_pred_norm - x_gt_norm) < threshold:
                    correct +=1
        return correct
    
    def compute_metrics(self):
        """
        Compute final metrics
        
        Returns:
            Dictionary with:
                - accuracy: TP / Total GT points
                - fp: False positive rate
                - fn: False negative rate
        """
        if self.total_gt == 0:
            accuracy = 0.0
        else:
            accuracy = self.total_tp / self.total_gt
        
        if self.total_samples == 0:
            fp_rate = 0.0
            fn_rate = 0.0
        else:
            fp_rate = self.total_fp / self.total_samples
            fn_rate = self.total_fn / self.total_samples
        
        return {
            'accuracy': accuracy,
            'fp': fp_rate,
            'fn': fn_rate,
            'tp': self.total_tp,
            'total_gt': self.total_gt
        }
    
    def __str__(self):
        metrics = self.compute_metrics()
        return (f"TuSimple Metrics:\n"
                f"  Accuracy: {metrics['accuracy']*100:.2f}%\n"
                f"  FP Rate: {metrics['fp']:.4f}\n"
                f"  FN Rate: {metrics['fn']:.4f}\n"
                f"  TP: {metrics['tp']}/{metrics['total_gt']}")


if __name__ == '__main__':
    # Test evaluator
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from configs.tusimple_config import Config
    
    cfg = Config()
    evaluator = TuSimpleEvaluator(cfg)
    
    # Create dummy predictions and targets
    B = 2
    predictions = {
        'loc_row': torch.randn(B, cfg.num_cell_row, cfg.num_row, cfg.num_lanes),
        'exist_row': torch.randn(B, 2, cfg.num_row, cfg.num_lanes)
    }
    
    targets = {
        'loc_row': torch.randn(B, cfg.num_cell_row, cfg.num_row, cfg.num_lanes),
        'exist_row': torch.randint(0, 2, (B, cfg.num_row, cfg.num_lanes))
    }
    
    # Update metrics
    evaluator.update(predictions, targets)
    
    # Print results
    print(evaluator)
