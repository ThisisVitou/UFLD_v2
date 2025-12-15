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
    
    def update(self, predictions, targets, h_samples=None):
        """
        Update metrics with batch predictions
        
        Args:
            predictions: Dictionary with model outputs
                - loc_row: [B, num_cell_row, num_row, num_lanes]
                - exist_row: [B, 2, num_row, num_lanes]
            targets: Dictionary with ground truth
                - loc_row: [B, num_cell_row, num_row, num_lanes]
                - exist_row: [B, num_row, num_lanes]
            h_samples: List of y-coordinates for evaluation
        """
        if h_samples is None:
            h_samples = self.cfg.h_samples
        
        batch_size = predictions['loc_row'].shape[0]
        
        for b in range(batch_size):
            # Decode predictions to lane points
            pred_lanes = self._decode_lanes(
                predictions['loc_row'][b],
                predictions['exist_row'][b],
                h_samples
            )
            
            # Decode ground truth
            gt_lanes = self._decode_lanes(
                targets['loc_row'][b],
                targets['exist_row'][b],
                h_samples,
                is_gt=True
            )
            
            # Compute metrics for this sample
            self._evaluate_sample(pred_lanes, gt_lanes)
            self.total_samples += 1
    
    def _decode_lanes(self, loc, exist, h_samples, is_gt=False):
        """
        Decode grid-based predictions to lane points
        
        Args:
            loc: [num_cell_row, num_row, num_lanes]
            exist: [2, num_row, num_lanes] or [num_row, num_lanes] for GT
            h_samples: List of y-coordinates
            is_gt: Whether this is ground truth
        
        Returns:
            List of lanes, each lane is list of (x, y) points
        """
        num_lanes = loc.shape[-1]
        lanes = []
        
        # Convert to numpy if tensor
        if torch.is_tensor(loc):
            loc = loc.cpu().numpy()
        if torch.is_tensor(exist):
            exist = exist.cpu().numpy()
        
        # Process each lane
        for lane_idx in range(num_lanes):
            lane_points = []
            
            for cls_idx in range(len(h_samples)):
                y = h_samples[cls_idx]
                
                # Check if lane exists at this position
                if is_gt:
                    exists = exist[cls_idx, lane_idx] > 0.5
                else:
                    # For predictions, use softmax
                    exist_prob = exist[:, cls_idx, lane_idx]
                    exists = exist_prob[1] > exist_prob[0]  # Class 1 (exists) > Class 0 (not exists)
                
                if not exists:
                    continue
                
                # Get location prediction
                loc_values = loc[:, cls_idx, lane_idx]
                
                # Find valid predictions (not -1e5)
                valid_mask = loc_values > -1e4
                if not valid_mask.any():
                    continue
                
                # Find cell with highest confidence (or first valid cell)
                valid_cells = np.where(valid_mask)[0]
                if len(valid_cells) == 0:
                    continue
                
                # Use first valid cell (could use argmax for multi-cell predictions)
                cell_idx = valid_cells[0]
                offset = loc_values[cell_idx]
                
                # Convert cell index and offset to x coordinate
                cell_start = cell_idx / loc.shape[0]
                cell_size = 1.0 / loc.shape[0]
                x_norm = cell_start + offset * cell_size
                
                # Convert to pixel coordinates
                x = x_norm * self.cfg.train_width
                
                # Clip to image bounds
                x = np.clip(x, 0, self.cfg.train_width - 1)
                
                lane_points.append((x, y))
            
            # Only add lanes with sufficient points
            if len(lane_points) >= 2:
                lanes.append(lane_points)
        
        return lanes
    
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
        threshold = 20  # pixels (TuSimple standard)
        
        # Find common y values
        gt_dict = {y: x for x, y in gt_lane}
        
        correct = 0
        for x_pred, y in pred_lane:
            if y in gt_dict:
                x_gt = gt_dict[y]
                if abs(x_pred - x_gt) < threshold:
                    correct += 1
        
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
