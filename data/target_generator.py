"""
Target Generator for Grid-Based Lane Detection
Converts sparse lane annotations to grid-based anchor representation
"""

import numpy as np
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TargetGenerator:
    """
    Generate grid-based targets for UFLD model
    Converts continuous lane points to discrete grid anchors
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_cell_row = cfg.num_cell_row
        self.num_row = cfg.num_row
        self.num_cell_col = cfg.num_cell_col
        self.num_col = cfg.num_col
        self.num_lanes = cfg.num_lanes
        self.train_width = cfg.train_width
        self.train_height = cfg.train_height
        
        # Generate row and column anchor positions
        self.row_anchor = np.linspace(0, 1, self.num_cell_row, dtype=np.float32)
        self.col_anchor = np.linspace(0, 1, self.num_cell_col, dtype=np.float32)
        
    def generate_targets(self, lanes, h_samples):
        """
        Convert lane annotations to grid-based targets
        
        Args:
            lanes: List of lane annotations, each lane is list of x-coordinates
                   -2 indicates lane is not visible at that height
            h_samples: List of y-coordinates where lanes are sampled
                   
        Returns:
            Dictionary containing:
                - loc_row: Location targets for row anchors [num_cell_row, num_cls_row, num_lanes]
                - loc_col: Location targets for column anchors [num_cell_col, num_cls_col, num_lanes]
                - exist_row: Existence labels for row anchors [num_cls_row, num_lanes]
                - exist_col: Existence labels for column anchors [num_cls_col, num_lanes]
                - seg_mask: Optional segmentation mask [H, W] (if use_aux=True)
        """
        
        # Initialize targets with -1 (invalid) for classification labels
        loc_row = np.ones((self.num_row, self.num_lanes), dtype=np.int64) * -1
        loc_col = np.ones((self.num_col, self.num_lanes), dtype=np.int64) * -1
        exist_row = np.zeros((self.num_row, self.num_lanes), dtype=np.long)
        exist_col = np.zeros((self.num_col, self.num_lanes), dtype=np.long)
        
        # Normalize h_samples to [0, 1] based on image height
        h_samples_norm = np.array(h_samples) / self.train_height
        
        # Process each lane
        for lane_idx, lane in enumerate(lanes):
            if lane_idx >= self.num_lanes:
                break
                
            # Filter out invalid points (-2)
            valid_points = [(x, y) for x, y in zip(lane, h_samples) if x >= 0]
            
            if len(valid_points) < 2:
                continue
                
            # Extract valid x and y coordinates (normalized)
            lane_x = np.array([x / self.train_width for x, _ in valid_points], dtype=np.float32)
            lane_y = np.array([y / self.train_height for _, y in valid_points], dtype=np.float32)
            
            # Generate row-based targets (given y, predict x)
            row_targets = self._generate_row_targets(lane_x, lane_y)
            if row_targets is not None:
                loc_row[:, lane_idx] = row_targets['loc']  # Now [56, 4] shape
                exist_row[:, lane_idx] = row_targets['exist']
            
            # Generate column-based targets (given x, predict y)
            col_targets = self._generate_col_targets(lane_x, lane_y)
            if col_targets is not None:
                loc_col[:, lane_idx] = col_targets['loc']  # Now [41, 4] shape
                exist_col[:, lane_idx] = col_targets['exist']
        
        targets = {
            'loc_row': torch.from_numpy(loc_row),
            'loc_col': torch.from_numpy(loc_col),
            'exist_row': torch.from_numpy(exist_row),
            'exist_col': torch.from_numpy(exist_col)
        }
        
        # Generate segmentation mask if needed
        if self.cfg.use_aux:
            seg_mask = self._generate_seg_mask(lanes, h_samples)
            targets['seg_mask'] = torch.from_numpy(seg_mask)
        
        return targets
    
    def _generate_row_targets(self, lane_x, lane_y):
        """
        Generate row-based targets: predict which grid cell (0-99)
        Returns classification labels, not regression offsets
        """
        loc_targets = np.ones(self.num_row, dtype=np.int64) * -1
        exist_targets = np.zeros(self.num_row, dtype=np.int64)
        
        for cls_idx in range(self.num_row):
            y_pos = cls_idx / (self.num_row - 1)
            
            if y_pos < lane_y.min() or y_pos > lane_y.max():
                continue
            
            x_pos = np.interp(y_pos, lane_y, lane_x)
            
            # CRITICAL: Store grid cell INDEX (0-99), not offset (0-1)
            cell_idx = int(x_pos * self.num_cell_row)
            cell_idx = np.clip(cell_idx, 0, self.num_cell_row - 1)
            
            loc_targets[cls_idx] = cell_idx  # Integer class label
            exist_targets[cls_idx] = 1
        
        return {'loc': loc_targets, 'exist': exist_targets}
    
    def _generate_col_targets(self, lane_x, lane_y):
        """
        Generate column-based targets: predict which grid cell (0-99)
        """
        loc_targets = np.ones(self.num_col, dtype=np.int64) * -1
        exist_targets = np.zeros(self.num_col, dtype=np.int64)
        
        for cls_idx in range(self.num_col):
            x_pos = cls_idx / (self.num_col - 1)
            
            if x_pos < lane_x.min() or x_pos > lane_x.max():
                continue
            
            if len(lane_x) < 2:
                continue
            
            sorted_indices = np.argsort(lane_x)
            sorted_x = lane_x[sorted_indices]
            sorted_y = lane_y[sorted_indices]
            
            if x_pos < sorted_x[0] or x_pos > sorted_x[-1]:
                continue
            
            y_pos = np.interp(x_pos, sorted_x, sorted_y)
            
            # CRITICAL: Store grid cell INDEX (0-99), not offset (0-1)
            cell_idx = int(y_pos * self.num_cell_col)
            cell_idx = np.clip(cell_idx, 0, self.num_cell_col - 1)
            
            loc_targets[cls_idx] = cell_idx  # Integer class label
            exist_targets[cls_idx] = 1
        
        return {'loc': loc_targets, 'exist': exist_targets}
    
    def _generate_seg_mask(self, lanes, h_samples):
        """
        Generate pixel-wise segmentation mask for auxiliary supervision
        
        Args:
            lanes: List of lane annotations
            h_samples: List of y-coordinates
            
        Returns:
            Segmentation mask [H, W] with lane labels (0=background, 1-4=lanes)
        """
        # Create mask at 1/8 resolution for efficiency (matching model output)
        mask_h = self.train_height // 8
        mask_w = self.train_width // 8
        seg_mask = np.zeros((mask_h, mask_w), dtype=np.long)
        
        for lane_idx, lane in enumerate(lanes):
            if lane_idx >= self.num_lanes:
                break
            
            # Filter valid points
            valid_points = [(int(x / 8), int(y / 8)) for x, y in zip(lane, h_samples) 
                          if x >= 0 and y >= 0]
            
            if len(valid_points) < 2:
                continue
            
            # Draw lane line on mask (simple implementation)
            for i in range(len(valid_points) - 1):
                x1, y1 = valid_points[i]
                x2, y2 = valid_points[i + 1]
                
                # Clip coordinates
                x1 = np.clip(x1, 0, mask_w - 1)
                x2 = np.clip(x2, 0, mask_w - 1)
                y1 = np.clip(y1, 0, mask_h - 1)
                y2 = np.clip(y2, 0, mask_h - 1)
                
                # Draw line segment (simple nearest neighbor)
                num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
                xs = np.linspace(x1, x2, num_points, dtype=int)
                ys = np.linspace(y1, y2, num_points, dtype=int)
                
                for x, y in zip(xs, ys):
                    if 0 <= y < mask_h and 0 <= x < mask_w:
                        seg_mask[y, x] = lane_idx + 1  # Lane ID (1-4)
        
        return seg_mask


if __name__ == '__main__':
    # Test target generator
    from configs.tusimple_config import Config
    
    cfg = Config()
    generator = TargetGenerator(cfg)
    
    # Test with sample lane data
    h_samples = list(range(160, 711, 10))
    lanes = [
        [500, 505, 510, 515, 520] + [-2] * (len(h_samples) - 5),  # Lane 1
        [700, 705, 710, 715, 720] + [-2] * (len(h_samples) - 5),  # Lane 2
    ]
    
    targets = generator.generate_targets(lanes, h_samples)
    
    print("Generated targets:")
    print(f"  loc_row shape: {targets['loc_row'].shape}")
    print(f"  loc_col shape: {targets['loc_col'].shape}")
    print(f"  exist_row shape: {targets['exist_row'].shape}")
    print(f"  exist_col shape: {targets['exist_col'].shape}")
    if 'seg_mask' in targets:
        print(f"  seg_mask shape: {targets['seg_mask'].shape}")
