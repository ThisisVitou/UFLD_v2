import torch
import cv2
import numpy as np
import os
from data.tusimple_dataset import TuSimpleDataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.tusimple_config import Config
    

# 1. Mock Configuration (Must match what you use in training)
class Config:
    img_w = 800
    img_h = 320
    
    # Grid settings
    num_cell_row = 100
    num_cell_col = 100
    
    # Anchor settings
    num_row = 72
    num_col = 81
    row_anchor_start = 160
    
    num_lanes = 4

def test_visualization():
    cfg = Config()
    # 2. Setup Paths - CHANGE THESE to your actual paths
    # data_root = cfg.dataset  # e.g., './data/tusimple'
    # json_file = 'label_data_0313.json'            # Any valid json file in that root
    
    
    # 3. Instantiate Dataset
    # We pass None for transforms to see the raw resized image first
    dataset = TuSimpleDataset(cfg=cfg, mode='val')
    
    print(f"Dataset length: {len(dataset)}")
    
    # 4. Get a sample
    idx = 0 # Change this to check different images
    data = dataset[idx]
    
    img = data['img'] # numpy array (H, W, 3) because transforms=None
    label_row = data['label_row'] # Tensor
    label_col = data['label_col'] # Tensor
    
    # If img is a Tensor (if you added transforms), convert back to numpy
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
    
    # Make a copy to draw on
    vis_img = img.copy()
    
    # 5. Visualize Row Anchors (Horizontal grids)
    # Reconstruct row_anchors array used in dataset
    row_anchors = np.linspace(cfg.row_anchor_start, cfg.img_h - 1, cfg.num_row)
    grid_width_row = cfg.img_w / cfg.num_cell_row
    
    for lane_idx in range(cfg.num_lanes):
        for row_idx in range(cfg.num_row):
            grid_idx = label_row[row_idx, lane_idx].item()
            
            # Label 0 is usually background, so we skip it
            # (In your code: grid_idx = calculated_idx + 1)
            if grid_idx > 0:
                y = int(row_anchors[row_idx])
                # Convert grid index back to x pixel coordinate
                # We use the center of the grid cell
                x = int((grid_idx - 1) * grid_width_row + grid_width_row / 2)
                
                # Draw a circle
                cv2.circle(vis_img, (x, y), 2, (0, 0, 255), -1) # Red for Row anchors

    # 6. Visualize Column Anchors (Vertical grids)
    col_anchors = np.linspace(0, cfg.img_w - 1, cfg.num_col)
    grid_height_col = cfg.img_h / cfg.num_cell_col
    
    for lane_idx in range(cfg.num_lanes):
        for col_idx in range(cfg.num_col):
            grid_idx = label_col[col_idx, lane_idx].item()
            
            if grid_idx > 0:
                x = int(col_anchors[col_idx])
                y = int((grid_idx - 1) * grid_height_col + grid_height_col / 2)
                
                cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1) # Green for Col anchors

    # 7. Save or Show
    cv2.imshow('Validation', vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optional: Save to file
    cv2.imwrite('test_output.jpg', vis_img)
    print("Saved visualization to test_output.jpg")

if __name__ == "__main__":
    test_visualization()