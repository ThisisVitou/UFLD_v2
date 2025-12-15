"""
Visualization utilities for lane detection
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import os


def decode_predictions(predictions, cfg, threshold=0.5):
    """
    Decode model predictions to lane points
    
    Args:
        predictions: Dictionary with model outputs
            - loc_row: [B, num_cell_row, num_row, num_lanes]
            - exist_row: [B, 2, num_row, num_lanes] or [B, num_row, num_lanes]
        cfg: Configuration object
        threshold: Existence threshold
    
    Returns:
        List of lanes for each image in batch
        Each lane is list of (x, y) tuples
    """
    import torch
    
    batch_size = predictions['loc_row'].shape[0]
    num_cell_row = predictions['loc_row'].shape[1]  # Get from actual tensor shape
    num_row = predictions['loc_row'].shape[2]
    num_lanes = predictions['loc_row'].shape[3]
    
    all_lanes = []
    
    for b in range(batch_size):
        batch_lanes = []
        
        # Process each lane
        for lane_idx in range(num_lanes):
            lane_points = []
            
            # Get location and existence predictions for this lane
            loc_row = predictions['loc_row'][b, :, :, lane_idx]  # [num_cell_row, num_row]
            
            # Handle existence - could be [2, num_row, num_lanes] or [num_row, num_lanes]
            if predictions['exist_row'].dim() == 4:  # [B, 2, num_row, num_lanes]
                exist_row = torch.softmax(predictions['exist_row'][b, :, :, lane_idx], dim=0)
                exist_prob = exist_row[1]  # Probability of existence
            else:  # [B, num_row, num_lanes]
                exist_prob = predictions['exist_row'][b, :, lane_idx]
            
            # For each row position
            for row_idx in range(num_row):
                # Check if lane exists at this row
                if exist_prob[row_idx] < threshold:
                    continue
                
                # Get location prediction for this row
                loc_pred = loc_row[:, row_idx]  # [num_cell_row]
                
                # Find the cell with maximum value (which grid cell contains the lane)
                max_cell = torch.argmax(loc_pred).item()
                offset = loc_pred[max_cell].item()
                
                # Skip if offset is invalid (uninitialized target value)
                if offset < -1e4 or offset > 1.0 or offset < 0.0:
                    continue
                
                # Convert cell index + offset to normalized x coordinate [0, 1]
                # max_cell is in range [0, num_cell_row-1]
                # offset is in range [0, 1] (position within the cell)
                cell_width = 1.0 / num_cell_row
                x_normalized = (max_cell * cell_width) + (offset * cell_width)
                
                # Clamp to valid range
                x_normalized = max(0.0, min(1.0, x_normalized))
                
                # Convert row index to normalized y coordinate [0, 1]
                y_normalized = row_idx / (num_row - 1) if num_row > 1 else 0.5
                
                # Scale to image dimensions
                x = x_normalized * cfg.train_width
                y = y_normalized * cfg.train_height
                
                lane_points.append((int(x), int(y)))
            
            # Only add lane if it has enough points
            if len(lane_points) >= 2:
                batch_lanes.append(lane_points)
        
        all_lanes.append(batch_lanes)
    
    return all_lanes


def visualize_predictions(image, lanes, cfg, gt_lanes=None):
    """
    Visualize lane predictions on image
    
    Args:
        image: Input image tensor [3, H, W] or numpy array [H, W, 3]
        lanes: List of predicted lanes, each lane is list of (x, y) points
        cfg: Configuration object
        gt_lanes: Optional ground truth lanes for comparison
    
    Returns:
        Visualization image as numpy array [H, W, 3]
    """
    # Convert image to numpy if tensor
    if torch.is_tensor(image):
        # Denormalize
        mean = np.array(cfg.mean).reshape(3, 1, 1)
        std = np.array(cfg.std).reshape(3, 1, 1)
        image = image.cpu().numpy()
        image = image * std + mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        image = image.transpose(1, 2, 0)  # [H, W, 3]
    
    # Convert RGB to BGR for OpenCV
    if image.shape[2] == 3:
        vis_img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    else:
        vis_img = image.copy()
    
    # Define colors for lanes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Cyan
    ]
    
    # Draw ground truth lanes (if provided) as dashed lines
    if gt_lanes is not None:
        for lane_idx, lane in enumerate(gt_lanes):
            if len(lane) < 2:
                continue
            color = tuple([c // 2 for c in colors[lane_idx % len(colors)]])  # Darker colors for GT
            
            for i in range(len(lane) - 1):
                x1, y1 = lane[i]
                x2, y2 = lane[i + 1]
                
                # Draw dashed line
                draw_dashed_line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
    
    # Draw predicted lanes
    for lane_idx, lane in enumerate(lanes):
        if len(lane) < 2:
            continue
        
        color = colors[lane_idx % len(colors)]
        
        # Draw lane line
        for i in range(len(lane) - 1):
            x1, y1 = lane[i]
            x2, y2 = lane[i + 1]
            cv2.line(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=3)
        
        # Draw points
        for x, y in lane:
            cv2.circle(vis_img, (int(x), int(y)), 4, color, -1)
    
    # Add legend
    legend_y = 30
    if gt_lanes is not None:
        cv2.putText(vis_img, "Predicted (solid)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        cv2.putText(vis_img, "Ground Truth (dashed)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_img


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed line"""
    dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    dashes = int(dist / dash_length)
    
    if dashes == 0:
        return
    
    for i in range(0, dashes, 2):
        start = i / dashes
        end = min((i + 1) / dashes, 1.0)
        
        x1 = int(pt1[0] + (pt2[0] - pt1[0]) * start)
        y1 = int(pt1[1] + (pt2[1] - pt1[1]) * start)
        x2 = int(pt1[0] + (pt2[0] - pt1[0]) * end)
        y2 = int(pt1[1] + (pt2[1] - pt1[1]) * end)
        
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def visualize_segmentation(seg_output, cfg):
    """
    Visualize segmentation output
    
    Args:
        seg_output: Segmentation logits [num_lanes+1, H, W]
        cfg: Configuration object
    
    Returns:
        Colored segmentation map as numpy array [H, W, 3]
    """
    if torch.is_tensor(seg_output):
        seg_output = seg_output.cpu().numpy()
    
    # Get predicted class for each pixel
    seg_pred = np.argmax(seg_output, axis=0)  # [H, W]
    
    # Create colored segmentation map
    h, w = seg_pred.shape
    seg_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Define colors for each lane class
    colors = [
        [0, 0, 0],        # Background - Black
        [255, 0, 0],      # Lane 1 - Red
        [0, 255, 0],      # Lane 2 - Green
        [0, 0, 255],      # Lane 3 - Blue
        [255, 255, 0],    # Lane 4 - Yellow
    ]
    
    for class_id in range(min(cfg.num_lanes + 1, len(colors))):
        mask = seg_pred == class_id
        seg_vis[mask] = colors[class_id]
    
    return seg_vis


def save_visualization(vis_img, save_path):
    """
    Save visualization image
    
    Args:
        vis_img: Visualization image (BGR format)
        save_path: Path to save image
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_img)


def plot_training_curves(train_losses, val_metrics, save_path):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses per epoch
        val_metrics: List of validation metrics per epoch
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    axes[0].plot(train_losses, label='Train Loss')
    if val_metrics:
        val_losses = [m['loss'] for m in val_metrics]
        val_epochs = [m['epoch'] for m in val_metrics]
        axes[0].plot(val_epochs, val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training/Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot validation accuracy
    if val_metrics:
        val_accuracies = [m['accuracy'] * 100 for m in val_metrics]
        val_epochs = [m['epoch'] for m in val_metrics]
        axes[1].plot(val_epochs, val_accuracies, label='Accuracy', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # Test visualization
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from configs.tusimple_config import Config
    
    cfg = Config()
    
    # Create dummy data
    image = np.random.randint(0, 255, (cfg.train_height, cfg.train_width, 3), dtype=np.uint8)
    
    # Create dummy lanes
    lanes = [
        [(200 + i*2, 160 + i*10) for i in range(30)],
        [(400 + i*2, 160 + i*10) for i in range(30)],
    ]
    
    gt_lanes = [
        [(205 + i*2, 160 + i*10) for i in range(30)],
        [(405 + i*2, 160 + i*10) for i in range(30)],
    ]
    
    # Visualize
    vis_img = visualize_predictions(image, lanes, cfg, gt_lanes)
    
    print(f"Visualization shape: {vis_img.shape}")
    print("Test passed!")
