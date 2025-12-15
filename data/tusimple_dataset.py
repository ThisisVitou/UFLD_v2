"""
TuSimple Dataset Loader for UFLD Training
Reads train_gt.txt and converts to grid-based targets
"""

import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from data.transform import TrainTransform, ValTransform
from data.target_generator import TargetGenerator


class TuSimpleDataset(Dataset):
    """
    TuSimple Lane Detection Dataset
    
    Loads images and annotations from train_gt.txt file
    Each line format: <image_path> <json_annotation>
    """
    
    def __init__(self, cfg, mode='train'):
        """
        Args:
            cfg: Configuration object
            mode: 'train' or 'val'
        """
        self.cfg = cfg
        self.mode = mode
        self.data_root = cfg.data_root
        self.data_set = cfg.dataset
        
        # Load annotations
        gt_file = os.path.join(cfg.data_root, cfg.train_gt_file)
        self.samples = self._load_annotations(gt_file)
        
        # Split train/val if needed
        if mode == 'val':
            val_size = int(len(self.samples) * cfg.val_split)
            self.samples = self.samples[-val_size:]
        else:  # train
            val_size = int(len(self.samples) * cfg.val_split)
            self.samples = self.samples[:-val_size] if val_size > 0 else self.samples
        
        # Initialize transforms and target generator
        if mode == 'train':
            self.transform = TrainTransform(cfg)
        else:
            self.transform = ValTransform(cfg)
        
        self.target_generator = TargetGenerator(cfg)
        
        print(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def _load_annotations(self, gt_file):
        """
        Load annotations from train_gt.txt
        
        Format: <image_path> <json_annotation>
        Example: clips/0313-1/6040/20.jpg {"lanes": [...], "h_samples": [...], "raw_file": "..."}
        """
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
        
        samples = []
        with open(gt_file, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Split by first space to separate path and json
                    parts = line.split(' ', 1)
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line {line_idx}: {line}")
                        continue
                    
                    img_path, json_str = parts
                    annotation = json.loads(json_str)
                    
                    # Validate annotation
                    if 'lanes' not in annotation or 'h_samples' not in annotation:
                        print(f"Warning: Missing 'lanes' or 'h_samples' in line {line_idx}")
                        continue
                    
                    # Full image path
                    full_img_path = os.path.join(self.data_set, img_path)
                    
                    # Check if image exists
                    if not os.path.exists(full_img_path):
                        print(f"Warning: Image not found: {full_img_path}")
                        continue
                    
                    samples.append({
                        'img_path': full_img_path,
                        'lanes': annotation['lanes'],
                        'h_samples': annotation['h_samples']
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_idx}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_idx}: {e}")
                    continue
        
        if len(samples) == 0:
            raise ValueError(f"No valid samples found in {gt_file}")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            Dictionary containing:
                - image: Tensor [3, H, W]
                - targets: Dictionary with loc_row, loc_col, exist_row, exist_col, seg_mask
                - img_path: String (for debugging)
        """
        sample = self.samples[idx]
        
        # Load image
        img = cv2.imread(sample['img_path'])
        if img is None:
            raise ValueError(f"Failed to load image: {sample['img_path']}")
        
        # Get lanes and h_samples
        lanes = sample['lanes']
        h_samples = sample['h_samples']
        
        # Apply transformations
        img, lanes, h_samples = self.transform(img, lanes, h_samples)
        
        # Generate targets
        targets = self.target_generator.generate_targets(lanes, h_samples)
        
        return {
            'image': img,
            'targets': targets,
            'img_path': sample['img_path']
        }
    
    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader
        
        Stacks images and targets into batches
        """
        images = torch.stack([item['image'] for item in batch])
        img_paths = [item['img_path'] for item in batch]
        
        # Stack targets
        targets = {
            'loc_row': torch.stack([item['targets']['loc_row'] for item in batch]),
            'loc_col': torch.stack([item['targets']['loc_col'] for item in batch]),
            'exist_row': torch.stack([item['targets']['exist_row'] for item in batch]),
            'exist_col': torch.stack([item['targets']['exist_col'] for item in batch])
        }
        
        # Stack segmentation masks if available
        if 'seg_mask' in batch[0]['targets']:
            targets['seg_mask'] = torch.stack([item['targets']['seg_mask'] for item in batch])
        
        return {
            'images': images,
            'targets': targets,
            'img_paths': img_paths
        }


def get_dataloaders(cfg):
    """
    Create train and validation dataloaders
    
    Args:
        cfg: Configuration object
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = TuSimpleDataset(cfg, mode='train')
    val_dataset = TuSimpleDataset(cfg, mode='val') if cfg.val_split > 0 else None
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=train_dataset.collate_fn,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=val_dataset.collate_fn,
            drop_last=False
        )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset loading
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from configs.tusimple_config import Config
    
    cfg = Config()
    
    print("Creating dataset...")
    dataset = TuSimpleDataset(cfg, mode='train')
    
    print(f"\nDataset size: {len(dataset)}")
    
    print("\nLoading first sample...")
    sample = dataset[0]
    
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Image path: {sample['img_path']}")
    print(f"  Targets keys: {list(sample['targets'].keys())}")
    print(f"  loc_row shape: {sample['targets']['loc_row'].shape}")
    print(f"  exist_row shape: {sample['targets']['exist_row'].shape}")
    
    print("\nTesting dataloader...")
    train_loader, val_loader = get_dataloaders(cfg)
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")
    
    print("\nLoading first batch...")
    batch = next(iter(train_loader))
    print(f"  Batch images shape: {batch['images'].shape}")
    print(f"  Batch targets keys: {list(batch['targets'].keys())}")
