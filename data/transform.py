"""
Data Augmentation and Transformation Pipeline
Includes geometric and color transformations with lane coordinate adjustments
"""

import cv2
import numpy as np
import torch
# from torchvision import transforms
import random


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TrainTransform:
    """
    Training data augmentation pipeline
    Applies transformations to both images and lane annotations
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_h = cfg.train_height
        self.img_w = cfg.train_width
        self.mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(cfg.std, dtype=np.float32).reshape(1, 1, 3)
        
        # Augmentation parameters
        self.h_flip_prob = cfg.aug_config.get('horizontal_flip', 0.5)
        self.rotation_deg = cfg.aug_config.get('rotation', 6)
        self.brightness = cfg.aug_config.get('brightness', 0.3)
        self.contrast = cfg.aug_config.get('contrast', 0.3)
        self.saturation = cfg.aug_config.get('saturation', 0.3)
        self.hue = cfg.aug_config.get('hue', 0.1)
        
    def __call__(self, img, lanes, h_samples):
        """
        Apply transformations to image and lanes
        
        Args:
            img: numpy array [H, W, 3] in BGR format
            lanes: list of lane annotations (x coordinates)
            h_samples: list of y coordinates
            
        Returns:
            Transformed image and lanes
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img, lanes, h_samples = self._resize(img, lanes, h_samples)
        
        # Random horizontal flip
        if random.random() < self.h_flip_prob:
            img, lanes = self._horizontal_flip(img, lanes)
        
        # Color jittering
        img = self._color_jitter(img)
        
        # Small rotation (disabled by default for lane detection)
        # Uncomment if needed, but be careful with lane coordinates
        # img, lanes, h_samples = self._rotate(img, lanes, h_samples)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # Convert to torch tensor [C, H, W]
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        return img, lanes, h_samples
    
    def _resize(self, img, lanes, h_samples):
        """Resize image and adjust lane coordinates"""
        h, w = img.shape[:2]
        scale_w = self.img_w / w
        scale_h = self.img_h / h
        
        # Resize image
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        
        # Scale lane coordinates
        lanes_scaled = []
        for lane in lanes:
            lane_scaled = [x * scale_w if x >= 0 else -2 for x in lane]
            lanes_scaled.append(lane_scaled)
        
        # Scale h_samples
        h_samples_scaled = [int(y * scale_h) for y in h_samples]
        
        return img, lanes_scaled, h_samples_scaled
    
    def _horizontal_flip(self, img, lanes):
        """Horizontal flip for image and lanes"""
        img = cv2.flip(img, 1)
        
        # Flip lane x coordinates
        lanes_flipped = []
        for lane in lanes:
            lane_flipped = [self.img_w - x if x >= 0 else -2 for x in lane]
            lanes_flipped.append(lane_flipped)
        
        # Reverse lane order (left becomes right)
        lanes_flipped = lanes_flipped[::-1]
        
        return img, lanes_flipped
    
    def _color_jitter(self, img):
        """Apply color jittering"""
        img = img.astype(np.float32)
        
        # Brightness
        if self.brightness > 0:
            delta = random.uniform(-self.brightness, self.brightness)
            img = img + delta * 255
        
        # Contrast
        if self.contrast > 0:
            alpha = random.uniform(1 - self.contrast, 1 + self.contrast)
            img = img * alpha
        
        # Saturation
        if self.saturation > 0:
            hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            alpha = random.uniform(1 - self.saturation, 1 + self.saturation)
            hsv[..., 1] = hsv[..., 1] * alpha
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # Hue
        if self.hue > 0:
            hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            delta = random.uniform(-self.hue, self.hue) * 180
            hsv[..., 0] = (hsv[..., 0] + delta) % 180
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def _rotate(self, img, lanes, h_samples):
        """
        Random rotation (use with caution for lane detection)
        This requires adjusting lane coordinates with rotation matrix
        """
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Rotate lane points
        lanes_rotated = []
        for lane in lanes:
            lane_rotated = []
            for x, y in zip(lane, h_samples):
                if x < 0:
                    lane_rotated.append(-2)
                else:
                    # Apply rotation matrix
                    point = np.array([x, y, 1])
                    new_point = M @ point
                    new_x = new_point[0]
                    
                    # Check if still within image bounds
                    if 0 <= new_x < w:
                        lane_rotated.append(new_x)
                    else:
                        lane_rotated.append(-2)
            lanes_rotated.append(lane_rotated)
        
        return img, lanes_rotated, h_samples


class ValTransform:
    """
    Validation/Test data transformation pipeline
    Only applies necessary preprocessing without augmentation
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_h = cfg.train_height
        self.img_w = cfg.train_width
        self.mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(cfg.std, dtype=np.float32).reshape(1, 1, 3)
        
    def __call__(self, img, lanes, h_samples):
        """
        Apply transformations to image and lanes
        
        Args:
            img: numpy array [H, W, 3] in BGR format
            lanes: list of lane annotations (x coordinates)
            h_samples: list of y coordinates
            
        Returns:
            Transformed image and lanes
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img, lanes, h_samples = self._resize(img, lanes, h_samples)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # Convert to torch tensor [C, H, W]
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        return img, lanes, h_samples
    
    def _resize(self, img, lanes, h_samples):
        """Resize image and adjust lane coordinates"""
        h, w = img.shape[:2]
        scale_w = self.img_w / w
        scale_h = self.img_h / h
        
        # Resize image
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        
        # Scale lane coordinates
        lanes_scaled = []
        for lane in lanes:
            lane_scaled = [x * scale_w if x >= 0 else -2 for x in lane]
            lanes_scaled.append(lane_scaled)
        
        # Scale h_samples
        h_samples_scaled = [int(y * scale_h) for y in h_samples]
        
        return img, lanes_scaled, h_samples_scaled


if __name__ == '__main__':
    # Test transforms
    from configs.tusimple_config import Config
    
    cfg = Config()
    train_transform = TrainTransform(cfg)
    val_transform = ValTransform(cfg)
    
    # Create dummy data
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    lanes = [[500 + i*10 for i in range(56)] for _ in range(4)]
    h_samples = list(range(160, 711, 10))
    
    print("Testing train transform...")
    img_train, lanes_train, h_samples_train = train_transform(img, lanes, h_samples)
    print(f"  Image shape: {img_train.shape}")
    print(f"  Lanes count: {len(lanes_train)}")
    print(f"  H samples: {len(h_samples_train)}")
    
    print("\nTesting val transform...")
    img_val, lanes_val, h_samples_val = val_transform(img, lanes, h_samples)
    print(f"  Image shape: {img_val.shape}")
    print(f"  Lanes count: {len(lanes_val)}")
    print(f"  H samples: {len(h_samples_val)}")
