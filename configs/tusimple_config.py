"""
Configuration for TuSimple Dataset Training
Optimized for Jetson Nano deployment with ResNet18 backbone
"""

class Config:
    def __init__(self):
        # ============ Dataset Configuration ============
        # self.dataset = '/home/sswaterlab/Documents/Norakvitou/archive/TUSimple/train_set'  # Path to TuSimple dataset
        # self.data_root = '/home/sswaterlab/Documents/Norakvitou/UFLD_v2'  # Root directory
        self.dataset = 'D://code//dataset//tusimple//train_set'  # Path to TuSimple dataset
        self.data_root = 'D://code//2nd_ufld_lane_detection'  # Root directory
        self.train_gt_file = 'train_gt.txt'  # Ground truth file
        self.num_lanes = 4  # Maximum number of lanes in TuSimple
        
        # ============ Model Configuration ============
        self.backbone = '18'  # ResNet18 - lightweight for Jetson Nano
        self.pretrained = True  # Use ImageNet pretrained weights
        self.use_aux = True  # Enable auxiliary segmentation head
        self.fc_norm = False  # Use LayerNorm before FC layers
        
        # Grid-based anchor configuration (standard UFLD settings for TuSimple)
        self.num_cell_row = 100  # Number of row grid cells (vertical anchors)
        self.num_row = 56        # Number of row classifications per grid
        self.num_cell_col = 100  # Number of column grid cells (horizontal anchors)
        self.num_col = 41        # Number of column classifications per grid
        
        # ============ Input Configuration ============
        # Jetson Nano optimized: 800x288 (can reduce to 640x360 if memory constrained)
        self.train_width = 800
        self.train_height = 288
        self.original_width = 1280  # TuSimple original image width
        self.original_height = 720  # TuSimple original image height
        
        # ============ Training Configuration ============
        self.batch_size = 32  # Reduce to 4 if OOM on Jetson Nano
        self.epochs = 100
        self.learning_rate = 0.05
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.optimizer = 'sgd'  # Options: 'sgd', 'adam', 'adamw'
        
        # Learning rate scheduler
        self.scheduler = 'multi'  # Options: 'cosine', 'step', 'poly'
        self.steps = [50, 75]
        self.gamma = 0.1
        self.warmup = 'linear'
        self.warmup_iters = 100
        # self.warmup_epochs = 5
        # self.min_lr = 1e-6
        
        # ============ Loss Weights ============
        self.loss_weights = {
            'loc': 1.0,      # Location classification loss
            'exist': 0.1,    # Existence classification loss
            'seg': 1.0,      # Segmentation loss (if use_aux=True)
            'relation': 0.0,      # Start at 0, can enable later
            'relation_dis': 0.00,   # Start at 0, can enable later
            'mean_loss': 0.05
        }
        
        # ============ Data Augmentation ============
        self.aug_config = {
            'horizontal_flip': 0.5,    # Probability of horizontal flip
            'rotation': 6,              # Random rotation range in degrees
            'brightness': 0.3,          # Brightness jitter
            'contrast': 0.3,            # Contrast jitter
            'saturation': 0.3,          # Saturation jitter
            'hue': 0.1                  # Hue jitter
        }
        
        # ImageNet normalization (for pretrained backbone)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        # ============ Data Loading ============
        self.num_workers = 4  # Reduce to 2 for Jetson Nano
        self.pin_memory = True
        
        # ============ Validation Configuration ============
        self.val_split = 0.1  # Use 10% of data for validation
        self.val_interval = 1  # Validate every N epochs
        
        # ============ Checkpoint Configuration ============
        self.save_dir = 'experiments/tusimple_resnet18'
        self.save_interval = 10  # Save checkpoint every N epochs
        self.keep_last_n = 5     # Keep only last N checkpoints
        
        # ============ Logging Configuration ============
        self.log_interval = 20  # Log every N iterations
        self.vis_interval = 100  # Visualize every N iterations
        
        # ============ TuSimple Specific ============
        # TuSimple uses 56 h_samples from y=160 to y=710
        self.h_samples = list(range(160, 711, 10))
        
        # Lane matching threshold for evaluation
        self.match_threshold = 0.9
        
        # ============ Jetson Nano Optimizations ============
        self.mixed_precision = True  # Enable FP16 training if supported
        self.gradient_clip = 1.0     # Gradient clipping for stability
        
    def __repr__(self):
        config_str = f"\n{'='*60}\n"
        config_str += "TuSimple Training Configuration (Jetson Nano Optimized)\n"
        config_str += f"{'='*60}\n"
        config_str += f"Model: ResNet{self.backbone} (pretrained={self.pretrained})\n"
        config_str += f"Input: {self.train_width}×{self.train_height}\n"
        config_str += f"Grid: {self.num_cell_row}×{self.num_cell_col} cells\n"
        config_str += f"Batch Size: {self.batch_size}\n"
        config_str += f"Learning Rate: {self.learning_rate}\n"
        config_str += f"Epochs: {self.epochs}\n"
        config_str += f"Auxiliary Segmentation: {self.use_aux}\n"
        config_str += f"{'='*60}\n"
        return config_str


# Create default config instance
cfg = Config()


if __name__ == '__main__':
    # Print configuration
    config = Config()
    print(config)
    
    # Verify configuration
    print("\nConfiguration Verification:")
    print(f"✓ Backbone: ResNet{config.backbone}")
    print(f"✓ Input size suitable for Jetson Nano: {config.train_width}x{config.train_height}")
    print(f"✓ Grid cells: Row={config.num_cell_row}, Col={config.num_cell_col}")
    print(f"✓ Batch size: {config.batch_size} (reduce if OOM)")
    print(f"✓ Data augmentation enabled: {len(config.aug_config)} transforms")
    print(f"✓ Mixed precision: {config.mixed_precision}")
