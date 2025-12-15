"""
Main Training Script for TuSimple Lane Detection
Optimized for Jetson Nano deployment with ResNet18 backbone
"""

import os
import sys
import torch
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.tusimple_config import Config
from data.tusimple_dataset import get_dataloaders
from model.model_tusimple import get_model as TusimpleLaneDetection
from train import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train UFLD on TuSimple Dataset')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load configuration
    cfg = Config()
    
    # Override config with command line arguments
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    
    print("\n" + "="*60)
    print("Ultra-Fast Lane Detection Training")
    print("="*60)
    print(cfg)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\nGPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        print("\nWarning: No GPU available, training on CPU")
        print("This will be very slow. Consider using a GPU.")
    
    # Create dataloaders
    print("\n" + "-"*60)
    print("Loading Dataset...")
    print("-"*60)
    
    try:
        train_loader, val_loader = get_dataloaders(cfg)
        print(f"✓ Train batches: {len(train_loader)}")
        if val_loader:
            print(f"✓ Val batches: {len(val_loader)}")
        else:
            print("✓ No validation set (val_split=0)")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print(f"  1. train_gt.txt exists at: {os.path.join(cfg.data_root, cfg.train_gt_file)}")
        print(f"  2. Image files are accessible from data_root: {cfg.data_root}")
        return
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create model
    print("\n" + "-"*60)
    print("Initializing Model...")
    print("-"*60)
    
    try:
        model = TusimpleLaneDetection(cfg)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model: TuSimple UFLD with ResNet{cfg.backbone}")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Pretrained backbone: {cfg.pretrained}")
        print(f"✓ Auxiliary segmentation: {cfg.use_aux}")
    except Exception as e:
        print(f"\n❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create trainer
    print("\n" + "-"*60)
    print("Initializing Trainer...")
    print("-"*60)
    
    try:
        trainer = Trainer(model, train_loader, val_loader, cfg)
        print("✓ Trainer initialized")
        print(f"✓ Optimizer: {trainer.optimizer.__class__.__name__}")
        print(f"✓ Scheduler: {cfg.scheduler}")
        print(f"✓ Mixed precision: {cfg.mixed_precision and torch.cuda.is_available()}")
        print(f"✓ Save directory: {cfg.save_dir}")
    except Exception as e:
        print(f"\n❌ Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        try:
            trainer.load_checkpoint(args.resume)
            print("✓ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return
    
    # Start training
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    print(f"Device: {trainer.device}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("="*60 + "\n")
    
    try:
        trainer.train()
        
        # Training completed successfully
        print("\n" + "="*60)
        print("✓ Training Completed Successfully!")
        print("="*60)
        print(f"Best accuracy: {trainer.best_accuracy*100:.2f}%")
        print(f"Model saved to: {cfg.save_dir}")
        print(f"Best model: {os.path.join(cfg.save_dir, 'best_model.pth')}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Training interrupted by user")
        print("="*60)
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print("✓ Checkpoint saved")
        
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving checkpoint before exit...")
        try:
            trainer.save_checkpoint()
            print("✓ Checkpoint saved")
        except:
            print("❌ Failed to save checkpoint")


if __name__ == '__main__':
    main()
