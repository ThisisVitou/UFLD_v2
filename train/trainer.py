"""
Trainer for Ultra-Fast Lane Detection
Handles training loop, validation, checkpointing, and logging
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import json

from .loss import UFLDLoss
from .evaluator import TuSimpleEvaluator


class Trainer:
    """
    Main trainer class for UFLD model
    """
    
    def __init__(self, model, train_loader, val_loader, cfg):
        """
        Args:
            model: UFLD model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            cfg: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Loss function
        self.criterion = UFLDLoss(cfg)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training (for Jetson Nano efficiency)
        self.scaler = torch.cuda.amp.GradScaler() if cfg.mixed_precision and torch.cuda.is_available() else None
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Create save directory
        os.makedirs(cfg.save_dir, exist_ok=True)
        
        # Save config
        self._save_config()
    
    def _create_optimizer(self):
        """Create optimizer"""
        if hasattr(self.cfg, 'optimizer') and self.cfg.optimizer == 'sgd':
            optimizer = SGD(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )
        else:
            optimizer = Adam(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay
            )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.cfg.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.epochs,
                eta_min=self.cfg.min_lr
            )
        elif self.cfg.scheduler == 'step':
            scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.cfg.scheduler == 'multi':
            from torch.optim.lr_scheduler import MultiStepLR
            scheduler = MultiStepLR(
                self.optimizer,
                milestones=list(self.cfg.steps[0]) if isinstance(self.cfg.steps, tuple) else self.cfg.steps,
                gamma=self.cfg.gamma[0] if isinstance(self.cfg.gamma, tuple) else self.cfg.gamma
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _save_config(self):
        """Save configuration to JSON"""
        config_dict = {
            'backbone': self.cfg.backbone,
            'num_lanes': self.cfg.num_lanes,
            'train_width': self.cfg.train_width,
            'train_height': self.cfg.train_height,
            'batch_size': self.cfg.batch_size,
            'learning_rate': self.cfg.learning_rate,
            'epochs': self.cfg.epochs,
            'num_cell_row': self.cfg.num_cell_row,
            'num_row': self.cfg.num_row,
            'num_cell_col': self.cfg.num_cell_col,
            'num_col': self.cfg.num_col,
            'use_aux': self.cfg.use_aux
        }
        
        config_path = os.path.join(self.cfg.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_losses = {'loc_loss': 0.0, 'exist_loss': 0.0, 'seg_loss': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.cfg.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    predictions = self.model(images)
                    
                    # Compute loss
                    loss_dict = self.criterion(predictions, targets)
                    loss = loss_dict['total_loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.cfg.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                # Gradient clipping
                if self.cfg.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                
                self.optimizer.step()
            
            # Debug verification on first iteration
            if batch_idx == 0 and self.current_epoch == 0:
                print("\n" + "="*60)
                print("DEBUG: First Iteration Verification")
                print("="*60)
                print(f"Model outputs:")
                print(f"  loc_row shape: {predictions['loc_row'].shape}")  # Should be [B, 100, 56, 4]
                print(f"  loc_row range: [{predictions['loc_row'].min():.2f}, {predictions['loc_row'].max():.2f}]")
                print(f"\nTargets:")
                print(f"  loc_row shape: {targets['loc_row'].shape}")  # Should be [B, 56, 4]
                print(f"  loc_row dtype: {targets['loc_row'].dtype}")  # Should be torch.int64
                print(f"  loc_row range: [{targets['loc_row'].min()}, {targets['loc_row'].max()}]")  # Should be [-1, 99]
                print(f"  exist_row range: [{targets['exist_row'].min()}, {targets['exist_row'].max()}]")
                print("="*60 + "\n")
            
            # Accumulate losses
            epoch_loss += loss.item()
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            
            # Update progress bar
            if (batch_idx + 1) % self.cfg.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})
        
        # Compute average losses
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Store training loss
        self.train_losses.append(avg_loss)
        
        return avg_loss, epoch_losses
    
    def validate(self):
        """Validate on validation set"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        evaluator = TuSimpleEvaluator(self.cfg)
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            
            for batch in pbar:
                # Move data to device
                images = batch['images'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                # Forward pass
                predictions = self.model(images)
                
                # Compute loss
                loss_dict = self.criterion(predictions, targets)
                val_loss += loss_dict['total_loss'].item()
                
                # Update evaluator
                evaluator.update(predictions, targets)
        
        # Compute metrics
        val_loss /= len(self.val_loader)
        metrics = evaluator.compute_metrics()
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  FP Rate: {metrics['fp']:.4f}")
        print(f"  FN Rate: {metrics['fn']:.4f}")
        
        # Store metrics
        self.val_metrics.append({
            'epoch': self.current_epoch,
            'loss': val_loss,
            **metrics
        })
        
        return metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        if (self.current_epoch + 1) % self.cfg.save_interval == 0:
            checkpoint_path = os.path.join(self.cfg.save_dir, f'checkpoint_epoch_{self.current_epoch + 1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.cfg.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Save latest
        latest_path = os.path.join(self.cfg.save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Remove old checkpoints (keep last N)
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keep only last N"""
        checkpoints = []
        for f in os.listdir(self.cfg.save_dir):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
                epoch = int(f.split('_')[-1].split('.')[0])
                checkpoints.append((epoch, f))
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x[0])
        
        # Remove old checkpoints
        if len(checkpoints) > self.cfg.keep_last_n:
            for epoch, filename in checkpoints[:-self.cfg.keep_last_n]:
                filepath = os.path.join(self.cfg.save_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(self.cfg)
        
        for epoch in range(self.current_epoch, self.cfg.epochs):
            self.current_epoch = epoch
            
            start_time = time.time()
            
            # Train one epoch
            train_loss, train_losses = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.cfg.val_interval == 0 and self.val_loader is not None:
                metrics = self.validate()
                
                # Check if best model
                is_best = metrics['accuracy'] > self.best_accuracy
                if is_best:
                    self.best_accuracy = metrics['accuracy']
                    print(f"New best accuracy: {self.best_accuracy*100:.2f}%")
            else:
                is_best = False
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            for key, value in train_losses.items():
                if value > 0:
                    print(f"  {key}: {value:.4f}")
            print()
        
        print("="*60)
        print("Training Completed!")
        print(f"Best Validation Accuracy: {self.best_accuracy*100:.2f}%")
        print("="*60)


if __name__ == '__main__':
    print("Trainer module - use train_tusimple.py to run training")
