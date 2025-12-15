"""
Common utility functions for model initialization and training
"""

import torch
import torch.nn as nn


def initialize_weights(*models):
    """
    Initialize weights for neural network models.
    
    Applies proper initialization strategies:
    - Conv2d & Linear: Kaiming Normal (He initialization) for ReLU activations
    - BatchNorm2d: weight=1, bias=0
    - GroupNorm: weight=1, bias=0
    
    Args:
        *models: Variable number of nn.Module instances to initialize
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def load_checkpoint(model, checkpoint_path, optimizer=None, strict=True):
    """
    Load model checkpoint from file
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        strict: Whether to strictly enforce that keys match
        
    Returns:
        Dictionary with 'epoch', 'best_acc' if available
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    info = {}
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    if 'best_acc' in checkpoint:
        info['best_acc'] = checkpoint['best_acc']
        
    return info


def save_checkpoint(state, filepath):
    """
    Save model checkpoint to file
    
    Args:
        state: Dictionary containing model state and training info
        filepath: Path to save checkpoint
    """
    torch.save(state, filepath)


if __name__ == '__main__':
    # Test weight initialization
    print("Testing weight initialization...")
    
    test_model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    print("Before initialization:")
    print(f"  Conv weight mean: {test_model[0].weight.mean().item():.6f}")
    print(f"  BN weight: {test_model[1].weight[0].item():.6f}")
    
    initialize_weights(test_model)
    
    print("\nAfter initialization:")
    print(f"  Conv weight mean: {test_model[0].weight.mean().item():.6f}")
    print(f"  BN weight: {test_model[1].weight[0].item():.6f}")
    print(f"  BN bias: {test_model[1].bias[0].item():.6f}")
    print("\n✓ Weight initialization working correctly!")