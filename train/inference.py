"""
Test/Inference Script for TuSimple Lane Detection
Evaluate trained model on test set or single images
"""

import os
import sys
import argparse
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Fix: Add parent directory to path (project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.tusimple_config import Config
from model.model_tusimple import get_model
from data.tusimple_dataset import TuSimpleDataset
from train.evaluator import TuSimpleEvaluator
from utils.visualization import decode_predictions, visualize_predictions, save_visualization


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test UFLD on TuSimple Dataset')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-file', type=str, default=None,
                       help='Path to test annotation file (default: uses val split)')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Directory of images for batch inference')
    parser.add_argument('--output-dir', type=str, default='test_outputs',
                       help='Directory to save visualization results')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for testing')
    parser.add_argument('--save-viz', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--compute-metrics', action='store_true',
                       help='Compute TuSimple accuracy metrics')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU ID to use')
    
    return parser.parse_args()


def load_model(checkpoint_path, cfg):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model = get_model(cfg)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Trainer saves with 'model_state_dict'
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        acc = checkpoint.get('best_accuracy', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch} with accuracy {acc}")
    elif 'model' in checkpoint:
        # Some checkpoints use 'model'
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.get('epoch', 'unknown')
        acc = checkpoint.get('best_acc', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch} with accuracy {acc}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (no metadata)")   
    
    model = model.cuda()
    model.eval()
    
    return model


def test_single_image(model, image_path, cfg, output_path=None):
    """Test model on a single image"""
    print(f"\nProcessing: {image_path}")
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    original_image = image.copy()
    h, w = image.shape[:2]
    
    # Resize to model input size
    image = cv2.resize(image, (cfg.train_width, cfg.train_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # Convert to tensor [1, 3, H, W]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
    
    # Inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # ========== ADD DEBUG CODE HERE ==========
    print("\n" + "="*60)
    print("DEBUG: Prediction Statistics")
    print("="*60)
    print(f"loc_row shape: {predictions['loc_row'].shape}")
    print(f"loc_row range: [{predictions['loc_row'].min():.2f}, {predictions['loc_row'].max():.2f}]")
    print(f"loc_row mean: {predictions['loc_row'].mean():.2f}")
    print(f"\nexist_row shape: {predictions['exist_row'].shape}")
    
    if predictions['exist_row'].dim() == 4:
        exist_probs = torch.softmax(predictions['exist_row'], dim=1)[:, 1]
        print(f"exist_prob range: [{exist_probs.min():.2f}, {exist_probs.max():.2f}]")
        print(f"exist_prob mean: {exist_probs.mean():.2f}")
        print(f"High confidence predictions (>0.5): {(exist_probs > 0.5).sum().item()}")
    
    # Check for invalid values
    invalid_count = (predictions['loc_row'] < -1e4).sum().item()
    print(f"\nInvalid location predictions: {invalid_count}")
    print("="*60 + "\n")
    # ========== END DEBUG CODE ==========
    
    # Decode predictions
    pred_dict = {
        'loc_row': predictions['loc_row'],
        'exist_row': predictions['exist_row']
    }
    lanes = decode_predictions(pred_dict, cfg)

    
    # ========== ADD MORE DEBUG HERE ==========
    print(f"Decoded {len(lanes[0])} lanes:")
    for i, lane in enumerate(lanes[0]):
        print(f"  Lane {i}: {len(lane)} points")
        if len(lane) > 0:
            xs = [p[0] for p in lane]
            ys = [p[1] for p in lane]
            print(f"    X range: [{min(xs)}, {max(xs)}]")
            print(f"    Y range: [{min(ys)}, {max(ys)}]")
    print()
    # ========== END DEBUG CODE ==========
    
    # Scale lanes back to original image size
    scaled_lanes = []
    for lane in lanes[0]:  # First batch
        scaled_lane = [(int(x * w / cfg.train_width), int(y * h / cfg.train_height)) 
                      for x, y in lane]
        scaled_lanes.append(scaled_lane)
    
    # Visualize
    vis_image = original_image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for lane_idx, lane in enumerate(scaled_lanes):
        color = colors[lane_idx % len(colors)]
        for i in range(len(lane) - 1):
            cv2.line(vis_image, lane[i], lane[i+1], color, 3)
    
    # Save visualization
    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"✓ Saved visualization to: {output_path}")
    
    # Display results
    print(f"✓ Detected {len(scaled_lanes)} lanes")
    
    return vis_image, scaled_lanes


def test_image_directory(model, image_dir, cfg, output_dir):
    """Test model on all images in a directory"""
    print(f"\nProcessing images from: {image_dir}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
    
    if len(image_paths) == 0:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        output_path = os.path.join(output_dir, f"pred_{image_path.name}")
        test_single_image(model, str(image_path), cfg, output_path)
    
    print(f"\n✓ Saved results to: {output_dir}")


def test_dataset(model, cfg, test_file=None, save_viz=False, compute_metrics=False, output_dir='test_outputs'):
    """Test model on TuSimple test/validation set"""
    print("\n" + "="*60)
    print("Testing on TuSimple Dataset")
    print("="*60)
    
    # Create dataset
    dataset = TuSimpleDataset(
        cfg=cfg,
        mode='val'
    )
    
    print(f"Test samples: {len(dataset)}")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create evaluator
    if compute_metrics:
        evaluator = TuSimpleEvaluator(cfg)
    
    # Create output directory
    if save_viz:
        os.makedirs(output_dir, exist_ok=True)
    
    # Test loop
    all_predictions = []
    all_ground_truths = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            images = batch['image'].cuda()
            
            # Forward pass
            predictions = model(images)
            
            # Decode predictions
            pred_dict = {
                'loc_row': predictions['loc_row'],
                'exist_row': predictions['exist_row']
            }
            batch_lanes = decode_predictions(pred_dict, cfg)
            
            # Store for metrics
            if compute_metrics:
                all_predictions.extend(batch_lanes)
                # Get ground truth lanes from batch
                # You'll need to decode ground truth similarly
            
            # Save visualizations
            if save_viz and batch_idx < 50:  # Save first 50 batches
                for i in range(images.shape[0]):
                    image_np = images[i].cpu().numpy().transpose(1, 2, 0)
                    # Denormalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_np = image_np * std + mean
                    image_np = (image_np * 255).astype(np.uint8)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # Draw lanes
                    lanes = batch_lanes[i]
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                    for lane_idx, lane in enumerate(lanes):
                        color = colors[lane_idx % len(colors)]
                        for j in range(len(lane) - 1):
                            pt1 = (int(lane[j][0]), int(lane[j][1]))
                            pt2 = (int(lane[j+1][0]), int(lane[j+1][1]))
                            cv2.line(image_np, pt1, pt2, color, 2)
                    
                    # Save
                    output_path = os.path.join(output_dir, f"test_{batch_idx:04d}_{i:02d}.jpg")
                    cv2.imwrite(output_path, image_np)
    
    # Compute metrics
    if compute_metrics:
        print("\n" + "="*60)
        print("Computing Metrics...")
        print("="*60)
        
        # Note: You need ground truth to compute metrics
        # This is a placeholder - implement proper metric computation
        print("⚠ Metric computation requires ground truth annotations")
        print("   Implement TuSimpleEvaluator.evaluate() with ground truth")
    
    print("\n✓ Testing complete!")
    if save_viz:
        print(f"✓ Visualizations saved to: {output_dir}")


def main():
    """Main testing function"""
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU (slow)")
    
    # Load configuration
    cfg = Config()
    
    print("\n" + "="*60)
    print("Ultra-Fast Lane Detection - Testing")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*60)
    
    # Load model
    try:
        model = load_model(args.checkpoint, cfg)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test single image
    if args.image:
        output_path = os.path.join(args.output_dir, f"pred_{Path(args.image).name}")
        os.makedirs(args.output_dir, exist_ok=True)
        test_single_image(model, args.image, cfg, output_path)
    
    # Test image directory
    elif args.image_dir:
        test_image_directory(model, args.image_dir, cfg, args.output_dir)
    
    # Test on dataset
    else:
        test_dataset(
            model, 
            cfg, 
            test_file=args.test_file,
            save_viz=args.save_viz,
            compute_metrics=args.compute_metrics,
            output_dir=args.output_dir
        )
    
    print("\n✓ Testing complete!")


if __name__ == '__main__':
    main()
    # D:\code\2nd_ufld_lane_detection\experiments\tusimple_resnet18\latest.pth