"""
Video Inference Script for Lane Detection
Process video files or webcam stream with trained UFLD model
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.tusimple_config import Config
from model.model_tusimple import get_model
from utils.visualization import decode_predictions


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video Inference for Lane Detection')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video file')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam instead of video file')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID for webcam (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output video (if not specified, display only)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (1=all frames, 2=every other frame)')
    parser.add_argument('--display-fps', action='store_true',
                       help='Display FPS on video')
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
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        acc = checkpoint.get('best_accuracy', 'unknown')
        print(f"✓ Loaded checkpoint from epoch {epoch} with accuracy {acc}")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("✓ Loaded checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded checkpoint")
    
    model = model.cuda()
    model.eval()
    
    return model


def preprocess_frame(frame, cfg):
    """Preprocess frame for model input"""
    # Resize to model input size
    original_h, original_w = frame.shape[:2]
    resized = cv2.resize(frame, (cfg.train_width, cfg.train_height))
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Convert to tensor [1, 3, H, W]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    
    return img_tensor, (original_h, original_w)


def draw_lanes(frame, lanes, color_palette=None):
    """Draw lanes on frame"""
    if color_palette is None:
        color_palette = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Cyan
        ]
    
    # Draw each lane
    for lane_idx, lane in enumerate(lanes):
        if len(lane) < 2:
            continue
        
        color = color_palette[lane_idx % len(color_palette)]
        
        # Draw lane as polyline
        points = np.array(lane, dtype=np.int32)
        cv2.polylines(frame, [points], False, color, 3, cv2.LINE_AA)
        
        # Draw points
        for point in lane:
            cv2.circle(frame, tuple(point), 3, color, -1)
    
    return frame


def process_video(model, video_path, cfg, args):
    """Process video file or webcam stream"""
    # Open video capture
    if args.webcam:
        print(f"Opening webcam (ID: {args.camera_id})...")
        cap = cv2.VideoCapture(args.camera_id)
    else:
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video source")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    if not args.webcam:
        print(f"  Total frames: {total_frames}")
    
    # Setup video writer if output specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")
    
    # Processing loop
    frame_count = 0
    processed_count = 0
    fps_list = []
    
    print("\nProcessing video... (Press 'q' to quit)")
    print("="*60)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if frame_count % args.skip_frames != 0:
                if out:
                    out.write(frame)
                continue
            
            # Process frame
            start_time = time.time()
            
            # Preprocess
            img_tensor, (orig_h, orig_w) = preprocess_frame(frame, cfg)
            
            # Inference
            with torch.no_grad():
                predictions = model(img_tensor)
            
            # Decode predictions
            pred_dict = {
                'loc_row': predictions['loc_row'],
                'exist_row': predictions['exist_row']
            }
            batch_lanes = decode_predictions(pred_dict, cfg)
            lanes = batch_lanes[0]  # First batch element
            
            # Scale lanes to original frame size
            scaled_lanes = []
            for lane in lanes:
                scaled_lane = [
                    (int(x * orig_w / cfg.train_width), 
                     int(y * orig_h / cfg.train_height))
                    for x, y in lane
                ]
                scaled_lanes.append(scaled_lane)
            
            # Draw lanes
            output_frame = draw_lanes(frame.copy(), scaled_lanes)
            
            # Calculate FPS
            inference_time = time.time() - start_time
            current_fps = 1.0 / inference_time
            fps_list.append(current_fps)
            
            # Display FPS and info
            if args.display_fps:
                avg_fps = np.mean(fps_list[-30:])  # Last 30 frames
                cv2.putText(output_frame, f"FPS: {avg_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.putText(output_frame, f"Lanes: {len(scaled_lanes)}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Lane Detection', output_frame)
            
            # Write to output video
            if out:
                out.write(output_frame)
            
            processed_count += 1
            
            # Print progress
            if not args.webcam and processed_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_fps = np.mean(fps_list[-30:])
                print(f"Progress: {progress:.1f}% | "
                      f"Frame: {frame_count}/{total_frames} | "
                      f"FPS: {avg_fps:.1f}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping...")
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*60)
        print("Processing Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Processed frames: {processed_count}")
        if fps_list:
            print(f"  Average FPS: {np.mean(fps_list):.2f}")
            print(f"  Min FPS: {np.min(fps_list):.2f}")
            print(f"  Max FPS: {np.max(fps_list):.2f}")
        if args.output:
            print(f"  Output saved to: {args.output}")
        print("="*60)


def main():
    """Main function"""
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU (very slow for video)")
    
    # Validate input
    if not args.webcam and not args.video:
        print("❌ Error: Must specify either --video or --webcam")
        return
    
    if args.webcam and args.video:
        print("❌ Error: Cannot use both --video and --webcam")
        return
    
    # Load configuration
    cfg = Config()
    
    print("\n" + "="*60)
    print("Ultra-Fast Lane Detection - Video Inference")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if args.webcam:
        print(f"Input: Webcam (ID: {args.camera_id})")
    else:
        print(f"Input: {args.video}")
    print("="*60 + "\n")
    
    # Load model
    try:
        model = load_model(args.checkpoint, cfg)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Process video
    try:
        process_video(model, args.video, cfg, args)
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Video processing complete!")


if __name__ == '__main__':
    main()