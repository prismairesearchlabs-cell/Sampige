"""
Inference Script for SaMPiGe-Distill
Runs inference on images and saves predictions
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from PIL import Image
import cv2
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from datasets import KITTIDataset, get_transforms
from models import YOLOStudent, DINOv2Teacher, TeacherWrapper, KnowledgeDistiller
from utils import DetectionVisualizer, plot_predictions


class ModelInferencer:
    """
    Runs inference using trained SaMPiGe-Distill model
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25
    ):
        self.model_path = model_path
        self.class_names = class_names or [
            'Car', 'Van', 'Truck', 'Pedestrian', 
            'Person_sitting', 'Cyclist', 'Tram', 'Misc'
        ]
        self.device = torch.device(device or config.system.DEVICE)
        self.confidence_threshold = confidence_threshold
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize visualizer
        self.visualizer = DetectionVisualizer(self.class_names)
        
        # Transform
        self.transform = get_transforms(
            image_size=config.data.IMAGE_SIZE,
            augment=False
        )
        
        print(f"Inferencer initialized with model: {model_path}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        print(f"Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Initialize student model
        num_classes = len(self.class_names)
        model = YOLOStudent(num_classes=num_classes)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def predict(
        self,
        image: Union[str, np.ndarray, torch.Tensor, Image.Image],
        visualize: bool = True,
        save_path: Optional[str] = None,
        return_predictions: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Run prediction on a single image
        
        Args:
            image: Input image (path, numpy array, tensor, or PIL Image)
            visualize: Whether to visualize predictions
            save_path: Path to save visualization
            return_predictions: Whether to return predictions
        
        Returns:
            Dictionary of predictions (if return_predictions=True)
        """
        # Load and preprocess image
        processed_image, original_image = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(processed_image.unsqueeze(0).to(self.device))
        
        # Process outputs
        predictions = self._process_outputs(outputs, original_image.size)
        
        # Visualize if requested
        if visualize:
            self._visualize_predictions(
                original_image, predictions, save_path
            )
        
        if return_predictions:
            return {
                'image_path': image if isinstance(image, str) else None,
                'original_size': original_image.size,
                'predictions': predictions,
                'outputs': outputs
            }
        
        return None
    
    def batch_predict(
        self,
        images: List[Union[str, np.ndarray, torch.Tensor, Image.Image]],
        visualize: bool = False,
        output_dir: Optional[str] = None,
        return_predictions: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run prediction on a batch of images
        
        Args:
            images: List of input images
            visualize: Whether to visualize predictions
            output_dir: Directory to save visualizations
            return_predictions: Whether to return predictions
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, image in enumerate(images):
            if output_dir:
                save_path = os.path.join(output_dir, f"prediction_{i:04d}.png")
            else:
                save_path = None
            
            result = self.predict(
                image,
                visualize=visualize,
                save_path=save_path,
                return_predictions=return_predictions
            )
            
            if result:
                results.append(result)
        
        return results
    
    def _preprocess_image(
        self,
        image: Union[str, np.ndarray, torch.Tensor, Image.Image]
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocess image for inference
        
        Args:
            image: Input image
        
        Returns:
            Tuple of (processed tensor, original PIL Image)
        """
        # Load image if it's a path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Convert to PIL
            image = image.detach().cpu()
            if image.shape[0] == 3:  # CHW format
                image = image.permute(1, 2, 0)
            image = (image.numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Store original
        original_image = image.copy()
        
        # Apply transform
        processed_image = self.transform(image)
        
        return processed_image, original_image
    
    def _process_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Process model outputs to get predictions
        
        Args:
            outputs: Model outputs
            original_size: Original image size (width, height)
        
        Returns:
            Array of predictions in xyxy format with class and confidence
        """
        detections = outputs.get('detections', None)
        
        if detections is None:
            return np.zeros((0, 6))  # No predictions
        
        # Convert to numpy
        if isinstance(detections, torch.Tensor):
            detections = detections.detach().cpu().numpy()
        
        # Handle different output formats
        if detections.ndim == 4:
            # (B, C, H, W) format - need to decode
            # This is simplified - actual decoding depends on model
            detections = self._decode_detections(detections[0])
        elif detections.ndim == 3:
            # (B, num_preds, 5 + num_classes) format
            detections = detections[0]
            detections = self._decode_detections(detections)
        elif detections.ndim == 2:
            # (num_preds, 5 + num_classes) format
            detections = self._decode_detections(detections)
        
        # Scale predictions back to original size
        if len(detections) > 0:
            scale_x = original_size[0] / config.data.IMAGE_SIZE
            scale_y = original_size[1] / config.data.IMAGE_SIZE
            
            detections[:, 0] *= scale_x  # x1
            detections[:, 1] *= scale_y  # y1
            detections[:, 2] *= scale_x  # x2
            detections[:, 3] *= scale_y  # y2
        
        return detections
    
    def _decode_detections(self, detections: np.ndarray) -> np.ndarray:
        """
        Decode detections from model output format
        
        Args:
            detections: Raw detections from model
        
        Returns:
            Decoded predictions (N, 6) in xyxy + class + confidence format
        """
        # This is a simplified decoder
        # Actual implementation depends on the model output format
        
        if detections.shape[1] == 5 + len(self.class_names):
            # xyxy + class probabilities
            # Get class with highest probability
            class_probs = detections[:, 5:]
            class_ids = np.argmax(class_probs, axis=1)
            confidences = np.max(class_probs, axis=1)
            
            # Filter by confidence
            mask = confidences >= self.confidence_threshold
            
            predictions = np.zeros((mask.sum(), 6))
            predictions[:, :4] = detections[mask, :4]
            predictions[:, 4] = class_ids[mask]
            predictions[:, 5] = confidences[mask]
            
            return predictions
        
        elif detections.shape[1] == 6:
            # Already in xyxy + class + confidence format
            mask = detections[:, 5] >= self.confidence_threshold
            return detections[mask]
        
        else:
            # Unknown format - return empty
            warnings.warn(f"Unknown detection format: {detections.shape}")
            return np.zeros((0, 6))
    
    def _visualize_predictions(
        self,
        image: Image.Image,
        predictions: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize predictions on image
        
        Args:
            image: Original image
            predictions: Predictions in xyxy + class + confidence format
            save_path: Path to save visualization
        """
        # Convert predictions to format expected by visualizer
        # visualizer expects (N, 5) or (N, 6) where last column is optional confidence
        
        if predictions.shape[1] >= 6:
            # Already has confidence
            pass
        elif predictions.shape[1] == 5:
            # Add confidence column
            predictions = np.column_stack([predictions, np.ones(predictions.shape[0])])
        
        # Visualize
        fig = self.visualizer.visualize(
            image,
            predictions=predictions,
            confidence_threshold=self.confidence_threshold,
            show_scores=True
        )
        
        # Save if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)


def run_inference(
    model_path: str,
    input_path: Union[str, List[str]],
    output_dir: str = config.path.OUTPUT_DIR,
    visualize: bool = True,
    save_predictions: bool = True,
    confidence_threshold: float = 0.25
) -> List[Dict[str, Any]]:
    """
    Run inference on images
    
    Args:
        model_path: Path to model checkpoint
        input_path: Input image path or list of paths
        output_dir: Directory to save results
        visualize: Whether to visualize predictions
        save_predictions: Whether to save predictions
        confidence_threshold: Minimum confidence for predictions
    
    Returns:
        List of prediction results
    """
    # Initialize inferencer
    inferencer = ModelInferencer(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle input path
    if isinstance(input_path, str):
        if os.path.isdir(input_path):
            # Directory - process all images
            image_paths = [
                os.path.join(input_path, f) 
                for f in os.listdir(input_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        else:
            # Single file
            image_paths = [input_path]
    else:
        image_paths = input_path
    
    print(f"Running inference on {len(image_paths)} images...")
    
    # Run inference
    results = []
    for i, image_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Run prediction
        result = inferencer.predict(
            image_path,
            visualize=visualize,
            save_path=os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_pred.png") if visualize else None,
            return_predictions=save_predictions
        )
        
        if result and save_predictions:
            results.append(result)
    
    # Save results if requested
    if save_predictions and results:
        results_path = os.path.join(output_dir, "predictions.json")
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {
                'image_path': result['image_path'],
                'original_size': list(result['original_size']),
                'predictions': result['predictions'].tolist()
            }
            serializable_results.append(serializable_result)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Predictions saved to: {results_path}")
    
    return results


def create_inference_video(
    model_path: str,
    video_path: str,
    output_path: str,
    frame_skip: int = 1,
    confidence_threshold: float = 0.25
):
    """
    Create inference video from input video
    
    Args:
        model_path: Path to model checkpoint
        video_path: Path to input video
        output_path: Path to output video
        frame_skip: Process every Nth frame
        confidence_threshold: Minimum confidence for predictions
    """
    # Initialize inferencer
    inferencer = ModelInferencer(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )
    
    print(f"Processing video: {total_frames} frames...")
    
    frame_count = 0
    processed_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames
        if frame_count % frame_skip != 0:
            out.write(frame)
            continue
        # Process frame
        processed_count += 1
        print(f"Processing frame {frame_count}/{total_frames} ({processed_count})")
        
        # Convert to PIL
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run prediction
        result = inferencer.predict(
            frame_pil,
            visualize=False,
            return_predictions=True
        )
        
        if result:
            predictions = result['predictions']
            
            # Draw predictions on frame
            frame_with_preds = inferencer._draw_predictions_on_frame(
                frame, predictions
            )
            
            out.write(frame_with_preds)
        else:
            out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video processing complete: {processed_count} frames processed")
    print(f"Output saved to: {output_path}")


def _draw_predictions_on_frame(
    self,
    frame: np.ndarray,
    predictions: np.ndarray
) -> np.ndarray:
    """Draw predictions on video frame"""
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # Draw predictions
    draw = ImageDraw.Draw(frame_pil)
    
    for pred in predictions:
        x1, y1, x2, y2, class_id, confidence = pred
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        
        # Get color
        color = self.visualizer.colors[class_id % len(self.visualizer.colors)]
        
        # Draw rectangle
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=2
        )
        
        # Draw label
        label = self.class_names[class_id] if class_id < len(self.class_names) else f"{class_id}"
        label += f" {confidence:.2f}"
        
        # Draw text background
        text_width, text_height = draw.textsize(label)
        draw.rectangle(
            [(x1, y1 - text_height - 5), (x1 + text_width + 5, y1)],
            fill=color
        )
        
        # Draw text
        draw.text(
            (x1 + 2, y1 - text_height - 3),
            label,
            fill=(255, 255, 255)
        )
    
    # Convert back to BGR
    frame_with_preds = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    return frame_with_preds


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SaMPiGe-Distill Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image, directory, or video')
    parser.add_argument('--output_dir', type=str, default=config.path.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to JSON')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--video', action='store_true', help='Input is a video file')
    parser.add_argument('--frame_skip', type=int, default=1, help='Process every Nth frame (for video)')
    
    args = parser.parse_args()
    
    if args.video:
        # Process video
        output_path = os.path.join(args.output_dir, os.path.basename(args.input))
        create_inference_video(
            model_path=args.model_path,
            video_path=args.input,
            output_path=output_path,
            frame_skip=args.frame_skip,
            confidence_threshold=args.confidence
        )
    else:
        # Process images
        run_inference(
            model_path=args.model_path,
            input_path=args.input,
            output_dir=args.output_dir,
            visualize=args.visualize,
            save_predictions=args.save_predictions,
            confidence_threshold=args.confidence
        )
    
    print("Inference completed!")


if __name__ == "__main__":
    main()
