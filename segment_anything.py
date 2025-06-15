"""
Segment Anything Model (SAM) Integration for Scene Manipulation

This module provides integration with Meta's Segment Anything Model (SAM)
for precise object segmentation and mask generation.
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Union
import sys
import os

# Add segment-anything to path if available
try:
    import segment_anything
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment-anything not available. Install with: pip install segment-anything")


class SegmentationMask:
    """Container for segmentation mask data."""
    
    def __init__(self, mask: np.ndarray, confidence: float = 1.0, 
                 bbox: Optional[List[float]] = None, label: str = "object"):
        self.mask = mask  # Binary mask (H, W)
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2] in normalized coordinates
        self.label = label
    
    def get_area(self) -> int:
        """Get the area of the mask in pixels."""
        return np.sum(self.mask)
    
    def get_center(self) -> Tuple[float, float]:
        """Get the center of mass of the mask."""
        if self.get_area() == 0:
            return (0.5, 0.5)  # Default to image center
        
        y_coords, x_coords = np.where(self.mask)
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)
        
        # Normalize to [0, 1]
        h, w = self.mask.shape
        return (center_x / w, center_y / h)
    
    def get_bbox(self) -> List[float]:
        """Get bounding box from mask."""
        if self.get_area() == 0:
            return [0.0, 0.0, 1.0, 1.0]
        
        y_coords, x_coords = np.where(self.mask)
        x1, x2 = np.min(x_coords), np.max(x_coords)
        y1, y2 = np.min(y_coords), np.max(y_coords)
        
        # Normalize to [0, 1]
        h, w = self.mask.shape
        return [x1/w, y1/h, x2/w, y2/h]
    
    def resize(self, target_size: Tuple[int, int]) -> 'SegmentationMask':
        """Resize the mask to target size."""
        resized_mask = cv2.resize(self.mask.astype(np.uint8), target_size, 
                                interpolation=cv2.INTER_NEAREST)
        return SegmentationMask(resized_mask.astype(bool), self.confidence, self.bbox, self.label)
    
    def dilate(self, kernel_size: int = 3) -> 'SegmentationMask':
        """Dilate the mask to expand boundaries."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(self.mask.astype(np.uint8), kernel, iterations=1)
        return SegmentationMask(dilated_mask.astype(bool), self.confidence, self.bbox, self.label)
    
    def erode(self, kernel_size: int = 3) -> 'SegmentationMask':
        """Erode the mask to shrink boundaries."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded_mask = cv2.erode(self.mask.astype(np.uint8), kernel, iterations=1)
        return SegmentationMask(eroded_mask.astype(bool), self.confidence, self.bbox, self.label)


class SAMSegmenter:
    """
    Segment Anything Model (SAM) wrapper for object segmentation.
    
    This class provides methods for generating precise segmentation masks
    using Meta's SAM model with various input types (points, boxes, masks).
    """
    
    def __init__(self, model_type: str = "vit_h", checkpoint_path: Optional[str] = None):
        """
        Initialize SAM segmenter.
        
        Args:
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            checkpoint_path: Path to SAM checkpoint file
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything is not available. Please install it first.")
        
        self.model_type = model_type
        
        # Set default checkpoint path if not provided
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path(model_type)
        
        # Load SAM model
        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.predictor = SamPredictor(self.sam)
            print(f"Loaded SAM model: {model_type}")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            print("SAM will not be available for segmentation.")
            self.sam = None
            self.predictor = None
    
    def _get_default_checkpoint_path(self, model_type: str) -> str:
        """Get default checkpoint path for SAM model."""
        checkpoint_dir = "models/sam"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_names = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        checkpoint_name = checkpoint_names.get(model_type, "sam_vit_h_4b8939.pth")
        return os.path.join(checkpoint_dir, checkpoint_name)
    
    def set_image(self, image: np.ndarray):
        """
        Set the image for segmentation.
        
        Args:
            image: Input image as numpy array (H, W, C)
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")
        
        self.predictor.set_image(image)
        self.image_shape = image.shape[:2]
    
    def segment_from_point(self, point: Tuple[int, int], label: int = 1) -> SegmentationMask:
        """
        Generate segmentation mask from a single point.
        
        Args:
            point: (x, y) coordinates of the point
            label: Point label (1 for foreground, 0 for background)
            
        Returns:
            SegmentationMask object
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")
        
        input_point = np.array([point])
        input_label = np.array([label])
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Select the best mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        return SegmentationMask(best_mask, confidence=best_score)
    
    def segment_from_box(self, bbox: List[float]) -> SegmentationMask:
        """
        Generate segmentation mask from bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates
            
        Returns:
            SegmentationMask object
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")
        
        input_box = np.array(bbox)
        
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Select the best mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        return SegmentationMask(best_mask, confidence=best_score, bbox=bbox)
    
    def segment_from_points_and_box(self, points: List[Tuple[int, int]], 
                                  labels: List[int], bbox: Optional[List[float]] = None) -> SegmentationMask:
        """
        Generate segmentation mask from points and optional bounding box.
        
        Args:
            points: List of (x, y) coordinates
            labels: List of point labels (1 for foreground, 0 for background)
            bbox: Optional [x1, y1, x2, y2] bounding box
            
        Returns:
            SegmentationMask object
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")
        
        input_point = np.array(points)
        input_label = np.array(labels)
        
        if bbox is not None:
            input_box = np.array(bbox)
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True
            )
        else:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
        
        # Select the best mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        return SegmentationMask(best_mask, confidence=best_score, bbox=bbox)
    
    def segment_from_mask(self, mask: np.ndarray) -> SegmentationMask:
        """
        Refine segmentation mask using SAM.
        
        Args:
            mask: Input binary mask
            
        Returns:
            Refined SegmentationMask object
        """
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")
        
        masks, scores, logits = self.predictor.predict(
            mask_input=mask,
            multimask_output=True
        )
        
        # Select the best mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        return SegmentationMask(best_mask, confidence=best_score)
    
    def segment_multiple_objects(self, image: np.ndarray, 
                               detection_results: List) -> List[SegmentationMask]:
        """
        Segment multiple objects from detection results.
        
        Args:
            image: Input image
            detection_results: List of detection results with bounding boxes
            
        Returns:
            List of SegmentationMask objects
        """
        self.set_image(image)
        masks = []
        
        for detection in detection_results:
            if hasattr(detection, 'bbox'):
                # Convert normalized bbox to pixel coordinates
                h, w = image.shape[:2]
                x1, y1, x2, y2 = detection.bbox
                bbox_pixels = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
                
                mask = self.segment_from_box(bbox_pixels)
                mask.label = getattr(detection, 'label', 'object')
                masks.append(mask)
        
        return masks
    
    def interactive_segmentation(self, image: np.ndarray, points: List[Tuple[int, int]], 
                               labels: List[int]) -> SegmentationMask:
        """
        Interactive segmentation with multiple points.
        
        Args:
            image: Input image
            points: List of (x, y) coordinates
            labels: List of point labels
            
        Returns:
            SegmentationMask object
        """
        self.set_image(image)
        return self.segment_from_points_and_box(points, labels)
    
    def refine_mask(self, mask: SegmentationMask, 
                   refinement_points: Optional[List[Tuple[int, int]]] = None) -> SegmentationMask:
        """
        Refine an existing mask with additional points.
        
        Args:
            mask: Existing segmentation mask
            refinement_points: Optional list of refinement points
            
        Returns:
            Refined SegmentationMask object
        """
        if refinement_points is None:
            return mask
        
        # Use the existing mask as input and add refinement points
        points = []
        labels = []
        
        for point, label in refinement_points:
            points.append(point)
            labels.append(label)
        
        return self.segment_from_points_and_box(points, labels, mask.bbox)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # Initialize SAM segmenter
        segmenter = SAMSegmenter()
        
        # Set image
        segmenter.set_image(test_image)
        
        # Segment from a point
        point = (320, 240)  # Center of image
        mask = segmenter.segment_from_point(point)
        
        print(f"Generated mask with area: {mask.get_area()} pixels")
        print(f"Mask confidence: {mask.confidence:.3f}")
        
        # Visualize result
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(test_image)
        plt.plot(point[0], point[1], 'ro', markersize=10)
        plt.title("Input Image with Point")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask.mask, cmap='gray')
        plt.title("Segmentation Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        overlay = test_image.copy()
        overlay[mask.mask] = [255, 0, 0]  # Red overlay
        plt.imshow(overlay)
        plt.title("Mask Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError as e:
        print(f"SAM not available: {e}")
        print("This is expected if segment-anything is not installed.") 