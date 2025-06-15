"""
Mask Processing Utilities for Scene Manipulation

This module provides utilities for processing, combining, and manipulating
segmentation masks for object relocation and scene manipulation.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from .segment_anything import SegmentationMask
import matplotlib.pyplot as plt


class MaskProcessor:
    """
    Utility class for processing and manipulating segmentation masks.
    
    This class provides methods for combining masks, filling holes,
    smoothing boundaries, and other mask operations needed for
    scene manipulation.
    """
    
    def __init__(self):
        """Initialize the mask processor."""
        pass
    
    def combine_masks(self, masks: List[SegmentationMask], 
                     operation: str = "union") -> SegmentationMask:
        """
        Combine multiple masks using specified operation.
        
        Args:
            masks: List of SegmentationMask objects
            operation: Operation type ("union", "intersection", "difference")
            
        Returns:
            Combined SegmentationMask
        """
        if not masks:
            raise ValueError("No masks provided")
        
        if len(masks) == 1:
            return masks[0]
        
        # Ensure all masks have the same size
        target_shape = masks[0].mask.shape
        processed_masks = []
        
        for mask in masks:
            if mask.mask.shape != target_shape:
                resized_mask = mask.resize(target_shape)
                processed_masks.append(resized_mask.mask)
            else:
                processed_masks.append(mask.mask)
        
        # Perform operation
        if operation == "union":
            combined_mask = np.logical_or.reduce(processed_masks)
        elif operation == "intersection":
            combined_mask = np.logical_and.reduce(processed_masks)
        elif operation == "difference":
            combined_mask = processed_masks[0]
            for mask in processed_masks[1:]:
                combined_mask = np.logical_and(combined_mask, np.logical_not(mask))
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Calculate average confidence
        avg_confidence = np.mean([mask.confidence for mask in masks])
        
        return SegmentationMask(combined_mask, confidence=avg_confidence)
    
    def fill_holes(self, mask: SegmentationMask, max_hole_size: int = 100) -> SegmentationMask:
        """
        Fill small holes in the mask.
        
        Args:
            mask: Input SegmentationMask
            max_hole_size: Maximum hole size to fill
            
        Returns:
            SegmentationMask with holes filled
        """
        filled_mask = mask.mask.copy()
        
        # Find contours
        contours, _ = cv2.findContours(mask.mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill small holes
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max_hole_size:
                cv2.fillPoly(filled_mask, [contour], 1)
        
        return SegmentationMask(filled_mask, mask.confidence, mask.bbox, mask.label)
    
    def smooth_boundaries(self, mask: SegmentationMask, 
                         kernel_size: int = 5, iterations: int = 1) -> SegmentationMask:
        """
        Smooth mask boundaries using morphological operations.
        
        Args:
            mask: Input SegmentationMask
            kernel_size: Size of morphological kernel
            iterations: Number of iterations
            
        Returns:
            Smoothed SegmentationMask
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close small gaps
        closed_mask = cv2.morphologyEx(mask.mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Open to remove small noise
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        return SegmentationMask(opened_mask.astype(bool), mask.confidence, mask.bbox, mask.label)
    
    def expand_mask(self, mask: SegmentationMask, 
                   expansion_pixels: int = 10) -> SegmentationMask:
        """
        Expand mask boundaries by specified number of pixels.
        
        Args:
            mask: Input SegmentationMask
            expansion_pixels: Number of pixels to expand
            
        Returns:
            Expanded SegmentationMask
        """
        kernel_size = 2 * expansion_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        expanded_mask = cv2.dilate(mask.mask.astype(np.uint8), kernel, iterations=1)
        
        return SegmentationMask(expanded_mask.astype(bool), mask.confidence, mask.bbox, mask.label)
    
    def shrink_mask(self, mask: SegmentationMask, 
                   shrink_pixels: int = 5) -> SegmentationMask:
        """
        Shrink mask boundaries by specified number of pixels.
        
        Args:
            mask: Input SegmentationMask
            shrink_pixels: Number of pixels to shrink
            
        Returns:
            Shrunk SegmentationMask
        """
        kernel_size = 2 * shrink_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        shrunk_mask = cv2.erode(mask.mask.astype(np.uint8), kernel, iterations=1)
        
        return SegmentationMask(shrunk_mask.astype(bool), mask.confidence, mask.bbox, mask.label)
    
    def create_boundary_mask(self, mask: SegmentationMask, 
                           boundary_width: int = 3) -> SegmentationMask:
        """
        Create a mask representing the boundary of the input mask.
        
        Args:
            mask: Input SegmentationMask
            boundary_width: Width of the boundary
            
        Returns:
            Boundary SegmentationMask
        """
        kernel_size = 2 * boundary_width + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate and erode to get boundary
        dilated = cv2.dilate(mask.mask.astype(np.uint8), kernel, iterations=1)
        eroded = cv2.erode(mask.mask.astype(np.uint8), kernel, iterations=1)
        
        boundary_mask = dilated - eroded
        
        return SegmentationMask(boundary_mask.astype(bool), mask.confidence, mask.bbox, mask.label)
    
    def remove_small_objects(self, mask: SegmentationMask, 
                           min_area: int = 100) -> SegmentationMask:
        """
        Remove small objects from the mask.
        
        Args:
            mask: Input SegmentationMask
            min_area: Minimum area to keep
            
        Returns:
            Cleaned SegmentationMask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.mask.astype(np.uint8), connectivity=8
        )
        
        # Create new mask keeping only large components
        cleaned_mask = np.zeros_like(mask.mask)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = True
        
        return SegmentationMask(cleaned_mask, mask.confidence, mask.bbox, mask.label)
    
    def extract_largest_component(self, mask: SegmentationMask) -> SegmentationMask:
        """
        Extract the largest connected component from the mask.
        
        Args:
            mask: Input SegmentationMask
            
        Returns:
            SegmentationMask with only the largest component
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.mask.astype(np.uint8), connectivity=8
        )
        
        if num_labels <= 1:
            return mask  # No components found
        
        # Find largest component
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        
        # Create mask with only largest component
        largest_mask = (labels == largest_label)
        
        return SegmentationMask(largest_mask, mask.confidence, mask.bbox, mask.label)
    
    def create_inpainting_mask(self, mask: SegmentationMask, 
                             feather_width: int = 10) -> SegmentationMask:
        """
        Create a mask suitable for inpainting with feathered edges.
        
        Args:
            mask: Input SegmentationMask
            feather_width: Width of feathering
            
        Returns:
            Feathered SegmentationMask for inpainting
        """
        # Create distance transform
        dist_transform = cv2.distanceTransform(mask.mask.astype(np.uint8), cv2.DIST_L2, 5)
        
        # Normalize and create feathered mask
        max_dist = np.max(dist_transform)
        if max_dist > 0:
            normalized_dist = dist_transform / max_dist
            feathered_mask = np.clip(normalized_dist / (feather_width / max_dist), 0, 1)
        else:
            feathered_mask = mask.mask.astype(np.float32)
        
        return SegmentationMask(feathered_mask > 0.5, mask.confidence, mask.bbox, mask.label)
    
    def calculate_mask_overlap(self, mask1: SegmentationMask, 
                             mask2: SegmentationMask) -> float:
        """
        Calculate overlap ratio between two masks.
        
        Args:
            mask1: First SegmentationMask
            mask2: Second SegmentationMask
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        # Ensure same size
        if mask1.mask.shape != mask2.mask.shape:
            mask2_resized = mask2.resize(mask1.mask.shape)
            mask2_mask = mask2_resized.mask
        else:
            mask2_mask = mask2.mask
        
        intersection = np.logical_and(mask1.mask, mask2_mask)
        union = np.logical_or(mask1.mask, mask2_mask)
        
        if np.sum(union) == 0:
            return 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def visualize_masks(self, image: np.ndarray, masks: List[SegmentationMask], 
                       save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize multiple masks on an image.
        
        Args:
            image: Input image
            masks: List of SegmentationMask objects
            save_path: Optional path to save visualization
            
        Returns:
            Image with masks visualized
        """
        vis_image = image.copy()
        
        # Define colors for different masks
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
        ]
        
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            
            # Create colored overlay
            overlay = vis_image.copy()
            overlay[mask.mask] = color
            
            # Blend with original image
            alpha = 0.5
            vis_image = cv2.addWeighted(vis_image, 1 - alpha, overlay, alpha, 0)
            
            # Draw bounding box
            if mask.bbox:
                h, w = image.shape[:2]
                x1, y1, x2, y2 = mask.bbox
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image


# Example usage
if __name__ == "__main__":
    # Create test masks
    h, w = 100, 100
    
    # Create a circular mask
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    radius = 30
    circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    mask1 = SegmentationMask(circle_mask, confidence=0.9, label="circle")
    
    # Create a rectangular mask
    rect_mask = np.zeros((h, w), dtype=bool)
    rect_mask[20:80, 20:80] = True
    
    mask2 = SegmentationMask(rect_mask, confidence=0.8, label="rectangle")
    
    # Initialize processor
    processor = MaskProcessor()
    
    # Test operations
    print("Original masks:")
    print(f"Circle area: {mask1.get_area()}")
    print(f"Rectangle area: {mask2.get_area()}")
    
    # Combine masks
    combined = processor.combine_masks([mask1, mask2], operation="union")
    print(f"Combined area: {combined.get_area()}")
    
    # Smooth boundaries
    smoothed = processor.smooth_boundaries(mask1)
    print(f"Smoothed area: {smoothed.get_area()}")
    
    # Expand mask
    expanded = processor.expand_mask(mask1, expansion_pixels=5)
    print(f"Expanded area: {expanded.get_area()}")
    
    # Calculate overlap
    overlap = processor.calculate_mask_overlap(mask1, mask2)
    print(f"Overlap ratio: {overlap:.3f}")
    
    # Visualize
    test_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    vis_image = processor.visualize_masks(test_image, [mask1, mask2])
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(mask1.mask, cmap='gray')
    plt.title("Circle Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask2.mask, cmap='gray')
    plt.title("Rectangle Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(combined.mask, cmap='gray')
    plt.title("Combined Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 