"""
Object Relocation Module for Scene Manipulation

This module handles the relocation of objects within scenes, including
object removal, destination planning, and seamless re-insertion.
"""

import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
from ..segmentation import SegmentationMask
from ..text_parser import SpatialReference
from .spatial_planner import SpatialPlanner
from .compositor import Compositor


class RelocationResult:
    """Container for object relocation results."""
    
    def __init__(self, original_image: np.ndarray, modified_image: np.ndarray,
                 object_mask: SegmentationMask, destination_mask: SegmentationMask,
                 intermediate_steps: List[Dict[str, Any]] = None):
        self.original_image = original_image
        self.modified_image = modified_image
        self.object_mask = object_mask
        self.destination_mask = destination_mask
        self.intermediate_steps = intermediate_steps or []
    
    def get_comparison_image(self) -> np.ndarray:
        """Create a side-by-side comparison of original and modified images."""
        h, w = self.original_image.shape[:2]
        
        # Create comparison image
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = self.original_image
        comparison[:, w:] = self.modified_image
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Modified", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return comparison


class ObjectRelocator:
    """
    Main class for relocating objects within scenes.
    
    This class coordinates the entire relocation process, including
    object detection, destination planning, removal, and re-insertion.
    """
    
    def __init__(self, inpainting_model=None):
        """
        Initialize the object relocator.
        
        Args:
            inpainting_model: Model for filling holes after object removal
        """
        self.spatial_planner = SpatialPlanner()
        self.compositor = Compositor()
        self.inpainting_model = inpainting_model
    
    def relocate_object(self, image: np.ndarray, object_mask: SegmentationMask,
                       destination: SpatialReference, 
                       preserve_scale: bool = True,
                       preserve_perspective: bool = True) -> RelocationResult:
        """
        Relocate an object to a new position in the scene.
        
        Args:
            image: Input image
            object_mask: Mask of the object to relocate
            destination: Target spatial reference
            preserve_scale: Whether to preserve object scale
            preserve_perspective: Whether to preserve perspective
        
        Returns:
            RelocationResult with original and modified images
        """
        intermediate_steps = []
        
        # Step 1: Extract object from original location
        extracted_object = self._extract_object(image, object_mask)
        intermediate_steps.append({
            "step": "object_extraction",
            "description": "Extracted object from original location",
            "data": extracted_object
        })
        
        # Step 2: Plan destination location
        destination_mask = self.spatial_planner.plan_destination(
            image, object_mask, destination
        )
        intermediate_steps.append({
            "step": "destination_planning",
            "description": f"Planned destination at {destination.value}",
            "data": destination_mask
        })
        
        # Step 3: Remove object from original location
        image_without_object = self._remove_object(image, object_mask)
        intermediate_steps.append({
            "step": "object_removal",
            "description": "Removed object from original location",
            "data": image_without_object
        })
        
        # Step 4: Transform object for new location
        transformed_object = self._transform_object(
            extracted_object, object_mask, destination_mask,
            preserve_scale, preserve_perspective
        )
        intermediate_steps.append({
            "step": "object_transformation",
            "description": "Transformed object for new location",
            "data": transformed_object
        })
        
        # Step 5: Insert object at new location
        final_image = self.compositor.composite_object(
            image_without_object, transformed_object, destination_mask
        )
        intermediate_steps.append({
            "step": "object_insertion",
            "description": "Inserted object at new location",
            "data": final_image
        })
        
        return RelocationResult(
            original_image=image,
            modified_image=final_image,
            object_mask=object_mask,
            destination_mask=destination_mask,
            intermediate_steps=intermediate_steps
        )
    
    def _extract_object(self, image: np.ndarray, mask: SegmentationMask) -> Dict[str, Any]:
        """
        Extract object from image using the mask.
        
        Args:
            image: Input image
            mask: Object mask
            
        Returns:
            Dictionary containing extracted object data
        """
        # Create object image with alpha channel
        object_image = image.copy()
        
        # Set background to transparent
        alpha_channel = np.zeros(image.shape[:2], dtype=np.uint8)
        alpha_channel[mask.mask] = 255
        
        # Create RGBA image
        rgba_image = np.dstack([object_image, alpha_channel])
        
        # Get bounding box
        bbox = mask.get_bbox()
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                         for i, coord in enumerate(bbox)]
        
        # Crop to object bounds
        cropped_object = rgba_image[y1:y2, x1:x2]
        cropped_mask = mask.mask[y1:y2, x1:x2]
        
        return {
            "image": cropped_object,
            "mask": cropped_mask,
            "bbox": bbox,
            "original_size": (y2 - y1, x2 - x1)
        }
    
    def _remove_object(self, image: np.ndarray, mask: SegmentationMask) -> np.ndarray:
        """
        Remove object from image using inpainting.
        
        Args:
            image: Input image
            mask: Object mask to remove
            
        Returns:
            Image with object removed
        """
        if self.inpainting_model is not None:
            # Use learned inpainting model
            return self._inpaint_with_model(image, mask)
        else:
            # Use traditional inpainting
            return self._inpaint_traditional(image, mask)
    
    def _inpaint_traditional(self, image: np.ndarray, mask: SegmentationMask) -> np.ndarray:
        """Use traditional OpenCV inpainting."""
        # Convert mask to uint8
        mask_uint8 = mask.mask.astype(np.uint8) * 255
        
        # Use TELEA algorithm for inpainting
        result = cv2.inpaint(image, mask_uint8, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def _inpaint_with_model(self, image: np.ndarray, mask: SegmentationMask) -> np.ndarray:
        """Use learned inpainting model (placeholder for future implementation)."""
        # This would integrate with a diffusion-based inpainting model
        # For now, fall back to traditional inpainting
        return self._inpaint_traditional(image, mask)
    
    def _transform_object(self, extracted_object: Dict[str, Any], 
                         original_mask: SegmentationMask,
                         destination_mask: SegmentationMask,
                         preserve_scale: bool, preserve_perspective: bool) -> Dict[str, Any]:
        """
        Transform object for insertion at new location.
        
        Args:
            extracted_object: Extracted object data
            original_mask: Original object mask
            destination_mask: Destination mask
            preserve_scale: Whether to preserve scale
            preserve_perspective: Whether to preserve perspective
            
        Returns:
            Transformed object data
        """
        object_image = extracted_object["image"]
        object_mask = extracted_object["mask"]
        
        # Get destination bounds
        dest_bbox = destination_mask.get_bbox()
        h, w = object_image.shape[:2]
        dest_h, dest_w = int(dest_bbox[3] * h) - int(dest_bbox[1] * h), \
                        int(dest_bbox[2] * w) - int(dest_bbox[0] * w)
        
        # Resize object to fit destination
        if preserve_scale:
            # Maintain aspect ratio
            aspect_ratio = object_image.shape[1] / object_image.shape[0]
            if dest_w / dest_h > aspect_ratio:
                new_h = dest_h
                new_w = int(dest_h * aspect_ratio)
            else:
                new_w = dest_w
                new_h = int(dest_w / aspect_ratio)
        else:
            new_h, new_w = dest_h, dest_w
        
        # Resize object
        resized_object = cv2.resize(object_image, (new_w, new_h), 
                                  interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(object_mask.astype(np.uint8), (new_w, new_h), 
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Apply perspective transformation if needed
        if preserve_perspective:
            # This would implement perspective-aware transformation
            # For now, use simple resizing
            pass
        
        return {
            "image": resized_object,
            "mask": resized_mask,
            "size": (new_h, new_w),
            "destination_bbox": dest_bbox
        }
    
    def batch_relocate(self, image: np.ndarray, 
                      relocations: List[Tuple[SegmentationMask, SpatialReference]]) -> RelocationResult:
        """
        Relocate multiple objects in a single operation.
        
        Args:
            image: Input image
            relocations: List of (mask, destination) tuples
            
        Returns:
            RelocationResult with all objects relocated
        """
        current_image = image.copy()
        intermediate_steps = []
        
        for i, (mask, destination) in enumerate(relocations):
            # Relocate each object
            result = self.relocate_object(current_image, mask, destination)
            current_image = result.modified_image
            intermediate_steps.extend(result.intermediate_steps)
        
        # Use the last object's mask for the result
        final_mask = relocations[-1][0] if relocations else SegmentationMask(
            np.zeros(image.shape[:2], dtype=bool)
        )
        
        return RelocationResult(
            original_image=image,
            modified_image=current_image,
            object_mask=final_mask,
            destination_mask=final_mask,  # This should be updated for multi-object
            intermediate_steps=intermediate_steps
        )
    
    def validate_relocation(self, result: RelocationResult) -> Tuple[bool, List[str]]:
        """
        Validate the quality of relocation result.
        
        Args:
            result: RelocationResult to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for obvious artifacts
        if self._has_obvious_artifacts(result.modified_image):
            issues.append("Obvious visual artifacts detected")
        
        # Check object integrity
        if self._object_integrity_lost(result):
            issues.append("Object integrity may be compromised")
        
        # Check spatial consistency
        if not self._spatial_consistency_check(result):
            issues.append("Spatial consistency issues detected")
        
        return len(issues) == 0, issues
    
    def _has_obvious_artifacts(self, image: np.ndarray) -> bool:
        """Check for obvious visual artifacts in the image."""
        # Simple edge detection to find potential artifacts
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # High edge density might indicate artifacts
        return edge_ratio > 0.1
    
    def _object_integrity_lost(self, result: RelocationResult) -> bool:
        """Check if object integrity is lost during relocation."""
        # Compare object areas before and after
        original_area = result.object_mask.get_area()
        
        # This is a simplified check - in practice, you'd want more sophisticated analysis
        return original_area < 10  # Very small objects might indicate issues
    
    def _spatial_consistency_check(self, result: RelocationResult) -> bool:
        """Check spatial consistency of the relocation."""
        # Check if object is within image bounds
        bbox = result.object_mask.get_bbox()
        return all(0 <= coord <= 1 for coord in bbox)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create a test mask (circular object)
    h, w = test_image.shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 3, w // 3
    radius = 50
    test_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    mask = SegmentationMask(test_mask, confidence=0.9, label="test_object")
    
    # Initialize relocator
    relocator = ObjectRelocator()
    
    # Relocate object
    result = relocator.relocate_object(
        test_image, mask, SpatialReference.RIGHT
    )
    
    # Validate result
    is_valid, issues = relocator.validate_relocation(result)
    print(f"Relocation valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(result.original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(result.object_mask.mask, cmap='gray')
    plt.title("Object Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result.modified_image)
    plt.title("Relocated Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 