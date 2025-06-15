"""
Compositor Module for Object Insertion

This module handles the seamless composition of objects into new locations
with proper blending, lighting adjustments, and visual consistency.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional
from ..segmentation import SegmentationMask


class Compositor:
    """
    Handles seamless composition of objects into new locations.
    
    This class provides methods for blending objects into scenes with
    proper lighting, shadows, and visual consistency.
    """
    
    def __init__(self):
        """Initialize the compositor."""
        self.blend_modes = {
            'normal': self._blend_normal,
            'multiply': self._blend_multiply,
            'screen': self._blend_screen,
            'overlay': self._blend_overlay,
            'soft_light': self._blend_soft_light
        }
    
    def composite_object(self, background_image: np.ndarray, 
                        object_data: Dict[str, Any],
                        destination_mask: SegmentationMask,
                        blend_mode: str = 'normal',
                        adjust_lighting: bool = True,
                        add_shadows: bool = True) -> np.ndarray:
        """
        Composite an object into the background image.
        
        Args:
            background_image: Background image to insert object into
            object_data: Dictionary containing object image and mask
            destination_mask: Mask of destination area
            blend_mode: Blending mode to use
            adjust_lighting: Whether to adjust lighting for consistency
            add_shadows: Whether to add shadows for realism
            
        Returns:
            Composited image
        """
        result_image = background_image.copy()
        
        # Extract object information
        object_image = object_data['image']
        object_mask = object_data['mask']
        
        # Get destination bounds
        dest_bbox = destination_mask.get_bbox()
        h, w = background_image.shape[:2]
        x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                         for i, coord in enumerate(dest_bbox)]
        
        # Resize object to fit destination
        dest_h, dest_w = y2 - y1, x2 - x1
        if object_image.shape[:2] != (dest_h, dest_w):
            object_image = cv2.resize(object_image, (dest_w, dest_h), 
                                    interpolation=cv2.INTER_LANCZOS4)
            object_mask = cv2.resize(object_mask.astype(np.uint8), (dest_w, dest_h), 
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Adjust lighting if requested
        if adjust_lighting:
            object_image = self._adjust_lighting(object_image, background_image, x1, y1, x2, y2)
        
        # Create alpha mask for blending
        alpha_mask = self._create_alpha_mask(object_mask, object_image)
        
        # Blend object into background
        blend_func = self.blend_modes.get(blend_mode, self._blend_normal)
        result_image = blend_func(result_image, object_image, alpha_mask, x1, y1)
        
        # Add shadows if requested
        if add_shadows:
            result_image = self._add_shadows(result_image, object_mask, x1, y1, dest_w, dest_h)
        
        return result_image
    
    def _create_alpha_mask(self, object_mask: np.ndarray, 
                          object_image: np.ndarray) -> np.ndarray:
        """
        Create alpha mask for smooth blending.
        
        Args:
            object_mask: Binary object mask
            object_image: Object image (may have alpha channel)
            
        Returns:
            Alpha mask for blending
        """
        if object_image.shape[2] == 4:  # RGBA image
            alpha = object_image[:, :, 3] / 255.0
        else:
            alpha = object_mask.astype(np.float32)
        
        # Apply feathering for smooth edges
        alpha = self._feather_mask(alpha)
        
        return alpha
    
    def _feather_mask(self, mask: np.ndarray, feather_radius: int = 3) -> np.ndarray:
        """
        Apply feathering to mask for smooth edges.
        
        Args:
            mask: Input mask
            feather_radius: Radius of feathering
            
        Returns:
            Feathered mask
        """
        # Apply Gaussian blur for feathering
        feathered = cv2.GaussianBlur(mask, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
        return feathered
    
    def _adjust_lighting(self, object_image: np.ndarray, background_image: np.ndarray,
                        x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Adjust object lighting to match background.
        
        Args:
            object_image: Object image to adjust
            background_image: Background image for reference
            x1, y1, x2, y2: Destination bounds
            
        Returns:
            Lighting-adjusted object image
        """
        # Extract background region around destination
        bg_region = background_image[y1:y2, x1:x2]
        
        # Calculate lighting statistics
        bg_mean = np.mean(bg_region, axis=(0, 1))
        obj_mean = np.mean(object_image[:, :, :3], axis=(0, 1))
        
        # Calculate adjustment factors
        adjustment = bg_mean / (obj_mean + 1e-8)  # Avoid division by zero
        
        # Apply lighting adjustment
        adjusted = object_image.copy()
        adjusted[:, :, :3] = np.clip(adjusted[:, :, :3] * adjustment, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def _blend_normal(self, background: np.ndarray, foreground: np.ndarray,
                     alpha: np.ndarray, x: int, y: int) -> np.ndarray:
        """Normal blending mode."""
        result = background.copy()
        
        # Ensure alpha has correct shape
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=2)
        
        # Blend foreground and background
        fg_region = foreground[:, :, :3]
        bg_region = background[y:y+alpha.shape[0], x:x+alpha.shape[1]]
        
        blended = fg_region * alpha + bg_region * (1 - alpha)
        
        result[y:y+alpha.shape[0], x:x+alpha.shape[1]] = blended
        
        return result
    
    def _blend_multiply(self, background: np.ndarray, foreground: np.ndarray,
                       alpha: np.ndarray, x: int, y: int) -> np.ndarray:
        """Multiply blending mode."""
        result = background.copy()
        
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=2)
        
        fg_region = foreground[:, :, :3] / 255.0
        bg_region = background[y:y+alpha.shape[0], x:x+alpha.shape[1]] / 255.0
        
        blended = fg_region * bg_region * alpha + bg_region * (1 - alpha)
        
        result[y:y+alpha.shape[0], x:x+alpha.shape[1]] = (blended * 255).astype(np.uint8)
        
        return result
    
    def _blend_screen(self, background: np.ndarray, foreground: np.ndarray,
                     alpha: np.ndarray, x: int, y: int) -> np.ndarray:
        """Screen blending mode."""
        result = background.copy()
        
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=2)
        
        fg_region = foreground[:, :, :3] / 255.0
        bg_region = background[y:y+alpha.shape[0], x:x+alpha.shape[1]] / 255.0
        
        blended = (1 - (1 - fg_region) * (1 - bg_region)) * alpha + bg_region * (1 - alpha)
        
        result[y:y+alpha.shape[0], x:x+alpha.shape[1]] = (blended * 255).astype(np.uint8)
        
        return result
    
    def _blend_overlay(self, background: np.ndarray, foreground: np.ndarray,
                      alpha: np.ndarray, x: int, y: int) -> np.ndarray:
        """Overlay blending mode."""
        result = background.copy()
        
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=2)
        
        fg_region = foreground[:, :, :3] / 255.0
        bg_region = background[y:y+alpha.shape[0], x:x+alpha.shape[1]] / 255.0
        
        # Overlay formula
        overlay = np.where(bg_region < 0.5,
                          2 * fg_region * bg_region,
                          1 - 2 * (1 - fg_region) * (1 - bg_region))
        
        blended = overlay * alpha + bg_region * (1 - alpha)
        
        result[y:y+alpha.shape[0], x:x+alpha.shape[1]] = (blended * 255).astype(np.uint8)
        
        return result
    
    def _blend_soft_light(self, background: np.ndarray, foreground: np.ndarray,
                         alpha: np.ndarray, x: int, y: int) -> np.ndarray:
        """Soft light blending mode."""
        result = background.copy()
        
        if len(alpha.shape) == 2:
            alpha = np.stack([alpha] * 3, axis=2)
        
        fg_region = foreground[:, :, :3] / 255.0
        bg_region = background[y:y+alpha.shape[0], x:x+alpha.shape[1]] / 255.0
        
        # Soft light formula
        soft_light = np.where(fg_region < 0.5,
                             bg_region * (2 * fg_region + bg_region * (1 - 2 * fg_region)),
                             bg_region * (1 - 2 * fg_region) + np.sqrt(bg_region) * (2 * fg_region - 1))
        
        blended = soft_light * alpha + bg_region * (1 - alpha)
        
        result[y:y+alpha.shape[0], x:x+alpha.shape[1]] = (blended * 255).astype(np.uint8)
        
        return result
    
    def _add_shadows(self, image: np.ndarray, object_mask: np.ndarray,
                    x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Add realistic shadows to the composited object.
        
        Args:
            image: Input image
            object_mask: Object mask
            x, y: Object position
            w, h: Object dimensions
            
        Returns:
            Image with shadows added
        """
        result = image.copy()
        
        # Create shadow mask (offset and scaled version of object mask)
        shadow_offset = (10, 10)  # Shadow offset
        shadow_scale = 1.2  # Shadow size multiplier
        
        # Create shadow mask
        shadow_mask = np.zeros_like(object_mask)
        
        # Scale and offset the object mask
        scaled_mask = cv2.resize(object_mask.astype(np.uint8), 
                               (int(w * shadow_scale), int(h * shadow_scale)),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Calculate shadow position
        shadow_x = x + shadow_offset[0]
        shadow_y = y + shadow_offset[1]
        
        # Ensure shadow is within image bounds
        if (shadow_x >= 0 and shadow_y >= 0 and 
            shadow_x + scaled_mask.shape[1] <= image.shape[1] and
            shadow_y + scaled_mask.shape[0] <= image.shape[0]):
            
            # Apply shadow
            shadow_region = result[shadow_y:shadow_y+scaled_mask.shape[0], 
                                 shadow_x:shadow_x+scaled_mask.shape[1]]
            
            # Darken the shadow region
            shadow_alpha = 0.3  # Shadow opacity
            shadow_region[scaled_mask] = shadow_region[scaled_mask] * (1 - shadow_alpha)
            
            result[shadow_y:shadow_y+scaled_mask.shape[0], 
                  shadow_x:shadow_x+scaled_mask.shape[1]] = shadow_region
        
        return result
    
    def composite_multiple_objects(self, background_image: np.ndarray,
                                 object_list: List[Dict[str, Any]],
                                 blend_mode: str = 'normal') -> np.ndarray:
        """
        Composite multiple objects into the background.
        
        Args:
            background_image: Background image
            object_list: List of object data dictionaries
            blend_mode: Blending mode to use
            
        Returns:
            Composited image with all objects
        """
        result = background_image.copy()
        
        # Sort objects by depth (background to foreground)
        # This is a simplified approach - in practice, you'd have depth information
        sorted_objects = sorted(object_list, key=lambda x: x.get('depth', 0))
        
        for object_data in sorted_objects:
            # Create dummy destination mask (in practice, this would come from spatial planning)
            h, w = background_image.shape[:2]
            dest_mask = SegmentationMask(np.ones((h, w), dtype=bool))
            
            result = self.composite_object(
                result, object_data, dest_mask, blend_mode
            )
        
        return result
    
    def create_comparison_image(self, original: np.ndarray, composited: np.ndarray,
                              object_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a side-by-side comparison image.
        
        Args:
            original: Original image
            composited: Composited image
            object_mask: Optional object mask for highlighting
            
        Returns:
            Comparison image
        """
        h, w = original.shape[:2]
        
        # Create comparison image
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original
        comparison[:, w:] = composited
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Composited", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Highlight object if mask provided
        if object_mask is not None:
            # Add red outline to object in composited image
            contours, _ = cv2.findContours(object_mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(comparison[:, w:], [contour], -1, (0, 0, 255), 2)
        
        return comparison


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create test images
    background = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create a simple object (colored rectangle)
    object_image = np.zeros((100, 150, 4), dtype=np.uint8)
    object_image[:, :, 0] = 255  # Red
    object_image[:, :, 3] = 255  # Full alpha
    
    # Create object mask
    object_mask = np.ones((100, 150), dtype=bool)
    
    object_data = {
        'image': object_image,
        'mask': object_mask,
        'size': (100, 150)
    }
    
    # Create destination mask
    dest_mask = SegmentationMask(np.zeros((480, 640), dtype=bool))
    dest_mask.mask[200:300, 250:400] = True
    
    # Initialize compositor
    compositor = Compositor()
    
    # Test different blend modes
    blend_modes = ['normal', 'multiply', 'screen', 'overlay', 'soft_light']
    
    plt.figure(figsize=(20, 8))
    
    for i, blend_mode in enumerate(blend_modes):
        # Composite object
        result = compositor.composite_object(
            background, object_data, dest_mask, blend_mode
        )
        
        # Visualize
        plt.subplot(2, 3, i + 1)
        plt.imshow(result)
        plt.title(f"Blend Mode: {blend_mode}")
        plt.axis('off')
    
    # Show comparison
    plt.subplot(2, 3, 6)
    comparison = compositor.create_comparison_image(background, result, object_mask)
    plt.imshow(comparison)
    plt.title("Comparison")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 