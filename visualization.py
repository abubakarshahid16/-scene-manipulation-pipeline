"""
Visualization Utilities for Scene Manipulation Pipeline

This module provides utilities for visualizing pipeline results,
intermediate steps, and creating comprehensive output displays.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional, Tuple
from ..segmentation import SegmentationMask


class VisualizationUtils:
    """Utility class for creating visualizations of pipeline results."""
    
    @staticmethod
    def create_pipeline_comparison(original: np.ndarray, final: np.ndarray,
                                 title: str = "Pipeline Results") -> np.ndarray:
        """
        Create a side-by-side comparison of original and final images.
        
        Args:
            original: Original image
            final: Final processed image
            title: Title for the comparison
            
        Returns:
            Comparison image
        """
        h, w = original.shape[:2]
        
        # Create comparison image
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original
        comparison[:, w:] = final
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Modified", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Add title
        cv2.putText(comparison, title, (w - 100, h - 20), font, 0.7, (255, 255, 255), 2)
        
        return comparison
    
    @staticmethod
    def visualize_object_masks(image: np.ndarray, masks: List[SegmentationMask],
                             colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        Visualize multiple object masks on an image.
        
        Args:
            image: Input image
            masks: List of segmentation masks
            colors: Optional list of colors for each mask
            
        Returns:
            Image with mask overlays
        """
        if colors is None:
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 165, 0),  # Orange
                (128, 0, 128),  # Purple
            ]
        
        result = image.copy()
        
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            
            # Create colored overlay
            overlay = result.copy()
            overlay[mask.mask] = color
            
            # Blend with original image
            alpha = 0.5
            result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
            
            # Draw bounding box
            bbox = mask.get_bbox()
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                             for i, coord in enumerate(bbox)]
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{mask.label}: {mask.confidence:.2f}"
            cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result
    
    @staticmethod
    def create_step_visualization(steps: List[Dict[str, Any]], 
                                image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Create a visualization of pipeline steps.
        
        Args:
            steps: List of step dictionaries with images and descriptions
            image_size: Size of each step image
            
        Returns:
            Grid visualization of all steps
        """
        if not steps:
            return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        
        # Calculate grid layout
        n_steps = len(steps)
        cols = min(4, n_steps)  # Max 4 columns
        rows = (n_steps + cols - 1) // cols
        
        # Create grid image
        grid_h = rows * image_size[0]
        grid_w = cols * image_size[1]
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i, step in enumerate(steps):
            row = i // cols
            col = i % cols
            
            # Get step image
            if 'image' in step:
                step_image = step['image']
                if step_image.shape[:2] != image_size:
                    step_image = cv2.resize(step_image, image_size)
            else:
                step_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            
            # Place in grid
            y1 = row * image_size[0]
            y2 = (row + 1) * image_size[0]
            x1 = col * image_size[1]
            x2 = (col + 1) * image_size[1]
            
            grid[y1:y2, x1:x2] = step_image
            
            # Add step label
            label = f"Step {i+1}: {step.get('description', 'Unknown')}"
            cv2.putText(grid, label, (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return grid
    
    @staticmethod
    def create_lighting_comparison(original: np.ndarray, 
                                 lighting_results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create a comparison of different lighting effects.
        
        Args:
            original: Original image
            lighting_results: List of lighting result dictionaries
            
        Returns:
            Grid comparison of lighting effects
        """
        if not lighting_results:
            return original
        
        # Calculate grid layout
        n_results = len(lighting_results) + 1  # +1 for original
        cols = min(3, n_results)
        rows = (n_results + cols - 1) // cols
        
        # Get image size
        h, w = original.shape[:2]
        
        # Create grid
        grid_h = rows * h
        grid_w = cols * w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # Add original image
        grid[:h, :w] = original
        cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add lighting results
        for i, result in enumerate(lighting_results):
            idx = i + 1  # +1 because we already added original
            row = idx // cols
            col = idx % cols
            
            y1 = row * h
            y2 = (row + 1) * h
            x1 = col * w
            x2 = (col + 1) * w
            
            # Get result image
            if 'transformed_image' in result:
                result_image = result['transformed_image']
            elif 'image' in result:
                result_image = result['image']
            else:
                result_image = original
            
            grid[y1:y2, x1:x2] = result_image
            
            # Add label
            lighting_type = result.get('lighting_type', 'Unknown')
            label = f"{lighting_type}"
            cv2.putText(grid, label, (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return grid
    
    @staticmethod
    def create_bbox_visualization(image: np.ndarray, 
                                detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Visualize object detections with bounding boxes.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with bounding box visualizations
        """
        result = image.copy()
        
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]
        
        for i, detection in enumerate(detections):
            color = colors[i % len(colors)]
            
            # Get bounding box
            if 'bbox' in detection:
                bbox = detection['bbox']
                h, w = image.shape[:2]
                x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                                 for i, coord in enumerate(bbox)]
                
                # Draw bounding box
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = detection.get('label', 'Unknown')
                confidence = detection.get('confidence', 0.0)
                label_text = f"{label}: {confidence:.2f}"
                
                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw label background
                cv2.rectangle(result, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(result, label_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    @staticmethod
    def create_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray,
                             alpha: float = 0.6) -> np.ndarray:
        """
        Overlay a heatmap on an image.
        
        Args:
            image: Input image
            heatmap: Heatmap array (same size as image)
            alpha: Transparency of heatmap overlay
            
        Returns:
            Image with heatmap overlay
        """
        # Normalize heatmap to [0, 1]
        heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Apply colormap
        heatmap_colored = plt.cm.jet(heatmap_norm)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Resize heatmap to match image if needed
        if heatmap_colored.shape[:2] != image.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return result
    
    @staticmethod
    def create_animation_frames(original: np.ndarray, final: np.ndarray,
                              num_frames: int = 10) -> List[np.ndarray]:
        """
        Create animation frames between original and final images.
        
        Args:
            original: Original image
            final: Final image
            num_frames: Number of intermediate frames
            
        Returns:
            List of animation frames
        """
        frames = []
        
        for i in range(num_frames + 1):
            alpha = i / num_frames
            frame = cv2.addWeighted(original, 1 - alpha, final, alpha, 0)
            frames.append(frame)
        
        return frames
    
    @staticmethod
    def save_visualization_grid(images: List[np.ndarray], 
                              labels: List[str],
                              output_path: str,
                              grid_size: Tuple[int, int] = None) -> bool:
        """
        Save a grid of images as a single visualization.
        
        Args:
            images: List of images to display
            labels: List of labels for each image
            output_path: Output file path
            grid_size: Optional (rows, cols) for grid layout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not images:
                return False
            
            # Determine grid size
            if grid_size is None:
                n_images = len(images)
                cols = min(4, n_images)
                rows = (n_images + cols - 1) // cols
            else:
                rows, cols = grid_size
            
            # Get image size
            h, w = images[0].shape[:2]
            
            # Create grid
            grid_h = rows * h
            grid_w = cols * w
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            
            # Place images in grid
            for i, (image, label) in enumerate(zip(images, labels)):
                row = i // cols
                col = i % cols
                
                y1 = row * h
                y2 = (row + 1) * h
                x1 = col * w
                x2 = (col + 1) * w
                
                # Resize image if needed
                if image.shape[:2] != (h, w):
                    image_resized = cv2.resize(image, (w, h))
                else:
                    image_resized = image
                
                grid[y1:y2, x1:x2] = image_resized
                
                # Add label
                cv2.putText(grid, label, (x1 + 10, y1 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save grid
            cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            return True
            
        except Exception as e:
            print(f"Error saving visualization grid: {e}")
            return False
    
    @staticmethod
    def create_matplotlib_visualization(images: List[np.ndarray],
                                      titles: List[str],
                                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create a matplotlib-based visualization.
        
        Args:
            images: List of images to display
            titles: List of titles for each image
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_images = len(images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (image, title) in enumerate(zip(images, titles)):
            row = i // cols
            col = i % cols
            
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_title(title)
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Create test images
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create test masks
    mask1 = np.zeros((480, 640), dtype=bool)
    mask1[100:200, 100:300] = True
    
    mask2 = np.zeros((480, 640), dtype=bool)
    mask2[300:400, 400:600] = True
    
    masks = [
        SegmentationMask(mask1, confidence=0.9, label="Object 1"),
        SegmentationMask(mask2, confidence=0.8, label="Object 2")
    ]
    
    # Test visualizations
    viz = VisualizationUtils()
    
    # Visualize masks
    mask_viz = viz.visualize_object_masks(test_image, masks)
    
    # Create comparison
    modified_image = test_image.copy()
    modified_image[mask1] = [255, 0, 0]  # Red overlay
    comparison = viz.create_pipeline_comparison(test_image, modified_image)
    
    # Save results
    cv2.imwrite("outputs/mask_visualization.png", 
                cv2.cvtColor(mask_viz, cv2.COLOR_RGB2BGR))
    cv2.imwrite("outputs/comparison.png", 
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    print("Visualization examples saved to outputs/") 