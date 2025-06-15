"""
Spatial Planning Module for Object Relocation

This module handles the planning of optimal locations for object placement
based on spatial references and scene analysis.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from ..segmentation import SegmentationMask
from ..text_parser import SpatialReference


class SpatialPlanner:
    """
    Plans optimal locations for object placement based on spatial references.
    
    This class analyzes the scene to find suitable locations for objects
    based on spatial constraints and scene understanding.
    """
    
    def __init__(self):
        """Initialize the spatial planner."""
        self.obstacle_threshold = 0.3  # Threshold for considering area as obstacle
        self.safety_margin = 0.1  # Safety margin around obstacles
    
    def plan_destination(self, image: np.ndarray, object_mask: SegmentationMask,
                        destination: SpatialReference) -> SegmentationMask:
        """
        Plan destination location for object relocation.
        
        Args:
            image: Input image
            object_mask: Mask of object to be relocated
            destination: Target spatial reference
            
        Returns:
            SegmentationMask representing the destination area
        """
        h, w = image.shape[:2]
        
        # Analyze scene structure
        scene_analysis = self._analyze_scene(image)
        
        # Get object dimensions
        object_bbox = object_mask.get_bbox()
        object_width = object_bbox[2] - object_bbox[0]
        object_height = object_bbox[3] - object_bbox[1]
        
        # Calculate destination based on spatial reference
        if destination == SpatialReference.LEFT:
            dest_bbox = self._plan_left_location(image, object_width, object_height, scene_analysis)
        elif destination == SpatialReference.RIGHT:
            dest_bbox = self._plan_right_location(image, object_width, object_height, scene_analysis)
        elif destination == SpatialReference.CENTER:
            dest_bbox = self._plan_center_location(image, object_width, object_height, scene_analysis)
        elif destination == SpatialReference.TOP:
            dest_bbox = self._plan_top_location(image, object_width, object_height, scene_analysis)
        elif destination == SpatialReference.BOTTOM:
            dest_bbox = self._plan_bottom_location(image, object_width, object_height, scene_analysis)
        elif destination == SpatialReference.FOREGROUND:
            dest_bbox = self._plan_foreground_location(image, object_width, object_height, scene_analysis)
        elif destination == SpatialReference.BACKGROUND:
            dest_bbox = self._plan_background_location(image, object_width, object_height, scene_analysis)
        else:
            # Default to center
            dest_bbox = self._plan_center_location(image, object_width, object_height, scene_analysis)
        
        # Create destination mask
        dest_mask = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                         for i, coord in enumerate(dest_bbox)]
        dest_mask[y1:y2, x1:x2] = True
        
        return SegmentationMask(dest_mask, confidence=0.8, bbox=dest_bbox)
    
    def _analyze_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze scene structure for obstacle detection and spatial understanding.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing scene analysis results
        """
        analysis = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge detection for obstacle identification
        edges = cv2.Canny(gray, 50, 150)
        analysis['edges'] = edges
        
        # Find contours (potential obstacles)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        obstacle_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                obstacle_contours.append(contour)
        
        analysis['obstacles'] = obstacle_contours
        
        # Create obstacle mask
        obstacle_mask = np.zeros_like(gray)
        cv2.fillPoly(obstacle_mask, obstacle_contours, 255)
        analysis['obstacle_mask'] = obstacle_mask
        
        # Analyze depth (simple approach using brightness)
        brightness = np.mean(gray, axis=(0, 1))
        analysis['average_brightness'] = brightness
        
        # Identify potential ground plane
        ground_mask = self._identify_ground_plane(gray)
        analysis['ground_mask'] = ground_mask
        
        return analysis
    
    def _identify_ground_plane(self, gray: np.ndarray) -> np.ndarray:
        """
        Identify potential ground plane in the image.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Binary mask of ground plane
        """
        h, w = gray.shape
        
        # Simple heuristic: bottom portion of image is likely ground
        ground_mask = np.zeros_like(gray)
        ground_mask[int(h * 0.6):, :] = 255
        
        # Refine using horizontal lines
        lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # If line is mostly horizontal and in bottom half
                if abs(y2 - y1) < abs(x2 - x1) and y1 > h * 0.5:
                    cv2.line(ground_mask, (x1, y1), (x2, y2), 255, 2)
        
        return ground_mask > 0
    
    def _plan_left_location(self, image: np.ndarray, object_width: float, 
                           object_height: float, scene_analysis: Dict[str, Any]) -> List[float]:
        """Plan location on the left side of the image."""
        h, w = image.shape[:2]
        
        # Define left region (left third of image)
        left_region = [0.0, 0.0, 0.33, 1.0]
        
        # Find best position within left region
        best_position = self._find_best_position_in_region(
            image, object_width, object_height, left_region, scene_analysis
        )
        
        return best_position
    
    def _plan_right_location(self, image: np.ndarray, object_width: float,
                            object_height: float, scene_analysis: Dict[str, Any]) -> List[float]:
        """Plan location on the right side of the image."""
        h, w = image.shape[:2]
        
        # Define right region (right third of image)
        right_region = [0.67, 0.0, 1.0, 1.0]
        
        # Find best position within right region
        best_position = self._find_best_position_in_region(
            image, object_width, object_height, right_region, scene_analysis
        )
        
        return best_position
    
    def _plan_center_location(self, image: np.ndarray, object_width: float,
                             object_height: float, scene_analysis: Dict[str, Any]) -> List[float]:
        """Plan location in the center of the image."""
        h, w = image.shape[:2]
        
        # Define center region (middle third of image)
        center_region = [0.33, 0.0, 0.67, 1.0]
        
        # Find best position within center region
        best_position = self._find_best_position_in_region(
            image, object_width, object_height, center_region, scene_analysis
        )
        
        return best_position
    
    def _plan_top_location(self, image: np.ndarray, object_width: float,
                          object_height: float, scene_analysis: Dict[str, Any]) -> List[float]:
        """Plan location in the top portion of the image."""
        h, w = image.shape[:2]
        
        # Define top region (top third of image)
        top_region = [0.0, 0.0, 1.0, 0.33]
        
        # Find best position within top region
        best_position = self._find_best_position_in_region(
            image, object_width, object_height, top_region, scene_analysis
        )
        
        return best_position
    
    def _plan_bottom_location(self, image: np.ndarray, object_width: float,
                             object_height: float, scene_analysis: Dict[str, Any]) -> List[float]:
        """Plan location in the bottom portion of the image."""
        h, w = image.shape[:2]
        
        # Define bottom region (bottom third of image)
        bottom_region = [0.0, 0.67, 1.0, 1.0]
        
        # Find best position within bottom region
        best_position = self._find_best_position_in_region(
            image, object_width, object_height, bottom_region, scene_analysis
        )
        
        return best_position
    
    def _plan_foreground_location(self, image: np.ndarray, object_width: float,
                                 object_height: float, scene_analysis: Dict[str, Any]) -> List[float]:
        """Plan location in the foreground (bottom portion)."""
        return self._plan_bottom_location(image, object_width, object_height, scene_analysis)
    
    def _plan_background_location(self, image: np.ndarray, object_width: float,
                                 object_height: float, scene_analysis: Dict[str, Any]) -> List[float]:
        """Plan location in the background (top portion)."""
        return self._plan_top_location(image, object_width, object_height, scene_analysis)
    
    def _find_best_position_in_region(self, image: np.ndarray, object_width: float,
                                     object_height: float, region: List[float],
                                     scene_analysis: Dict[str, Any]) -> List[float]:
        """
        Find the best position for an object within a specified region.
        
        Args:
            image: Input image
            object_width: Width of object to place
            object_height: Height of object to place
            region: [x1, y1, x2, y2] region to search within
            scene_analysis: Scene analysis results
            
        Returns:
            Best position as [x1, y1, x2, y2]
        """
        h, w = image.shape[:2]
        
        # Convert region to pixel coordinates
        x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                         for i, coord in enumerate(region)]
        
        # Get obstacle mask for this region
        obstacle_mask = scene_analysis['obstacle_mask'][y1:y2, x1:x2]
        
        # Convert object dimensions to pixels
        obj_w_pixels = int(object_width * w)
        obj_h_pixels = int(object_height * h)
        
        # Find valid positions (not overlapping with obstacles)
        valid_positions = []
        
        for y in range(0, y2 - y1 - obj_h_pixels, 10):  # Step by 10 pixels
            for x in range(0, x2 - x1 - obj_w_pixels, 10):
                # Check if this position is valid
                if self._is_position_valid(obstacle_mask, x, y, obj_w_pixels, obj_h_pixels):
                    valid_positions.append((x, y))
        
        if not valid_positions:
            # If no valid positions found, use center of region
            center_x = (x1 + x2) // 2 - obj_w_pixels // 2
            center_y = (y1 + y2) // 2 - obj_h_pixels // 2
            return [center_x / w, center_y / h, 
                   (center_x + obj_w_pixels) / w, (center_y + obj_h_pixels) / h]
        
        # Score positions and select best
        best_position = self._score_positions(valid_positions, region, scene_analysis)
        
        # Convert back to normalized coordinates
        best_x, best_y = best_position
        return [(best_x + x1) / w, (best_y + y1) / h,
                (best_x + x1 + obj_w_pixels) / w, (best_y + y1 + obj_h_pixels) / h]
    
    def _is_position_valid(self, obstacle_mask: np.ndarray, x: int, y: int,
                          obj_w: int, obj_h: int) -> bool:
        """
        Check if a position is valid (not overlapping with obstacles).
        
        Args:
            obstacle_mask: Binary mask of obstacles
            x, y: Position coordinates
            obj_w, obj_h: Object dimensions
            
        Returns:
            True if position is valid
        """
        # Check if object fits within bounds
        if x < 0 or y < 0 or x + obj_w > obstacle_mask.shape[1] or y + obj_h > obstacle_mask.shape[0]:
            return False
        
        # Check for obstacle overlap
        region = obstacle_mask[y:y+obj_h, x:x+obj_w]
        obstacle_ratio = np.sum(region > 0) / region.size
        
        return obstacle_ratio < self.obstacle_threshold
    
    def _score_positions(self, positions: List[Tuple[int, int]], region: List[float],
                        scene_analysis: Dict[str, Any]) -> Tuple[int, int]:
        """
        Score positions and return the best one.
        
        Args:
            positions: List of valid positions
            region: Region being searched
            scene_analysis: Scene analysis results
            
        Returns:
            Best position as (x, y)
        """
        if not positions:
            return (0, 0)
        
        best_score = -1
        best_position = positions[0]
        
        for x, y in positions:
            score = 0
            
            # Prefer positions closer to ground
            if 'ground_mask' in scene_analysis:
                ground_mask = scene_analysis['ground_mask']
                if ground_mask[y, x]:
                    score += 10
            
            # Prefer positions away from edges
            edge_distance = min(x, y, region[2] - region[0] - x, region[3] - region[1] - y)
            score += edge_distance
            
            # Prefer positions with good lighting (brighter areas)
            if 'average_brightness' in scene_analysis:
                # This is a simplified scoring - in practice, you'd analyze local brightness
                score += 5
            
            if score > best_score:
                best_score = score
                best_position = (x, y)
        
        return best_position
    
    def get_available_regions(self, image: np.ndarray, 
                            scene_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, List[float]]:
        """
        Get available regions for object placement.
        
        Args:
            image: Input image
            scene_analysis: Optional pre-computed scene analysis
            
        Returns:
            Dictionary mapping region names to bounding boxes
        """
        if scene_analysis is None:
            scene_analysis = self._analyze_scene(image)
        
        h, w = image.shape[:2]
        
        regions = {
            'left': [0.0, 0.0, 0.33, 1.0],
            'center': [0.33, 0.0, 0.67, 1.0],
            'right': [0.67, 0.0, 1.0, 1.0],
            'top': [0.0, 0.0, 1.0, 0.33],
            'bottom': [0.0, 0.67, 1.0, 1.0],
            'foreground': [0.0, 0.67, 1.0, 1.0],
            'background': [0.0, 0.0, 1.0, 0.33]
        }
        
        return regions


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some "obstacles" (rectangles)
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.rectangle(test_image, (400, 300), (500, 400), (0, 255, 0), -1)
    
    # Initialize spatial planner
    planner = SpatialPlanner()
    
    # Analyze scene
    scene_analysis = planner._analyze_scene(test_image)
    
    # Plan different locations
    object_width, object_height = 0.2, 0.2  # 20% of image size
    
    locations = {
        'left': SpatialReference.LEFT,
        'center': SpatialReference.CENTER,
        'right': SpatialReference.RIGHT,
        'top': SpatialReference.TOP,
        'bottom': SpatialReference.BOTTOM
    }
    
    plt.figure(figsize=(20, 12))
    
    for i, (name, location) in enumerate(locations.items()):
        # Create dummy mask for planning
        dummy_mask = SegmentationMask(np.zeros((480, 640), dtype=bool))
        
        # Plan location
        dest_mask = planner.plan_destination(test_image, dummy_mask, location)
        
        # Visualize
        plt.subplot(2, 3, i + 1)
        vis_image = test_image.copy()
        vis_image[dest_mask.mask] = [255, 255, 0]  # Yellow overlay
        plt.imshow(vis_image)
        plt.title(f"Destination: {name}")
        plt.axis('off')
    
    # Show obstacle mask
    plt.subplot(2, 3, 6)
    plt.imshow(scene_analysis['obstacle_mask'], cmap='gray')
    plt.title("Obstacle Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 