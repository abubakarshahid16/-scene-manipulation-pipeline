"""
Object Detection Module for Scene Manipulation

This module provides object detection capabilities using state-of-the-art
models like DETR (Detection Transformer) for identifying and localizing
objects in images.
"""

import torch
import torchvision
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2


class DetectionResult:
    """Container for object detection results."""
    
    def __init__(self, bbox: List[float], label: str, confidence: float, mask: Optional[np.ndarray] = None):
        self.bbox = bbox  # [x1, y1, x2, y2] in normalized coordinates
        self.label = label
        self.confidence = confidence
        self.mask = mask  # Binary mask if available
    
    def get_bbox_pixels(self, image_width: int, image_height: int) -> List[int]:
        """Convert normalized bbox to pixel coordinates."""
        x1, y1, x2, y2 = self.bbox
        return [
            int(x1 * image_width),
            int(y1 * image_height),
            int(x2 * image_width),
            int(y2 * image_height)
        ]
    
    def get_center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_area(self) -> float:
        """Get the area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class ObjectDetector:
    """
    Object detector using DETR (Detection Transformer) model.
    
    This class provides methods for detecting objects in images and
    returning structured results with bounding boxes, labels, and
    confidence scores.
    """
    
    def __init__(self, model_name: str = "facebook/detr-resnet-50", confidence_threshold: float = 0.5):
        """
        Initialize the object detector.
        
        Args:
            model_name: Name of the DETR model to use
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Load DETR model and processor
        try:
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = DetrForObjectDetection.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Loaded DETR model: {model_name}")
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Error loading DETR model: {e}")
            print("Falling back to basic object detection...")
            self.model = None
            self.processor = None
    
    def detect_objects(self, image: np.ndarray, target_objects: Optional[List[str]] = None) -> List[DetectionResult]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            target_objects: Optional list of target object names to filter for
            
        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            return self._fallback_detection(image, target_objects)
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Prepare inputs
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]
        
        # Convert to DetectionResult objects
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Get label name
            label_name = self.model.config.id2label[label.item()]
            
            # Filter by target objects if specified
            if target_objects and label_name.lower() not in [obj.lower() for obj in target_objects]:
                continue
            
            # Convert box to normalized coordinates
            x1, y1, x2, y2 = box.cpu().numpy()
            h, w = image_pil.size[1], image_pil.size[0]
            bbox_normalized = [x1/w, y1/h, x2/w, y2/h]
            
            detection = DetectionResult(
                bbox=bbox_normalized,
                label=label_name,
                confidence=score.item()
            )
            detections.append(detection)
        
        return detections
    
    def detect_specific_object(self, image: np.ndarray, object_name: str, 
                             attributes: Optional[List[str]] = None) -> List[DetectionResult]:
        """
        Detect a specific object with optional attributes.
        
        Args:
            image: Input image as numpy array
            object_name: Name of the object to detect
            attributes: Optional list of attributes to consider
            
        Returns:
            List of DetectionResult objects for the specified object
        """
        # Get all detections
        all_detections = self.detect_objects(image)
        
        # Filter by object name
        target_detections = []
        for detection in all_detections:
            if object_name.lower() in detection.label.lower():
                target_detections.append(detection)
        
        # If attributes are specified, try to filter by them
        if attributes and target_detections:
            # This is a simplified approach - in practice, you might want to use
            # a more sophisticated method to match attributes
            filtered_detections = []
            for detection in target_detections:
                # Check if any attribute matches the detection
                for attr in attributes:
                    if attr.lower() in detection.label.lower():
                        filtered_detections.append(detection)
                        break
                else:
                    # If no attributes match, still include the detection
                    filtered_detections.append(detection)
            
            target_detections = filtered_detections
        
        return target_detections
    
    def get_best_detection(self, detections: List[DetectionResult], 
                          criteria: str = "confidence") -> Optional[DetectionResult]:
        """
        Get the best detection based on specified criteria.
        
        Args:
            detections: List of DetectionResult objects
            criteria: Criteria for selection ("confidence", "area", "center")
            
        Returns:
            Best DetectionResult or None if no detections
        """
        if not detections:
            return None
        
        if criteria == "confidence":
            return max(detections, key=lambda x: x.confidence)
        elif criteria == "area":
            return max(detections, key=lambda x: x.get_area())
        elif criteria == "center":
            # Return detection closest to image center
            image_center = (0.5, 0.5)
            return min(detections, key=lambda x: self._distance(x.get_center(), image_center))
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
    
    def _distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _fallback_detection(self, image: np.ndarray, target_objects: Optional[List[str]] = None) -> List[DetectionResult]:
        """
        Fallback detection method using basic computer vision techniques.
        
        This is used when the DETR model is not available.
        """
        detections = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple contour detection
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue
            
            # Get bounding box
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            
            # Convert to normalized coordinates
            bbox_normalized = [x/w, y/h, (x + w_contour)/w, (y + h_contour)/h]
            
            # Create a generic detection
            detection = DetectionResult(
                bbox=bbox_normalized,
                label="object",
                confidence=0.5  # Default confidence
            )
            detections.append(detection)
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[DetectionResult], 
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detections on the image.
        
        Args:
            image: Input image
            detections: List of DetectionResult objects
            save_path: Optional path to save the visualization
            
        Returns:
            Image with detections visualized
        """
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        for detection in detections:
            # Get pixel coordinates
            x1, y1, x2, y2 = detection.get_bbox_pixels(w, h)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection.label}: {detection.confidence:.2f}"
            cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Detect objects
    detections = detector.detect_objects(test_image, target_objects=["person", "car"])
    
    print(f"Found {len(detections)} objects:")
    for detection in detections:
        print(f"  - {detection.label}: {detection.confidence:.3f}")
    
    # Visualize results
    vis_image = detector.visualize_detections(test_image, detections)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_image)
    plt.title("Object Detections")
    plt.axis('off')
    plt.show() 