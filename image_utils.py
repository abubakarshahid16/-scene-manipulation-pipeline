"""
Image Utility Functions

This module provides common image processing utilities used throughout
the scene manipulation pipeline.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List, Union
import os


class ImageUtils:
    """Utility class for image processing operations."""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load image from path with error handling.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str, 
                  convert_bgr: bool = True) -> bool:
        """
        Save image to path with error handling.
        
        Args:
            image: Image to save
            output_path: Output path
            convert_bgr: Whether to convert RGB to BGR for OpenCV
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert RGB to BGR if needed
            if convert_bgr:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Save image
            success = cv2.imwrite(output_path, image_bgr)
            
            if success:
                print(f"Image saved to: {output_path}")
            else:
                print(f"Failed to save image: {output_path}")
            
            return success
            
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                    maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        if maintain_aspect_ratio:
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = target_w, target_h
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    @staticmethod
    def crop_image(image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crop image using normalized bounding box.
        
        Args:
            image: Input image
            bbox: [x1, y1, x2, y2] in normalized coordinates
            
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                         for i, coord in enumerate(bbox)]
        
        # Ensure bounds are valid
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def create_mask_from_bbox(image_shape: Tuple[int, int], 
                            bbox: List[float]) -> np.ndarray:
        """
        Create binary mask from normalized bounding box.
        
        Args:
            image_shape: (height, width) of image
            bbox: [x1, y1, x2, y2] in normalized coordinates
            
        Returns:
            Binary mask
        """
        h, w = image_shape
        x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                         for i, coord in enumerate(bbox)]
        
        # Ensure bounds are valid
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))
        
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        
        return mask
    
    @staticmethod
    def blend_images(image1: np.ndarray, image2: np.ndarray, 
                    alpha: float = 0.5) -> np.ndarray:
        """
        Blend two images with specified alpha.
        
        Args:
            image1: First image
            image2: Second image
            alpha: Blending factor (0.0 to 1.0)
            
        Returns:
            Blended image
        """
        # Ensure images have same size
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # Blend images
        blended = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
        
        return blended
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, 
                                 brightness: float = 1.0,
                                 contrast: float = 1.0) -> np.ndarray:
        """
        Adjust brightness and contrast of image.
        
        Args:
            image: Input image
            brightness: Brightness multiplier
            contrast: Contrast multiplier
            
        Returns:
            Adjusted image
        """
        # Convert to float
        image_float = image.astype(np.float32)
        
        # Apply brightness
        image_float *= brightness
        
        # Apply contrast
        mean_value = np.mean(image_float)
        image_float = (image_float - mean_value) * contrast + mean_value
        
        # Clip and convert back to uint8
        return np.clip(image_float, 0, 255).astype(np.uint8)
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale.
        
        Args:
            image: Input RGB image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray, 
                       min_val: float = 0.0, 
                       max_val: float = 1.0) -> np.ndarray:
        """
        Normalize image to specified range.
        
        Args:
            image: Input image
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Normalized image
        """
        image_float = image.astype(np.float32)
        
        # Normalize to [0, 1]
        image_norm = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-8)
        
        # Scale to target range
        image_scaled = image_norm * (max_val - min_val) + min_val
        
        return image_scaled
    
    @staticmethod
    def create_noise_image(shape: Tuple[int, int], 
                          noise_type: str = 'gaussian',
                          **kwargs) -> np.ndarray:
        """
        Create noise image of specified shape.
        
        Args:
            shape: (height, width) of image
            noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
            **kwargs: Additional parameters for noise generation
            
        Returns:
            Noise image
        """
        if noise_type == 'gaussian':
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 1)
            noise = np.random.normal(mean, std, shape)
            
        elif noise_type == 'uniform':
            low = kwargs.get('low', 0)
            high = kwargs.get('high', 1)
            noise = np.random.uniform(low, high, shape)
            
        elif noise_type == 'salt_pepper':
            prob = kwargs.get('prob', 0.05)
            noise = np.random.random(shape)
            noise = np.where(noise < prob/2, 0, np.where(noise > 1 - prob/2, 1, 0.5))
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return noise
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> Dict[str, Any]:
        """
        Get information about an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image information
        """
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'mean_value': float(np.mean(image)),
            'std_value': float(np.std(image))
        }
        
        if len(image.shape) == 3:
            info['channels'] = image.shape[2]
            info['channel_means'] = [float(np.mean(image[:, :, i])) for i in range(image.shape[2])]
        
        return info
    
    @staticmethod
    def create_test_image(shape: Tuple[int, int] = (512, 512, 3),
                         image_type: str = 'random') -> np.ndarray:
        """
        Create a test image for experimentation.
        
        Args:
            shape: Shape of image to create
            image_type: Type of test image ('random', 'gradient', 'checkerboard')
            
        Returns:
            Test image
        """
        if image_type == 'random':
            return np.random.randint(0, 255, shape, dtype=np.uint8)
            
        elif image_type == 'gradient':
            h, w = shape[:2]
            y, x = np.ogrid[:h, :w]
            
            if len(shape) == 3:
                # RGB gradient
                r = (x / w * 255).astype(np.uint8)
                g = (y / h * 255).astype(np.uint8)
                b = ((x + y) / (w + h) * 255).astype(np.uint8)
                return np.stack([r, g, b], axis=2)
            else:
                # Grayscale gradient
                return ((x + y) / (w + h) * 255).astype(np.uint8)
                
        elif image_type == 'checkerboard':
            h, w = shape[:2]
            checker_size = 32
            
            checker = np.zeros((h, w), dtype=np.uint8)
            for i in range(0, h, checker_size):
                for j in range(0, w, checker_size):
                    if (i // checker_size + j // checker_size) % 2 == 0:
                        checker[i:i+checker_size, j:j+checker_size] = 255
            
            if len(shape) == 3:
                return np.stack([checker] * 3, axis=2)
            else:
                return checker
                
        else:
            raise ValueError(f"Unknown image type: {image_type}")


# Example usage
if __name__ == "__main__":
    # Create a test image
    test_image = ImageUtils.create_test_image((480, 640, 3), 'gradient')
    
    # Get image info
    info = ImageUtils.get_image_info(test_image)
    print("Image Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test various operations
    resized = ImageUtils.resize_image(test_image, (320, 240))
    blurred = ImageUtils.apply_gaussian_blur(test_image, 15)
    adjusted = ImageUtils.adjust_brightness_contrast(test_image, 1.2, 1.5)
    
    # Save results
    ImageUtils.save_image(test_image, "outputs/original.png")
    ImageUtils.save_image(resized, "outputs/resized.png")
    ImageUtils.save_image(blurred, "outputs/blurred.png")
    ImageUtils.save_image(adjusted, "outputs/adjusted.png") 