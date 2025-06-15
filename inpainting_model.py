"""
Inpainting Model for Object Removal and Scene Completion

This module provides inpainting capabilities for removing objects
and completing scenes using diffusion-based models.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from diffusers import StableDiffusionInpaintPipeline
from ..segmentation import SegmentationMask


class InpaintingModel:
    """
    Wrapper for diffusion-based inpainting models.
    
    This class provides methods for removing objects from images
    and completing the resulting holes using state-of-the-art
    inpainting techniques.
    """
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-inpainting"):
        """
        Initialize the inpainting model.
        
        Args:
            model_name: Name of the inpainting model to use
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            self.pipeline.to(self.device)
            print(f"Loaded inpainting model: {model_name}")
        except Exception as e:
            print(f"Error loading inpainting model: {e}")
            print("Falling back to basic inpainting...")
            self.pipeline = None
    
    def inpaint_object(self, image: np.ndarray, mask: SegmentationMask,
                      prompt: Optional[str] = None, 
                      negative_prompt: Optional[str] = None,
                      num_inference_steps: int = 50,
                      guidance_scale: float = 7.5) -> np.ndarray:
        """
        Remove an object and inpaint the resulting hole.
        
        Args:
            image: Input image
            mask: Mask of object to remove
            prompt: Optional prompt for inpainting
            negative_prompt: Optional negative prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            Image with object removed and hole inpainted
        """
        if self.pipeline is None:
            return self._fallback_inpaint(image, mask)
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Prepare mask
        mask_pil = Image.fromarray(mask.mask.astype(np.uint8) * 255)
        
        # Generate prompt if not provided
        if prompt is None:
            prompt = self._generate_inpainting_prompt(image, mask)
        
        if negative_prompt is None:
            negative_prompt = "blurry, low quality, distorted, artifacts"
        
        # Run inpainting
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        return np.array(result)
    
    def _generate_inpainting_prompt(self, image: np.ndarray, mask: SegmentationMask) -> str:
        """
        Generate a prompt for inpainting based on image context.
        
        Args:
            image: Input image
            mask: Object mask
            
        Returns:
            Generated prompt
        """
        # This is a simplified prompt generation
        # In practice, you might use a vision-language model to analyze the scene
        
        # Analyze the area around the mask
        bbox = mask.get_bbox()
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) 
                         for i, coord in enumerate(bbox)]
        
        # Get surrounding context
        context_region = image[max(0, y1-50):min(h, y2+50), 
                             max(0, x1-50):min(w, x2+50)]
        
        # Simple heuristic-based prompt generation
        if self._is_outdoor_scene(context_region):
            return "natural landscape, grass, trees, sky, seamless background"
        elif self._is_indoor_scene(context_region):
            return "indoor room, wall, floor, furniture, seamless background"
        else:
            return "seamless background, natural continuation"
    
    def _is_outdoor_scene(self, region: np.ndarray) -> bool:
        """Simple heuristic to detect outdoor scenes."""
        # Check for sky-like colors (blue in upper portion)
        upper_region = region[:region.shape[0]//3, :]
        blue_ratio = np.sum(upper_region[:, :, 2] > upper_region[:, :, 0]) / upper_region.size
        return blue_ratio > 0.3
    
    def _is_indoor_scene(self, region: np.ndarray) -> bool:
        """Simple heuristic to detect indoor scenes."""
        # Check for wall-like colors (neutral, uniform)
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        std_dev = np.std(gray_region)
        return std_dev < 50  # Low variation suggests indoor
    
    def _fallback_inpaint(self, image: np.ndarray, mask: SegmentationMask) -> np.ndarray:
        """
        Fallback inpainting using traditional methods.
        
        Args:
            image: Input image
            mask: Object mask
            
        Returns:
            Inpainted image
        """
        # Use OpenCV inpainting
        mask_uint8 = mask.mask.astype(np.uint8) * 255
        
        # Use TELEA algorithm
        result = cv2.inpaint(image, mask_uint8, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def inpaint_with_guidance(self, image: np.ndarray, mask: SegmentationMask,
                            guidance_image: np.ndarray,
                            guidance_strength: float = 0.8) -> np.ndarray:
        """
        Inpaint with guidance from another image.
        
        Args:
            image: Input image
            mask: Object mask
            guidance_image: Image to use as guidance
            guidance_strength: Strength of guidance (0.0 to 1.0)
            
        Returns:
            Inpainted image
        """
        if self.pipeline is None:
            return self._fallback_inpaint(image, mask)
        
        # Convert to PIL
        image_pil = Image.fromarray(image)
        guidance_pil = Image.fromarray(guidance_image)
        mask_pil = Image.fromarray(mask.mask.astype(np.uint8) * 255)
        
        # Generate prompt based on guidance image
        prompt = self._generate_guidance_prompt(guidance_image)
        
        # Run inpainting with guidance
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                guidance_image=guidance_pil,
                guidance_strength=guidance_strength
            ).images[0]
        
        return np.array(result)
    
    def _generate_guidance_prompt(self, guidance_image: np.ndarray) -> str:
        """Generate prompt based on guidance image."""
        # This would ideally use a vision-language model
        # For now, return a generic prompt
        return "seamless background continuation, natural scene"
    
    def batch_inpaint(self, images: list, masks: list,
                     prompts: Optional[list] = None) -> list:
        """
        Inpaint multiple images in batch.
        
        Args:
            images: List of input images
            masks: List of object masks
            prompts: Optional list of prompts
            
        Returns:
            List of inpainted images
        """
        if prompts is None:
            prompts = [None] * len(images)
        
        results = []
        for image, mask, prompt in zip(images, masks, prompts):
            result = self.inpaint_object(image, mask, prompt)
            results.append(result)
        
        return results
    
    def create_inpainting_comparison(self, original: np.ndarray, 
                                   inpainted: np.ndarray,
                                   mask: SegmentationMask) -> np.ndarray:
        """
        Create a comparison image showing original vs inpainted.
        
        Args:
            original: Original image
            inpainted: Inpainted image
            mask: Object mask
            
        Returns:
            Comparison image
        """
        h, w = original.shape[:2]
        
        # Create comparison image
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original
        comparison[:, w:] = inpainted
        
        # Add mask overlay to show removed area
        mask_overlay = original.copy()
        mask_overlay[mask.mask] = [255, 0, 0]  # Red overlay
        comparison[:h//2, :w] = mask_overlay
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Inpainted", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return comparison


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Add a test object (rectangle)
    cv2.rectangle(test_image, (200, 200), (300, 300), (255, 0, 0), -1)
    
    # Create mask for the object
    mask = np.zeros((512, 512), dtype=bool)
    mask[200:300, 200:300] = True
    seg_mask = SegmentationMask(mask)
    
    # Initialize inpainting model
    inpainter = InpaintingModel()
    
    # Inpaint the object
    inpainted = inpainter.inpaint_object(test_image, seg_mask)
    
    # Create comparison
    comparison = inpainter.create_inpainting_comparison(test_image, inpainted, seg_mask)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Object Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(inpainted)
    plt.title("Inpainted Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 