"""
Main Pipeline for Scene Manipulation

This module provides the main pipeline that orchestrates all components
for text-controlled object relighting and relocation.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
import os
import json
from datetime import datetime

from .text_parser import InstructionParser, ActionExtractor
from .segmentation import ObjectDetector, SAMSegmenter, MaskProcessor
from .relocation import ObjectRelocator, SpatialPlanner, Compositor
from .relighting import LightingTransformer
from .diffusion import InpaintingModel


class PipelineResult:
    """Container for pipeline execution results."""
    
    def __init__(self, original_image: np.ndarray, final_image: np.ndarray,
                 instruction: str, actions: List[Dict[str, Any]],
                 intermediate_results: Dict[str, Any],
                 metadata: Dict[str, Any]):
        self.original_image = original_image
        self.final_image = final_image
        self.instruction = instruction
        self.actions = actions
        self.intermediate_results = intermediate_results
        self.metadata = metadata
    
    def save_outputs(self, output_dir: str):
        """Save all outputs to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, f"original_{timestamp}.png"), 
                   cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"final_{timestamp}.png"), 
                   cv2.cvtColor(self.final_image, cv2.COLOR_RGB2BGR))
        
        # Save comparison image
        comparison = self.create_comparison_image()
        cv2.imwrite(os.path.join(output_dir, f"comparison_{timestamp}.png"), 
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"Results saved to: {output_dir}")
    
    def create_comparison_image(self) -> np.ndarray:
        """Create a side-by-side comparison image."""
        h, w = self.original_image.shape[:2]
        
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = self.original_image
        comparison[:, w:] = self.final_image
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Modified", (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        return comparison


class SceneManipulationPipeline:
    """
    Main pipeline for scene manipulation via text-controlled object relighting and relocation.
    
    This class orchestrates all components:
    1. Text instruction parsing
    2. Object identification and segmentation
    3. Object relocation
    4. Relighting
    5. Output generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline with all components.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        print("Initializing Scene Manipulation Pipeline...")
        
        # Text parsing components
        self.instruction_parser = InstructionParser()
        self.action_extractor = ActionExtractor()
        
        # Segmentation components
        self.object_detector = ObjectDetector()
        self.sam_segmenter = SAMSegmenter()
        self.mask_processor = MaskProcessor()
        
        # Relocation components
        self.object_relocator = ObjectRelocator()
        self.spatial_planner = SpatialPlanner()
        self.compositor = Compositor()
        
        # Relighting components
        self.lighting_transformer = LightingTransformer()
        
        # Diffusion components
        self.inpainting_model = InpaintingModel()
        
        print("Pipeline initialization complete!")
    
    def process(self, image_path: str, instruction: str,
                save_intermediate: bool = False) -> PipelineResult:
        """
        Process an image with a text instruction.
        
        Args:
            image_path: Path to input image
            instruction: Natural language instruction
            save_intermediate: Whether to save intermediate results
            
        Returns:
            PipelineResult with all outputs
        """
        print(f"Processing image: {image_path}")
        print(f"Instruction: {instruction}")
        
        # Load image
        image = self._load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Initialize results tracking
        intermediate_results = {}
        actions = []
        metadata = {
            'instruction': instruction,
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0.0'
        }
        
        try:
            # Step 1: Parse text instruction
            print("Step 1: Parsing text instruction...")
            parsed_instruction = self.instruction_parser.parse(instruction)
            intermediate_results['parsed_instruction'] = parsed_instruction
            actions.append({
                'step': 'text_parsing',
                'description': 'Parsed natural language instruction',
                'data': {
                    'actions_count': len(parsed_instruction.actions),
                    'global_lighting': parsed_instruction.global_lighting.value if parsed_instruction.global_lighting else None
                }
            })
            
            # Step 2: Object detection and segmentation
            print("Step 2: Object detection and segmentation...")
            object_masks = self._detect_and_segment_objects(image, parsed_instruction)
            intermediate_results['object_masks'] = object_masks
            actions.append({
                'step': 'object_detection',
                'description': f'Detected {len(object_masks)} objects',
                'data': {'object_count': len(object_masks)}
            })
            
            # Step 3: Execute actions
            print("Step 3: Executing actions...")
            current_image = image.copy()
            
            for i, action in enumerate(parsed_instruction.actions):
                print(f"  Executing action {i+1}/{len(parsed_instruction.actions)}: {action.action_type.value}")
                
                if action.action_type.value in ['move', 'relocate']:
                    # Object relocation
                    result = self._execute_relocation_action(current_image, action, object_masks)
                    current_image = result.modified_image
                    intermediate_results[f'relocation_step_{i}'] = result
                    
                elif action.action_type.value in ['relight', 'add_lighting']:
                    # Lighting transformation
                    result = self._execute_lighting_action(current_image, action, object_masks)
                    current_image = result.transformed_image
                    intermediate_results[f'lighting_step_{i}'] = result
            
            # Step 4: Apply global lighting if specified
            if parsed_instruction.global_lighting:
                print("Step 4: Applying global lighting...")
                global_result = self.lighting_transformer.apply_lighting(
                    current_image, parsed_instruction.global_lighting
                )
                current_image = global_result.transformed_image
                intermediate_results['global_lighting'] = global_result
            
            # Step 5: Final processing and validation
            print("Step 5: Final processing...")
            final_image = self._finalize_image(current_image, image)
            
            # Create pipeline result
            result = PipelineResult(
                original_image=image,
                final_image=final_image,
                instruction=instruction,
                actions=actions,
                intermediate_results=intermediate_results,
                metadata=metadata
            )
            
            print("Pipeline execution completed successfully!")
            return result
            
        except Exception as e:
            print(f"Error during pipeline execution: {e}")
            # Return partial result if possible
            return PipelineResult(
                original_image=image,
                final_image=image,  # Return original if processing failed
                instruction=instruction,
                actions=actions,
                intermediate_results=intermediate_results,
                metadata={**metadata, 'error': str(e)}
            )
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from path."""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _detect_and_segment_objects(self, image: np.ndarray, 
                                  parsed_instruction) -> List[Dict[str, Any]]:
        """
        Detect and segment objects mentioned in the instruction.
        
        Args:
            image: Input image
            parsed_instruction: Parsed instruction with actions
            
        Returns:
            List of object masks and metadata
        """
        object_masks = []
        
        for action in parsed_instruction.actions:
            target_object = action.target_object
            
            # Detect objects
            detections = self.object_detector.detect_specific_object(
                image, target_object.name, target_object.attributes
            )
            
            if detections:
                # Get best detection
                best_detection = self.object_detector.get_best_detection(detections)
                
                # Segment using SAM
                if self.sam_segmenter.sam is not None:
                    self.sam_segmenter.set_image(image)
                    
                    # Get bounding box
                    bbox = best_detection.get_bbox_pixels(image.shape[1], image.shape[0])
                    
                    # Segment object
                    mask = self.sam_segmenter.segment_from_box(bbox)
                    mask.label = target_object.name
                    
                    # Process mask
                    processed_mask = self.mask_processor.smooth_boundaries(mask)
                    processed_mask = self.mask_processor.fill_holes(processed_mask)
                    
                    object_masks.append({
                        'action': action,
                        'detection': best_detection,
                        'mask': processed_mask,
                        'bbox': bbox
                    })
        
        return object_masks
    
    def _execute_relocation_action(self, image: np.ndarray, action,
                                 object_masks: List[Dict[str, Any]]) -> Any:
        """
        Execute a relocation action.
        
        Args:
            image: Current image
            action: Action to execute
            object_masks: List of object masks
            
        Returns:
            Relocation result
        """
        # Find matching object mask
        target_mask = None
        for obj_data in object_masks:
            if obj_data['action'].target_object.name == action.target_object.name:
                target_mask = obj_data['mask']
                break
        
        if target_mask is None:
            raise ValueError(f"Could not find mask for object: {action.target_object.name}")
        
        # Execute relocation
        result = self.object_relocator.relocate_object(
            image, target_mask, action.destination
        )
        
        return result
    
    def _execute_lighting_action(self, image: np.ndarray, action,
                               object_masks: List[Dict[str, Any]]) -> Any:
        """
        Execute a lighting action.
        
        Args:
            image: Current image
            action: Action to execute
            object_masks: List of object masks
            
        Returns:
            Lighting result
        """
        # Find matching object mask
        target_mask = None
        for obj_data in object_masks:
            if obj_data['action'].target_object.name == action.target_object.name:
                target_mask = obj_data['mask']
                break
        
        # Apply lighting transformation
        result = self.lighting_transformer.apply_lighting(
            image, action.lighting_type, action.intensity, target_mask
        )
        
        return result
    
    def _finalize_image(self, processed_image: np.ndarray, 
                       original_image: np.ndarray) -> np.ndarray:
        """
        Apply final processing to the image.
        
        Args:
            processed_image: Processed image
            original_image: Original image for reference
            
        Returns:
            Finalized image
        """
        # Ensure image is in valid range
        final_image = np.clip(processed_image, 0, 255).astype(np.uint8)
        
        # Optional: Apply final quality improvements
        # This could include denoising, sharpening, etc.
        
        return final_image
    
    def batch_process(self, image_paths: List[str], instructions: List[str]) -> List[PipelineResult]:
        """
        Process multiple images with their respective instructions.
        
        Args:
            image_paths: List of image paths
            instructions: List of instructions
            
        Returns:
            List of PipelineResult objects
        """
        if len(image_paths) != len(instructions):
            raise ValueError("Number of images and instructions must match")
        
        results = []
        for image_path, instruction in zip(image_paths, instructions):
            try:
                result = self.process(image_path, instruction)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                # Add error result
                results.append(None)
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline and its components."""
        return {
            'pipeline_version': '1.0.0',
            'components': {
                'text_parser': 'InstructionParser + ActionExtractor',
                'segmentation': 'ObjectDetector + SAMSegmenter + MaskProcessor',
                'relocation': 'ObjectRelocator + SpatialPlanner + Compositor',
                'relighting': 'LightingTransformer',
                'diffusion': 'InpaintingModel'
            },
            'supported_actions': [
                'move', 'relocate', 'relight', 'add_lighting'
            ],
            'supported_lighting_types': [
                'sunset', 'golden_hour', 'dramatic', 'soft', 'harsh',
                'warm', 'cool', 'natural', 'artificial'
            ],
            'supported_spatial_references': [
                'left', 'right', 'center', 'top', 'bottom',
                'foreground', 'background', 'front', 'back'
            ]
        }


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SceneManipulationPipeline()
    
    # Get pipeline info
    info = pipeline.get_pipeline_info()
    print("Pipeline Information:")
    print(json.dumps(info, indent=2))
    
    # Example usage (commented out since we don't have actual images)
    """
    # Process a single image
    result = pipeline.process(
        image_path="path/to/image.jpg",
        instruction="Move the red car to the left and add sunset lighting"
    )
    
    # Save results
    result.save_outputs("outputs/")
    
    # Batch processing
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    instructions = [
        "Move the car to the center",
        "Add golden hour lighting",
        "Relocate the person to the background"
    ]
    
    results = pipeline.batch_process(image_paths, instructions)
    
    for i, result in enumerate(results):
        if result:
            result.save_outputs(f"outputs/batch_{i}/")
    """ 