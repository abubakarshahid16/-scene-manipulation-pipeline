#!/usr/bin/env python3
"""
Example Usage of Scene Manipulation Pipeline

This script demonstrates how to use the scene manipulation pipeline
with simple examples.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.pipeline import SceneManipulationPipeline
from src.utils.image_utils import ImageUtils


def create_sample_image():
    """Create a sample image for demonstration."""
    # Create a simple outdoor scene
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Sky (blue gradient)
    for y in range(200):
        blue_intensity = int(135 + (y / 200) * 120)
        image[y, :] = [blue_intensity, blue_intensity + 50, 255]
    
    # Ground (green)
    image[200:, :] = [34, 139, 34]
    
    # Add objects
    # Car
    cv2.rectangle(image, (400, 220), (500, 260), (255, 0, 0), -1)
    cv2.circle(image, (420, 260), 15, (0, 0, 0), -1)
    cv2.circle(image, (480, 260), 15, (0, 0, 0), -1)
    
    # Person
    cv2.circle(image, (300, 180), 15, (255, 218, 185), -1)  # Head
    cv2.rectangle(image, (295, 195), (305, 250), (0, 0, 255), -1)  # Body
    
    # Tree
    cv2.circle(image, (150, 150), 40, (0, 100, 0), -1)
    cv2.rectangle(image, (145, 190), (155, 280), (139, 69, 19), -1)
    
    return image


def example_basic_usage():
    """Example of basic pipeline usage."""
    print("=== Basic Pipeline Usage Example ===")
    
    # Create output directory
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create sample image
    print("Creating sample image...")
    sample_image = create_sample_image()
    
    # Save original image
    ImageUtils.save_image(sample_image, str(output_dir / "original.png"))
    print("✓ Original image saved")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = SceneManipulationPipeline()
    
    # Example 1: Simple object relocation
    print("\nExample 1: Moving the car to the left")
    instruction1 = "Move the red car to the left"
    
    try:
        result1 = pipeline.process(
            image_path=str(output_dir / "original.png"),
            instruction=instruction1
        )
        
        # Save result
        ImageUtils.save_image(result1.final_image, str(output_dir / "result1_car_moved.png"))
        print("✓ Car relocation completed")
        
    except Exception as e:
        print(f"✗ Error in Example 1: {e}")
    
    # Example 2: Lighting transformation
    print("\nExample 2: Adding sunset lighting")
    instruction2 = "Add sunset lighting to the entire scene"
    
    try:
        result2 = pipeline.process(
            image_path=str(output_dir / "original.png"),
            instruction=instruction2
        )
        
        # Save result
        ImageUtils.save_image(result2.final_image, str(output_dir / "result2_sunset_lighting.png"))
        print("✓ Sunset lighting applied")
        
    except Exception as e:
        print(f"✗ Error in Example 2: {e}")
    
    # Example 3: Combined operation
    print("\nExample 3: Combined relocation and lighting")
    instruction3 = "Move the person to the center and add golden hour lighting"
    
    try:
        result3 = pipeline.process(
            image_path=str(output_dir / "original.png"),
            instruction=instruction3
        )
        
        # Save result
        ImageUtils.save_image(result3.final_image, str(output_dir / "result3_combined.png"))
        print("✓ Combined operation completed")
        
    except Exception as e:
        print(f"✗ Error in Example 3: {e}")
    
    print(f"\n📁 All results saved to: {output_dir}")


def example_advanced_usage():
    """Example of advanced pipeline usage."""
    print("\n=== Advanced Pipeline Usage Example ===")
    
    # Create output directory
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create sample image
    sample_image = create_sample_image()
    ImageUtils.save_image(sample_image, str(output_dir / "advanced_original.png"))
    
    # Initialize pipeline
    pipeline = SceneManipulationPipeline()
    
    # Example: Multiple instructions in sequence
    instructions = [
        "Move the red car to the left side of the road",
        "Add dramatic lighting to the person",
        "Apply warm lighting to the tree area"
    ]
    
    current_image = sample_image
    
    for i, instruction in enumerate(instructions):
        print(f"\nProcessing instruction {i+1}: {instruction}")
        
        try:
            # Save current state
            ImageUtils.save_image(current_image, str(output_dir / f"advanced_step_{i}.png"))
            
            # Process instruction
            result = pipeline.process(
                image_path=str(output_dir / f"advanced_step_{i}.png"),
                instruction=instruction
            )
            
            # Update current image
            current_image = result.final_image
            
            # Save result
            ImageUtils.save_image(current_image, str(output_dir / f"advanced_result_{i+1}.png"))
            print(f"✓ Step {i+1} completed")
            
        except Exception as e:
            print(f"✗ Error in step {i+1}: {e}")
    
    print(f"\n📁 Advanced results saved to: {output_dir}")


def example_custom_instructions():
    """Example with custom instructions."""
    print("\n=== Custom Instructions Example ===")
    
    # Create output directory
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create sample image
    sample_image = create_sample_image()
    ImageUtils.save_image(sample_image, str(output_dir / "custom_original.png"))
    
    # Initialize pipeline
    pipeline = SceneManipulationPipeline()
    
    # Custom instructions
    custom_instructions = [
        "Relocate the tree to the background",
        "Add cool lighting to the car",
        "Move the person to the foreground and apply soft lighting"
    ]
    
    for i, instruction in enumerate(custom_instructions):
        print(f"\nCustom instruction {i+1}: {instruction}")
        
        try:
            result = pipeline.process(
                image_path=str(output_dir / "custom_original.png"),
                instruction=instruction
            )
            
            # Save result
            ImageUtils.save_image(result.final_image, str(output_dir / f"custom_result_{i+1}.png"))
            print(f"✓ Custom instruction {i+1} completed")
            
        except Exception as e:
            print(f"✗ Error in custom instruction {i+1}: {e}")
    
    print(f"\n📁 Custom results saved to: {output_dir}")


def main():
    """Main function to run examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scene Manipulation Pipeline Examples")
    parser.add_argument('--example', choices=['basic', 'advanced', 'custom', 'all'],
                       default='all', help='Type of example to run')
    
    args = parser.parse_args()
    
    print("🎨 Scene Manipulation Pipeline Examples")
    print("=" * 50)
    
    if args.example == 'basic' or args.example == 'all':
        example_basic_usage()
    
    if args.example == 'advanced' or args.example == 'all':
        example_advanced_usage()
    
    if args.example == 'custom' or args.example == 'all':
        example_custom_instructions()
    
    print("\n" + "=" * 50)
    print("✅ Examples completed!")
    print("📁 Check example_outputs/ directory for all results")


if __name__ == "__main__":
    main() 