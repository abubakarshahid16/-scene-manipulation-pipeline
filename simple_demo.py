#!/usr/bin/env python3
"""
Simplified Demo for Scene Manipulation Pipeline

This demo showcases the core functionality without requiring all dependencies.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append('src')

def create_simple_image():
    """Create a simple test image using numpy."""
    # Create a simple scene
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Sky (blue gradient)
    for y in range(150):
        blue_intensity = int(100 + (y / 150) * 155)
        image[y, :] = [blue_intensity, blue_intensity + 50, 255]
    
    # Ground (green)
    image[150:, :] = [34, 139, 34]
    
    # Add simple objects
    # Car (red rectangle)
    image[200:230, 300:350] = [255, 0, 0]
    
    # Person (blue rectangle)
    image[180:220, 100:120] = [0, 0, 255]
    
    # Tree (green circle)
    for y in range(100, 140):
        for x in range(50, 90):
            if (x - 70)**2 + (y - 120)**2 < 400:
                image[y, x] = [0, 100, 0]
    
    return image

def simple_lighting_transform(image, lighting_type="warm"):
    """Apply simple lighting transformation."""
    result = image.copy().astype(np.float32)
    
    if lighting_type == "warm":
        # Increase red channel
        result[:, :, 0] *= 1.3
        result[:, :, 2] *= 0.8  # Reduce blue
    elif lighting_type == "cool":
        # Increase blue channel
        result[:, :, 2] *= 1.3
        result[:, :, 0] *= 0.8  # Reduce red
    elif lighting_type == "dramatic":
        # Increase contrast and reduce brightness
        result = (result - 128) * 1.5 + 128
        result *= 0.8
    
    return np.clip(result, 0, 255).astype(np.uint8)

def simple_object_move(image, object_type="car"):
    """Simple object relocation simulation."""
    result = image.copy()
    
    if object_type == "car":
        # Move car from right to left
        car_region = result[200:230, 300:350]
        result[200:230, 50:100] = car_region
        result[200:230, 300:350] = [34, 139, 34]  # Fill with ground color
    
    elif object_type == "person":
        # Move person to center
        person_region = result[180:220, 100:120]
        result[180:220, 190:210] = person_region
        result[180:220, 100:120] = [34, 139, 34]  # Fill with ground color
    
    return result

def demo_basic_operations():
    """Demonstrate basic operations."""
    print("🎨 Scene Manipulation Pipeline - Simplified Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("simple_demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create original image
    print("Creating sample scene...")
    original = create_simple_image()
    
    # Save original
    plt.imsave(str(output_dir / "original.png"), original)
    print("✓ Original image saved")
    
    # Demo 1: Object relocation
    print("\nDemo 1: Moving the car to the left")
    car_moved = simple_object_move(original, "car")
    plt.imsave(str(output_dir / "car_moved.png"), car_moved)
    print("✓ Car moved to the left")
    
    # Demo 2: Lighting transformation
    print("\nDemo 2: Adding warm lighting")
    warm_lighting = simple_lighting_transform(original, "warm")
    plt.imsave(str(output_dir / "warm_lighting.png"), warm_lighting)
    print("✓ Warm lighting applied")
    
    # Demo 3: Cool lighting
    print("\nDemo 3: Adding cool lighting")
    cool_lighting = simple_lighting_transform(original, "cool")
    plt.imsave(str(output_dir / "cool_lighting.png"), cool_lighting)
    print("✓ Cool lighting applied")
    
    # Demo 4: Dramatic lighting
    print("\nDemo 4: Adding dramatic lighting")
    dramatic_lighting = simple_lighting_transform(original, "dramatic")
    plt.imsave(str(output_dir / "dramatic_lighting.png"), dramatic_lighting)
    print("✓ Dramatic lighting applied")
    
    # Demo 5: Combined operations
    print("\nDemo 5: Combined relocation and lighting")
    combined = simple_object_move(original, "car")
    combined = simple_lighting_transform(combined, "warm")
    plt.imsave(str(output_dir / "combined.png"), combined)
    print("✓ Combined operations completed")
    
    # Create comparison visualization
    print("\nCreating comparison visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    images = [original, car_moved, warm_lighting, cool_lighting, dramatic_lighting, combined]
    titles = ["Original", "Car Moved", "Warm Lighting", "Cool Lighting", "Dramatic Lighting", "Combined"]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Comparison visualization saved")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print("\n" + "=" * 50)
    print("✅ Simplified demo completed successfully!")
    print("\nThis demonstrates the core concepts of the Scene Manipulation Pipeline:")
    print("• Object relocation (moving car from right to left)")
    print("• Lighting transformations (warm, cool, dramatic)")
    print("• Combined operations (relocation + lighting)")
    print("\nThe full pipeline includes:")
    print("• Natural language instruction parsing")
    print("• Advanced object detection (DETR)")
    print("• Precise segmentation (SAM)")
    print("• Diffusion-based inpainting")
    print("• Comprehensive evaluation metrics")

def demo_text_parsing():
    """Demonstrate text parsing functionality."""
    print("\n" + "=" * 50)
    print("📝 Text Parsing Demo")
    print("=" * 50)
    
    # Simulate text parsing
    test_instructions = [
        "Move the red car to the left",
        "Add sunset lighting to the entire scene",
        "Relocate the person to the center and apply golden hour lighting",
        "Move the tree to the background and add dramatic lighting"
    ]
    
    print("Example instructions that the pipeline can parse:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n{i}. {instruction}")
        
        # Simulate parsing
        if "car" in instruction.lower():
            print("   → Detected object: car")
        if "person" in instruction.lower():
            print("   → Detected object: person")
        if "tree" in instruction.lower():
            print("   → Detected object: tree")
        
        if "left" in instruction.lower():
            print("   → Spatial reference: left")
        if "center" in instruction.lower():
            print("   → Spatial reference: center")
        if "background" in instruction.lower():
            print("   → Spatial reference: background")
        
        if "sunset" in instruction.lower():
            print("   → Lighting type: sunset")
        if "golden hour" in instruction.lower():
            print("   → Lighting type: golden hour")
        if "dramatic" in instruction.lower():
            print("   → Lighting type: dramatic")

def main():
    """Main function."""
    try:
        demo_basic_operations()
        demo_text_parsing()
        
        print("\n🎉 Demo completed!")
        print("\nTo run the full pipeline with all features:")
        print("1. Install additional dependencies: pip install -r requirements.txt")
        print("2. Download models: python scripts/download_models.py")
        print("3. Run full demo: python demo.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 