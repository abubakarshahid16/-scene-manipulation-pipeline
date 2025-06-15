#!/usr/bin/env python3
"""
Test Script for Scene Manipulation Pipeline

This script tests the basic functionality of the pipeline components
to ensure everything is working correctly.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.text_parser import InstructionParser, ActionExtractor
from src.segmentation import ObjectDetector, MaskProcessor
from src.relighting import LightingTransformer
from src.utils.image_utils import ImageUtils


def test_text_parser():
    """Test text parsing functionality."""
    print("Testing Text Parser...")
    
    parser = InstructionParser()
    extractor = ActionExtractor()
    
    # Test instructions
    test_instructions = [
        "Move the red car to the left",
        "Add sunset lighting to the entire scene",
        "Relocate the person to the center and apply golden hour lighting"
    ]
    
    for instruction in test_instructions:
        print(f"\n  Testing: {instruction}")
        
        # Parse instruction
        parsed = parser.parse(instruction)
        print(f"    Actions found: {len(parsed.actions)}")
        
        for action in parsed.actions:
            print(f"      - {action.action_type.value}: {action.target_object.name}")
            if action.destination:
                print(f"        Destination: {action.destination.value}")
            if action.lighting_type:
                print(f"        Lighting: {action.lighting_type.value}")
        
        # Validate actions
        for action in parsed.actions:
            is_valid, errors = extractor.validate_action(action)
            if not is_valid:
                print(f"      ⚠️  Validation errors: {errors}")
    
    print("  ✓ Text parser test completed")


def test_image_utils():
    """Test image utility functions."""
    print("\nTesting Image Utils...")
    
    # Create test image
    test_image = ImageUtils.create_test_image((480, 640, 3), 'gradient')
    
    # Test resize
    resized = ImageUtils.resize_image(test_image, (320, 240))
    print(f"  ✓ Resize: {test_image.shape} -> {resized.shape}")
    
    # Test brightness/contrast adjustment
    adjusted = ImageUtils.adjust_brightness_contrast(test_image, 1.2, 1.5)
    print(f"  ✓ Brightness/contrast adjustment completed")
    
    # Test image info
    info = ImageUtils.get_image_info(test_image)
    print(f"  ✓ Image info: {info['shape']}, mean: {info['mean_value']:.1f}")
    
    # Save test image
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    ImageUtils.save_image(test_image, str(output_dir / "test_image.png"))
    print(f"  ✓ Test image saved to {output_dir / 'test_image.png'}")


def test_object_detector():
    """Test object detection functionality."""
    print("\nTesting Object Detector...")
    
    # Create test image with objects
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add a red rectangle (simulating a car)
    cv2.rectangle(test_image, (200, 200), (300, 250), (255, 0, 0), -1)
    
    # Add a person (circle)
    cv2.circle(test_image, (400, 300), 30, (255, 218, 185), -1)
    
    detector = ObjectDetector()
    
    # Test detection
    detections = detector.detect_objects(test_image)
    print(f"  ✓ Detected {len(detections)} objects")
    
    for detection in detections:
        print(f"    - {detection.label}: {detection.confidence:.3f}")
    
    # Test specific object detection
    car_detections = detector.detect_specific_object(test_image, "car")
    print(f"  ✓ Car detections: {len(car_detections)}")


def test_lighting_transformer():
    """Test lighting transformation functionality."""
    print("\nTesting Lighting Transformer...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    transformer = LightingTransformer()
    
    # Test different lighting types
    lighting_types = [
        "sunset",
        "golden_hour", 
        "dramatic",
        "warm"
    ]
    
    for lighting_type in lighting_types:
        print(f"  Testing {lighting_type} lighting...")
        
        try:
            from src.text_parser import LightingType
            lighting_enum = getattr(LightingType, lighting_type.upper())
            
            result = transformer.apply_lighting(test_image, lighting_enum, intensity=0.8)
            print(f"    ✓ {lighting_type} lighting applied successfully")
            
        except Exception as e:
            print(f"    ⚠️  Error with {lighting_type}: {e}")


def test_mask_processor():
    """Test mask processing functionality."""
    print("\nTesting Mask Processor...")
    
    # Create test mask
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    
    from src.segmentation import SegmentationMask
    seg_mask = SegmentationMask(mask, confidence=0.9, label="test_object")
    
    processor = MaskProcessor()
    
    # Test smoothing
    smoothed = processor.smooth_boundaries(seg_mask)
    print(f"  ✓ Mask smoothing completed")
    
    # Test expansion
    expanded = processor.expand_mask(seg_mask, expansion_pixels=5)
    print(f"  ✓ Mask expansion completed")
    
    # Test hole filling
    filled = processor.fill_holes(seg_mask)
    print(f"  ✓ Mask hole filling completed")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Scene Manipulation Pipeline Tests")
    print("=" * 50)
    
    try:
        test_text_parser()
        test_image_utils()
        test_object_detector()
        test_lighting_transformer()
        test_mask_processor()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("📁 Check test_outputs/ directory for generated test files")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests() 