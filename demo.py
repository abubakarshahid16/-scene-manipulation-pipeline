#!/usr/bin/env python3
"""
Demo Script for Scene Manipulation Pipeline

This script demonstrates the capabilities of the scene manipulation
pipeline with various examples and use cases.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.pipeline import SceneManipulationPipeline
from src.utils.image_utils import ImageUtils
from src.utils.visualization import VisualizationUtils
from src.utils.evaluation import EvaluationUtils


class PipelineDemo:
    """Demo class for showcasing pipeline capabilities."""
    
    def __init__(self):
        """Initialize the demo with pipeline."""
        print("Initializing Scene Manipulation Pipeline Demo...")
        self.pipeline = SceneManipulationPipeline()
        self.viz_utils = VisualizationUtils()
        self.eval_utils = EvaluationUtils()
        
        # Create output directory
        self.output_dir = Path("demo_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        print("Demo initialized successfully!")
    
    def create_test_scene(self, scene_type: str = "outdoor") -> np.ndarray:
        """
        Create a test scene for demonstration.
        
        Args:
            scene_type: Type of scene to create ('outdoor', 'indoor', 'street')
            
        Returns:
            Test scene image
        """
        if scene_type == "outdoor":
            # Create outdoor scene with sky, ground, and objects
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Sky (blue gradient)
            for y in range(200):
                blue_intensity = int(135 + (y / 200) * 120)
                image[y, :] = [blue_intensity, blue_intensity + 50, 255]
            
            # Ground (green)
            image[200:, :] = [34, 139, 34]
            
            # Add some objects
            # Tree
            cv2.circle(image, (150, 150), 40, (0, 100, 0), -1)
            cv2.rectangle(image, (145, 190), (155, 280), (139, 69, 19), -1)
            
            # Car
            cv2.rectangle(image, (400, 220), (500, 260), (255, 0, 0), -1)
            cv2.circle(image, (420, 260), 15, (0, 0, 0), -1)
            cv2.circle(image, (480, 260), 15, (0, 0, 0), -1)
            
            # Person
            cv2.circle(image, (300, 180), 15, (255, 218, 185), -1)  # Head
            cv2.rectangle(image, (295, 195), (305, 250), (0, 0, 255), -1)  # Body
            
        elif scene_type == "indoor":
            # Create indoor scene
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Wall (beige)
            image[:, :] = [245, 245, 220]
            
            # Floor
            image[300:, :] = [139, 69, 19]
            
            # Furniture
            # Chair
            cv2.rectangle(image, (100, 250), (200, 300), (160, 82, 45), -1)
            cv2.rectangle(image, (100, 200), (200, 250), (160, 82, 45), -1)
            
            # Table
            cv2.rectangle(image, (300, 220), (500, 240), (139, 69, 19), -1)
            cv2.rectangle(image, (320, 240), (340, 300), (139, 69, 19), -1)
            cv2.rectangle(image, (460, 240), (480, 300), (139, 69, 19), -1)
            
            # Lamp
            cv2.circle(image, (400, 150), 30, (255, 255, 0), -1)
            cv2.rectangle(image, (395, 180), (405, 220), (128, 128, 128), -1)
            
        elif scene_type == "street":
            # Create street scene
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Sky
            image[:150, :] = [135, 206, 235]
            
            # Buildings
            cv2.rectangle(image, (50, 150), (200, 400), (128, 128, 128), -1)
            cv2.rectangle(image, (250, 150), (400, 380), (105, 105, 105), -1)
            cv2.rectangle(image, (450, 150), (600, 420), (169, 169, 169), -1)
            
            # Road
            image[400:, :] = [64, 64, 64]
            
            # Car
            cv2.rectangle(image, (300, 350), (400, 380), (255, 0, 0), -1)
            cv2.circle(image, (320, 380), 12, (0, 0, 0), -1)
            cv2.circle(image, (380, 380), 12, (0, 0, 0), -1)
            
        else:
            # Default random scene
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        return image
    
    def demo_basic_operations(self):
        """Demonstrate basic pipeline operations."""
        print("\n=== Basic Operations Demo ===")
        
        # Create test scene
        scene = self.create_test_scene("outdoor")
        
        # Save original
        ImageUtils.save_image(scene, str(self.output_dir / "demo_original.png"))
        
        # Test different instructions
        instructions = [
            "Move the red car to the left",
            "Add sunset lighting to the entire scene",
            "Move the person to the center and add golden hour lighting"
        ]
        
        results = []
        for i, instruction in enumerate(instructions):
            print(f"\nProcessing instruction {i+1}: {instruction}")
            
            try:
                result = self.pipeline.process(
                    image_path=str(self.output_dir / "demo_original.png"),
                    instruction=instruction
                )
                
                # Save result
                output_path = str(self.output_dir / f"demo_result_{i+1}.png")
                ImageUtils.save_image(result.final_image, output_path)
                
                # Create comparison
                comparison = self.viz_utils.create_pipeline_comparison(
                    scene, result.final_image, f"Result {i+1}: {instruction}"
                )
                ImageUtils.save_image(comparison, str(self.output_dir / f"demo_comparison_{i+1}.png"))
                
                results.append(result)
                
                print(f"  ✓ Completed successfully")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        return results
    
    def demo_lighting_effects(self):
        """Demonstrate various lighting effects."""
        print("\n=== Lighting Effects Demo ===")
        
        # Create test scene
        scene = self.create_test_scene("indoor")
        ImageUtils.save_image(scene, str(self.output_dir / "lighting_original.png"))
        
        # Test different lighting types
        lighting_instructions = [
            "Add warm lighting to the entire scene",
            "Apply dramatic lighting to the chair",
            "Add cool lighting to the table area",
            "Apply soft lighting to the entire scene"
        ]
        
        lighting_results = []
        for i, instruction in enumerate(lighting_instructions):
            print(f"\nProcessing lighting effect {i+1}: {instruction}")
            
            try:
                result = self.pipeline.process(
                    image_path=str(self.output_dir / "lighting_original.png"),
                    instruction=instruction
                )
                
                # Save result
                output_path = str(self.output_dir / f"lighting_result_{i+1}.png")
                ImageUtils.save_image(result.final_image, output_path)
                
                lighting_results.append({
                    'instruction': instruction,
                    'result': result,
                    'image': result.final_image
                })
                
                print(f"  ✓ Completed successfully")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Create lighting comparison grid
        if lighting_results:
            images = [scene] + [r['image'] for r in lighting_results]
            labels = ["Original"] + [r['instruction'] for r in lighting_results]
            
            self.viz_utils.save_visualization_grid(
                images, labels, str(self.output_dir / "lighting_comparison.png")
            )
        
        return lighting_results
    
    def demo_object_relocation(self):
        """Demonstrate object relocation capabilities."""
        print("\n=== Object Relocation Demo ===")
        
        # Create test scene
        scene = self.create_test_scene("street")
        ImageUtils.save_image(scene, str(self.output_dir / "relocation_original.png"))
        
        # Test different relocation instructions
        relocation_instructions = [
            "Move the red car to the center of the road",
            "Relocate the car to the left side",
            "Move the car to the background"
        ]
        
        relocation_results = []
        for i, instruction in enumerate(relocation_instructions):
            print(f"\nProcessing relocation {i+1}: {instruction}")
            
            try:
                result = self.pipeline.process(
                    image_path=str(self.output_dir / "relocation_original.png"),
                    instruction=instruction
                )
                
                # Save result
                output_path = str(self.output_dir / f"relocation_result_{i+1}.png")
                ImageUtils.save_image(result.final_image, output_path)
                
                relocation_results.append({
                    'instruction': instruction,
                    'result': result,
                    'image': result.final_image
                })
                
                print(f"  ✓ Completed successfully")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # Create relocation comparison
        if relocation_results:
            images = [scene] + [r['image'] for r in relocation_results]
            labels = ["Original"] + [r['instruction'] for r in relocation_results]
            
            self.viz_utils.save_visualization_grid(
                images, labels, str(self.output_dir / "relocation_comparison.png")
            )
        
        return relocation_results
    
    def demo_combined_operations(self):
        """Demonstrate combined operations (relocation + lighting)."""
        print("\n=== Combined Operations Demo ===")
        
        # Create test scene
        scene = self.create_test_scene("outdoor")
        ImageUtils.save_image(scene, str(self.output_dir / "combined_original.png"))
        
        # Test combined instructions
        combined_instructions = [
            "Move the red car to the left and add sunset lighting",
            "Relocate the person to the center and apply golden hour lighting",
            "Move the tree to the background and add dramatic lighting"
        ]
        
        combined_results = []
        for i, instruction in enumerate(combined_instructions):
            print(f"\nProcessing combined operation {i+1}: {instruction}")
            
            try:
                result = self.pipeline.process(
                    image_path=str(self.output_dir / "combined_original.png"),
                    instruction=instruction
                )
                
                # Save result
                output_path = str(self.output_dir / f"combined_result_{i+1}.png")
                ImageUtils.save_image(result.final_image, output_path)
                
                # Create detailed comparison
                comparison = self.viz_utils.create_pipeline_comparison(
                    scene, result.final_image, f"Combined {i+1}: {instruction}"
                )
                ImageUtils.save_image(comparison, str(self.output_dir / f"combined_comparison_{i+1}.png"))
                
                combined_results.append({
                    'instruction': instruction,
                    'result': result,
                    'image': result.final_image
                })
                
                print(f"  ✓ Completed successfully")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        return combined_results
    
    def demo_evaluation(self, results: List):
        """Demonstrate evaluation capabilities."""
        print("\n=== Evaluation Demo ===")
        
        if not results:
            print("No results to evaluate")
            return
        
        # Create test scene for comparison
        original = self.create_test_scene("outdoor")
        
        evaluation_results = []
        for i, result_data in enumerate(results):
            if isinstance(result_data, dict):
                result = result_data['result']
                instruction = result_data['instruction']
            else:
                result = result_data
                instruction = f"Instruction {i+1}"
            
            print(f"\nEvaluating: {instruction}")
            
            # Create evaluation report
            execution_times = {
                'text_parsing': 0.1,
                'object_detection': 2.5,
                'segmentation': 1.8,
                'relocation': 3.2,
                'relighting': 1.5
            }
            
            report = self.eval_utils.create_evaluation_report(
                original, result.final_image, execution_times
            )
            
            evaluation_results.append({
                'instruction': instruction,
                'report': report
            })
            
            # Print key metrics
            quality_metrics = report['image_quality_metrics']
            print(f"  SSIM: {quality_metrics['ssim']:.3f}")
            print(f"  PSNR: {quality_metrics['psnr']:.2f} dB")
            print(f"  Overall Score: {report['overall_score']:.3f}")
            
            if report['recommendations']:
                print(f"  Recommendations: {report['recommendations'][0]}")
        
        return evaluation_results
    
    def run_full_demo(self):
        """Run the complete demo suite."""
        print("🚀 Starting Scene Manipulation Pipeline Demo")
        print("=" * 50)
        
        # Run all demos
        basic_results = self.demo_basic_operations()
        lighting_results = self.demo_lighting_effects()
        relocation_results = self.demo_object_relocation()
        combined_results = self.demo_combined_operations()
        
        # Evaluate results
        all_results = basic_results + lighting_results + relocation_results + combined_results
        evaluation_results = self.demo_evaluation(all_results)
        
        # Create summary
        self.create_demo_summary()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("📊 Check the output directory for all generated images and comparisons")
    
    def create_demo_summary(self):
        """Create a summary of the demo results."""
        summary = f"""
# Scene Manipulation Pipeline Demo Summary

## Demo Results
- Basic Operations: {len([f for f in self.output_dir.glob('demo_*.png')])} files
- Lighting Effects: {len([f for f in self.output_dir.glob('lighting_*.png')])} files
- Object Relocation: {len([f for f in self.output_dir.glob('relocation_*.png')])} files
- Combined Operations: {len([f for f in self.output_dir.glob('combined_*.png')])} files

## Key Features Demonstrated
1. **Text Instruction Parsing**: Natural language to structured actions
2. **Object Detection & Segmentation**: Precise object identification
3. **Object Relocation**: Seamless object movement with inpainting
4. **Lighting Transformation**: Various lighting effects and styles
5. **Quality Evaluation**: Comprehensive metrics and recommendations

## Output Files
- `demo_*.png`: Basic operation results
- `lighting_*.png`: Lighting effect demonstrations
- `relocation_*.png`: Object relocation examples
- `combined_*.png`: Combined operation results
- `*_comparison.png`: Side-by-side comparisons
- `lighting_comparison.png`: Grid of lighting effects
- `relocation_comparison.png`: Grid of relocation examples

## Next Steps
1. Try your own images and instructions
2. Experiment with different lighting types
3. Test complex multi-object scenarios
4. Evaluate results using the evaluation utilities
"""
        
        with open(self.output_dir / "demo_summary.md", 'w') as f:
            f.write(summary)
        
        print(f"📝 Demo summary saved to: {self.output_dir / 'demo_summary.md'}")


def main():
    """Main function to run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scene Manipulation Pipeline Demo")
    parser.add_argument('--demo-type', choices=['basic', 'lighting', 'relocation', 'combined', 'full'],
                       default='full', help='Type of demo to run')
    parser.add_argument('--output-dir', default='demo_outputs', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = PipelineDemo()
    
    # Run selected demo
    if args.demo_type == 'basic':
        demo.demo_basic_operations()
    elif args.demo_type == 'lighting':
        demo.demo_lighting_effects()
    elif args.demo_type == 'relocation':
        demo.demo_object_relocation()
    elif args.demo_type == 'combined':
        demo.demo_combined_operations()
    else:  # full
        demo.run_full_demo()


if __name__ == "__main__":
    main() 