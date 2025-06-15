"""
Evaluation Utilities for Scene Manipulation Pipeline

This module provides utilities for evaluating pipeline performance,
output quality, and comparing results.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import structural_similarity as ssim
from ..segmentation import SegmentationMask


class EvaluationUtils:
    """Utility class for evaluating pipeline results and performance."""
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, modified: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between images.
        
        Args:
            original: Original image
            modified: Modified image
            
        Returns:
            SSIM score (0 to 1, higher is better)
        """
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original
            
        if len(modified.shape) == 3:
            modified_gray = cv2.cvtColor(modified, cv2.COLOR_RGB2GRAY)
        else:
            modified_gray = modified
        
        # Calculate SSIM
        score = ssim(original_gray, modified_gray)
        return score
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, modified: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between images.
        
        Args:
            original: Original image
            modified: Modified image
            
        Returns:
            PSNR score (higher is better)
        """
        # Convert to float
        original_float = original.astype(np.float64)
        modified_float = modified.astype(np.float64)
        
        # Calculate MSE
        mse = np.mean((original_float - modified_float) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    @staticmethod
    def calculate_lpips_similarity(original: np.ndarray, modified: np.ndarray) -> float:
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity) score.
        
        Note: This is a simplified implementation. For production use,
        consider using the official LPIPS implementation.
        
        Args:
            original: Original image
            modified: Modified image
            
        Returns:
            LPIPS score (lower is better)
        """
        # This is a placeholder for LPIPS calculation
        # In practice, you would use the official LPIPS implementation
        
        # Simplified version using color histogram similarity
        original_hist = cv2.calcHist([original], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        modified_hist = cv2.calcHist([modified], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        original_hist = cv2.normalize(original_hist, original_hist).flatten()
        modified_hist = cv2.normalize(modified_hist, modified_hist).flatten()
        
        # Calculate correlation
        correlation = cv2.compareHist(original_hist, modified_hist, cv2.HISTCMP_CORREL)
        
        # Convert to similarity score (0 to 1)
        similarity = (correlation + 1) / 2
        
        return similarity
    
    @staticmethod
    def evaluate_object_detection_accuracy(predictions: List[Dict[str, Any]],
                                         ground_truth: List[Dict[str, Any]],
                                         iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate object detection accuracy.
        
        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
            iou_threshold: IoU threshold for considering a detection correct
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not predictions or not ground_truth:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'mAP': 0.0
            }
        
        # Calculate IoU for each prediction-ground truth pair
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                iou_matrix[i, j] = EvaluationUtils._calculate_iou(
                    pred.get('bbox', []), gt.get('bbox', [])
                )
        
        # Find matches
        matched_predictions = set()
        matched_ground_truth = set()
        
        for i in range(len(predictions)):
            for j in range(len(ground_truth)):
                if (i not in matched_predictions and 
                    j not in matched_ground_truth and 
                    iou_matrix[i, j] >= iou_threshold):
                    matched_predictions.add(i)
                    matched_ground_truth.add(j)
        
        # Calculate metrics
        tp = len(matched_predictions)
        fp = len(predictions) - tp
        fn = len(ground_truth) - len(matched_ground_truth)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Simplified mAP calculation
        map_score = precision  # In practice, this would be more complex
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mAP': map_score
        }
    
    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: [x1, y1, x2, y2] for first bounding box
            bbox2: [x1, y1, x2, y2] for second bounding box
            
        Returns:
            IoU score (0 to 1)
        """
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def evaluate_segmentation_accuracy(predicted_mask: SegmentationMask,
                                     ground_truth_mask: SegmentationMask) -> Dict[str, float]:
        """
        Evaluate segmentation accuracy.
        
        Args:
            predicted_mask: Predicted segmentation mask
            ground_truth_mask: Ground truth segmentation mask
            
        Returns:
            Dictionary with evaluation metrics
        """
        pred = predicted_mask.mask
        gt = ground_truth_mask.mask
        
        # Ensure same size
        if pred.shape != gt.shape:
            gt = cv2.resize(gt.astype(np.uint8), (pred.shape[1], pred.shape[0]))
            gt = gt.astype(bool)
        
        # Calculate metrics
        intersection = np.logical_and(pred, gt)
        union = np.logical_or(pred, gt)
        
        # IoU (Jaccard Index)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
        
        # Dice Coefficient (F1 Score)
        dice = 2 * np.sum(intersection) / (np.sum(pred) + np.sum(gt)) if (np.sum(pred) + np.sum(gt)) > 0 else 0.0
        
        # Pixel Accuracy
        pixel_accuracy = np.sum(pred == gt) / pred.size
        
        # Precision and Recall
        tp = np.sum(intersection)
        fp = np.sum(pred) - tp
        fn = np.sum(gt) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            'iou': iou,
            'dice': dice,
            'pixel_accuracy': pixel_accuracy,
            'precision': precision,
            'recall': recall
        }
    
    @staticmethod
    def evaluate_lighting_consistency(original: np.ndarray, 
                                    modified: np.ndarray,
                                    lighting_type: str) -> Dict[str, float]:
        """
        Evaluate lighting consistency and quality.
        
        Args:
            original: Original image
            modified: Modified image with lighting changes
            lighting_type: Type of lighting applied
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert to different color spaces for analysis
        original_hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
        modified_hsv = cv2.cvtColor(modified, cv2.COLOR_RGB2HSV)
        
        # Calculate brightness consistency
        original_brightness = np.mean(original_hsv[:, :, 2])
        modified_brightness = np.mean(modified_hsv[:, :, 2])
        brightness_change = abs(modified_brightness - original_brightness) / 255.0
        
        # Calculate saturation consistency
        original_saturation = np.mean(original_hsv[:, :, 1])
        modified_saturation = np.mean(modified_hsv[:, :, 1])
        saturation_change = abs(modified_saturation - original_saturation) / 255.0
        
        # Calculate color temperature consistency
        original_temp = EvaluationUtils._estimate_color_temperature(original)
        modified_temp = EvaluationUtils._estimate_color_temperature(modified)
        temperature_change = abs(modified_temp - original_temp)
        
        # Calculate overall consistency score
        consistency_score = 1.0 - (brightness_change + saturation_change + temperature_change / 100.0) / 3.0
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        return {
            'brightness_change': brightness_change,
            'saturation_change': saturation_change,
            'temperature_change': temperature_change,
            'consistency_score': consistency_score,
            'lighting_type': lighting_type
        }
    
    @staticmethod
    def _estimate_color_temperature(image: np.ndarray) -> float:
        """
        Estimate color temperature of an image.
        
        Args:
            image: Input image
            
        Returns:
            Estimated color temperature (Kelvin)
        """
        # Simple estimation based on RGB ratios
        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])
        
        # Calculate warm/cool ratio
        warm_ratio = r_mean / (g_mean + b_mean + 1e-8)
        
        # Convert to approximate temperature (simplified)
        # This is a very rough estimation
        if warm_ratio > 1.2:
            return 3000  # Warm
        elif warm_ratio < 0.8:
            return 7000  # Cool
        else:
            return 5500  # Neutral
    
    @staticmethod
    def evaluate_pipeline_performance(execution_times: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate pipeline performance based on execution times.
        
        Args:
            execution_times: Dictionary mapping step names to execution times
            
        Returns:
            Dictionary with performance metrics
        """
        if not execution_times:
            return {}
        
        total_time = sum(execution_times.values())
        
        # Calculate step-wise breakdown
        step_breakdown = {}
        for step, time in execution_times.items():
            percentage = (time / total_time) * 100 if total_time > 0 else 0
            step_breakdown[step] = {
                'time': time,
                'percentage': percentage
            }
        
        # Identify bottlenecks
        bottlenecks = sorted(execution_times.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'total_time': total_time,
            'step_breakdown': step_breakdown,
            'bottlenecks': bottlenecks,
            'average_time_per_step': total_time / len(execution_times) if execution_times else 0
        }
    
    @staticmethod
    def create_evaluation_report(original: np.ndarray,
                               modified: np.ndarray,
                               execution_times: Dict[str, float],
                               additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report.
        
        Args:
            original: Original image
            modified: Modified image
            execution_times: Pipeline execution times
            additional_metrics: Additional metrics to include
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'image_quality_metrics': {
                'ssim': EvaluationUtils.calculate_ssim(original, modified),
                'psnr': EvaluationUtils.calculate_psnr(original, modified),
                'lpips_similarity': EvaluationUtils.calculate_lpips_similarity(original, modified)
            },
            'performance_metrics': EvaluationUtils.evaluate_pipeline_performance(execution_times),
            'additional_metrics': additional_metrics or {}
        }
        
        # Calculate overall score
        quality_score = report['image_quality_metrics']['ssim']
        performance_score = 1.0 / (1.0 + report['performance_metrics']['total_time'])
        
        overall_score = (quality_score * 0.7 + performance_score * 0.3)
        
        report['overall_score'] = overall_score
        report['recommendations'] = EvaluationUtils._generate_recommendations(report)
        
        return report
    
    @staticmethod
    def _generate_recommendations(report: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on evaluation results.
        
        Args:
            report: Evaluation report
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Quality-based recommendations
        ssim = report['image_quality_metrics']['ssim']
        if ssim < 0.5:
            recommendations.append("Low image quality: Consider improving segmentation or inpainting")
        elif ssim < 0.8:
            recommendations.append("Moderate image quality: Fine-tune lighting and composition")
        
        # Performance-based recommendations
        total_time = report['performance_metrics']['total_time']
        if total_time > 60:
            recommendations.append("Slow execution: Consider using smaller models or GPU acceleration")
        
        bottlenecks = report['performance_metrics']['bottlenecks']
        if bottlenecks:
            slowest_step = bottlenecks[0][0]
            recommendations.append(f"Bottleneck detected: {slowest_step} takes {bottlenecks[0][1]:.2f}s")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Create test images
    original = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    modified = original.copy()
    modified[100:200, 100:300] = [255, 0, 0]  # Add red rectangle
    
    # Test evaluation metrics
    evaluator = EvaluationUtils()
    
    # Calculate quality metrics
    ssim_score = evaluator.calculate_ssim(original, modified)
    psnr_score = evaluator.calculate_psnr(original, modified)
    
    print(f"SSIM Score: {ssim_score:.3f}")
    print(f"PSNR Score: {psnr_score:.2f} dB")
    
    # Test performance evaluation
    execution_times = {
        'text_parsing': 0.1,
        'object_detection': 2.5,
        'segmentation': 1.8,
        'relocation': 3.2,
        'relighting': 1.5
    }
    
    performance = evaluator.evaluate_pipeline_performance(execution_times)
    print(f"Total execution time: {performance['total_time']:.2f}s")
    
    # Create comprehensive report
    report = evaluator.create_evaluation_report(original, modified, execution_times)
    print(f"Overall score: {report['overall_score']:.3f}")
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}") 