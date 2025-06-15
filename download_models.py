#!/usr/bin/env python3
"""
Model Download Script for Scene Manipulation Pipeline

This script downloads and sets up all required pre-trained models
for the scene manipulation pipeline.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm


class ModelDownloader:
    """Handles downloading and setup of pre-trained models."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model downloader.
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models = {
            'sam': {
                'vit_h': {
                    'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                    'filename': 'sam_vit_h_4b8939.pth',
                    'size_mb': 2560
                },
                'vit_l': {
                    'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
                    'filename': 'sam_vit_l_0b3195.pth',
                    'size_mb': 1250
                },
                'vit_b': {
                    'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                    'filename': 'sam_vit_b_01ec64.pth',
                    'size_mb': 375
                }
            },
            'detr': {
                'url': 'https://huggingface.co/facebook/detr-resnet-50/resolve/main/pytorch_model.bin',
                'filename': 'detr_resnet50.pth',
                'size_mb': 160
            }
        }
    
    def download_file(self, url: str, filename: str, expected_size_mb: int = None) -> bool:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            filename: Local filename
            expected_size_mb: Expected file size in MB for progress bar
            
        Returns:
            True if successful, False otherwise
        """
        filepath = self.models_dir / filename
        
        # Check if file already exists
        if filepath.exists():
            print(f"File already exists: {filepath}")
            return True
        
        print(f"Downloading {filename} from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Successfully downloaded: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False
    
    def download_sam_models(self, model_type: str = 'vit_h') -> bool:
        """
        Download SAM models.
        
        Args:
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            
        Returns:
            True if successful, False otherwise
        """
        if model_type not in self.models['sam']:
            print(f"Unknown SAM model type: {model_type}")
            return False
        
        model_config = self.models['sam'][model_type]
        
        # Create SAM directory
        sam_dir = self.models_dir / 'sam'
        sam_dir.mkdir(exist_ok=True)
        
        # Download model
        success = self.download_file(
            model_config['url'],
            f"sam/{model_config['filename']}",
            model_config['size_mb']
        )
        
        return success
    
    def download_detr_model(self) -> bool:
        """
        Download DETR model.
        
        Returns:
            True if successful, False otherwise
        """
        detr_config = self.models['detr']
        
        # Create DETR directory
        detr_dir = self.models_dir / 'detr'
        detr_dir.mkdir(exist_ok=True)
        
        # Download model
        success = self.download_file(
            detr_config['url'],
            f"detr/{detr_config['filename']}",
            detr_config['size_mb']
        )
        
        return success
    
    def download_all_models(self) -> bool:
        """
        Download all required models.
        
        Returns:
            True if all successful, False otherwise
        """
        print("Starting model download...")
        
        success = True
        
        # Download SAM models (default to vit_h for size considerations)
        print("\nDownloading SAM models...")
        sam_success = self.download_sam_models('vit_h')
        if not sam_success:
            print("Warning: Failed to download SAM model")
            success = False
        
        # Download DETR model
        print("\nDownloading DETR model...")
        detr_success = self.download_detr_model()
        if not detr_success:
            print("Warning: Failed to download DETR model")
            success = False
        
        # Note: Stable Diffusion models are downloaded automatically by diffusers
        print("\nNote: Stable Diffusion models will be downloaded automatically when first used.")
        
        if success:
            print("\nAll model downloads completed successfully!")
        else:
            print("\nSome model downloads failed. Check the output above for details.")
        
        return success
    
    def check_model_files(self) -> Dict[str, bool]:
        """
        Check which model files are present.
        
        Returns:
            Dictionary mapping model names to availability status
        """
        status = {}
        
        # Check SAM models
        sam_dir = self.models_dir / 'sam'
        for model_type, config in self.models['sam'].items():
            filepath = sam_dir / config['filename']
            status[f'sam_{model_type}'] = filepath.exists()
        
        # Check DETR model
        detr_dir = self.models_dir / 'detr'
        detr_filepath = detr_dir / self.models['detr']['filename']
        status['detr'] = detr_filepath.exists()
        
        return status
    
    def get_download_size(self) -> int:
        """
        Get total download size in MB.
        
        Returns:
            Total size in MB
        """
        total_size = 0
        
        # SAM models (default to vit_h)
        total_size += self.models['sam']['vit_h']['size_mb']
        
        # DETR model
        total_size += self.models['detr']['size_mb']
        
        return total_size
    
    def cleanup_partial_downloads(self):
        """Remove any partial downloads."""
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith('.tmp') or file.endswith('.part'):
                    filepath = Path(root) / file
                    try:
                        filepath.unlink()
                        print(f"Removed partial download: {filepath}")
                    except Exception as e:
                        print(f"Could not remove {filepath}: {e}")


def main():
    """Main function for model download script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for Scene Manipulation Pipeline")
    parser.add_argument('--models-dir', default='models', help='Directory to store models')
    parser.add_argument('--sam-model', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model type to download')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check which models are available')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up partial downloads')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(args.models_dir)
    
    if args.cleanup:
        print("Cleaning up partial downloads...")
        downloader.cleanup_partial_downloads()
        return
    
    if args.check_only:
        print("Checking model availability...")
        status = downloader.check_model_files()
        
        print("\nModel Status:")
        for model, available in status.items():
            status_str = "✓ Available" if available else "✗ Missing"
            print(f"  {model}: {status_str}")
        
        return
    
    # Show download information
    total_size = downloader.get_download_size()
    print(f"Total download size: {total_size} MB")
    
    # Check available space
    try:
        free_space = shutil.disk_usage(args.models_dir).free / (1024 * 1024 * 1024)  # GB
        if free_space < total_size / 1024:
            print(f"Warning: Low disk space. Available: {free_space:.1f} GB, Required: {total_size/1024:.1f} GB")
    except:
        pass  # Skip space check if it fails
    
    # Confirm download
    response = input(f"\nDownload {total_size} MB of models? (y/N): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Download models
    success = downloader.download_all_models()
    
    if success:
        print("\nSetup complete! You can now run the scene manipulation pipeline.")
    else:
        print("\nSetup incomplete. Some models failed to download.")
        print("The pipeline will use fallback methods where possible.")


if __name__ == "__main__":
    main() 