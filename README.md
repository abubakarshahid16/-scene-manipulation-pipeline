# Scene Manipulation via Text-Controlled Object Relighting and Relocation

This project implements a comprehensive pipeline for scene manipulation using natural language instructions. The system can perform object-level modifications such as relocation and relighting by leveraging segmentation models and diffusion-based inpainting.

## 🎯 Project Overview

The pipeline consists of five main components:

1. **Text Instruction Parsing** - Converts natural language instructions into structured actions
2. **Object Identification and Segmentation** - Uses SAM/DETR to localize and segment target objects
3. **Object Relocation** - Removes and re-inserts objects in new locations
4. **Relighting** - Applies lighting transformations using diffusion models
5. **Output Generation** - Provides comprehensive visual results

## 🏗️ Project Structure

```
├── src/
│   ├── text_parser/          # Natural language instruction parsing
│   ├── segmentation/         # Object detection and segmentation
│   ├── relocation/           # Object relocation logic
│   ├── relighting/           # Lighting transformation
│   ├── diffusion/            # Diffusion model utilities
│   └── utils/                # Helper functions
├── models/                   # Pre-trained model downloads
├── data/                     # Dataset and sample images
├── outputs/                  # Generated results
├── notebooks/                # Jupyter notebooks for experimentation
└── tests/                    # Unit tests
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd scene-manipulation-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required models:
```bash
python scripts/download_models.py
```

### Basic Usage

```python
from src.pipeline import SceneManipulationPipeline

# Initialize the pipeline
pipeline = SceneManipulationPipeline()

# Process an image with text instruction
result = pipeline.process(
    image_path="path/to/image.jpg",
    instruction="Move the car to the left and add sunset lighting"
)

# Save results
result.save_outputs("outputs/")
```

## 📋 Features

- **Natural Language Understanding**: Parse complex instructions like "Move the red car to the left and add golden hour lighting"
- **State-of-the-art Segmentation**: Uses SAM and DETR for precise object detection
- **Advanced Relighting**: Neural relighting with diffusion models
- **Seamless Object Relocation**: Maintains visual consistency during object movement
- **Comprehensive Outputs**: Side-by-side comparisons with intermediate results

## 🎨 Example Instructions

- "Move the car to the left side of the road"
- "Add sunset lighting to the entire scene"
- "Relocate the person to the center and add dramatic shadows"
- "Move the tree to the background and apply golden hour lighting"

## 📊 Evaluation Metrics

- CLIP similarity scores between instruction and output
- Object detection accuracy
- Lighting consistency metrics
- Visual quality assessment

## 🔬 Technical Details

### Models Used
- **SAM (Segment Anything Model)**: For precise object segmentation
- **DETR (Detection Transformer)**: For object detection
- **Stable Diffusion**: For inpainting and generation
- **CLIP**: For text-image similarity scoring

### Key Algorithms
- Diffusion-based inpainting for object removal
- Neural relighting with learned lighting adjustments
- Prompt-conditioned generation for object re-insertion
- Multi-scale consistency checking

## 📚 References

- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Neural Gaffer: Relighting Any Object via Diffusion](https://arxiv.org/abs/2303.12503)
- [Paint by Inpaint: Learning to Add Image Objects by Removing Them](https://arxiv.org/abs/2303.17693)

## 🤝 Contributing

This is an open-ended research project. Contributions are welcome! Please feel free to:
- Propose alternative approaches
- Add new features
- Improve documentation
- Submit bug reports

## 📄 License

This project is for educational and research purposes. 