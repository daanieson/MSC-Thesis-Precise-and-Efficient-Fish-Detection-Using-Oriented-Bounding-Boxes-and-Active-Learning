# Object Detection for Underwater Fish Detection: OBB vs HBB Comparison and Active Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the code and experiments for my thesis on **object detection in underwater environments**, specifically comparing Oriented Bounding Boxes (OBB) and Horizontal Bounding Boxes (HBB) detection methods, and implementing Active Learning strategies for efficient data labelling.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Experiments](#experiments)
  - [OBB vs HBB Comparison](#1-obb-vs-hbb-comparison)
  - [Active Learning](#2-active-learning)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 🔍 Overview

Underwater fish detection presents unique challenges due to varying orientations of fish, complex backgrounds, and limited labelled data. This thesis addresses these challenges through two main contributions:

1. **OBB vs HBB Analysis**: Comparing the effectiveness of Oriented Bounding Boxes against traditional Horizontal Bounding Boxes for detecting fish at various angles and orientations.

2. **Active Learning Pipeline**: Implementing and evaluating multiple active learning strategies to reduce annotation costs while maintaining detection performance.

### Key Features

- Comprehensive comparison of YOLO architectures (v8, v9, v10, v11, v12) with both OBB and HBB heads
- Feature extraction and clustering-based active learning
- Multiple selection strategies: Random, Uncertainty-based, Cluster-proportional, and Diversity-based
- Reproducible experiment pipelines with caching support

## 📁 Repository Structure

```
├── OBB_vs_HBB_Comparison.ipynb    # OBB vs HBB experiment notebook
├── Active_Learning_Experiment.ipynb # Active learning pipeline notebook
├── data.yaml                       # Dataset configuration
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: NVIDIA RTX series with 24GB+ VRAM)
- CUDA 12.x

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/underwater-fish-detection.git
cd underwater-fish-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements

```txt
ultralytics>=8.3.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
tqdm>=4.65.0
hdbscan>=0.8.33
```

## 🧪 Experiments

### 1. OBB vs HBB Comparison

**Notebook**: `OBB_vs_HBB_Comparison.ipynb`

This experiment compares the performance of Oriented Bounding Box detection against Horizontal Bounding Box detection for underwater fish detection.

#### Methodology

- **Models Tested**: YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12
- **Detection Heads**: Standard (HBB) and Oriented (OBB)
- **Image Size**: 1024×1024
- **Metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall

#### Running the Experiment

```python
from ultralytics import YOLO

# Train OBB model
model = YOLO('yolo12x-obb.yaml')
model.load('yolo12x.pt')
results = model.train(data='data.yaml', imgsz=1024, batch=8, epochs=300)

# Evaluate
metrics = model.val(data='data.yaml', split='test')
```

### 2. Active Learning

**Notebook**: `Active_Learning_Experiment.ipynb`

This experiment implements an active learning pipeline to efficiently select the most informative samples for labelling.

#### Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Part 1:       │     │   Part 2:       │     │   Part 3:       │     │   Part 4:       │
│   Layer         │────▶│   Clustering    │────▶│   Selection     │────▶│   Training &    │
│   Analysis      │     │   & Viz         │     │   Strategies    │     │   Evaluation    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

#### Selection Strategies

| Strategy | Description |
|----------|-------------|
| **Random** | Baseline random sampling |
| **Pure Uncertainty** | Selects samples with lowest model confidence |
| **Cluster Uncertainty** | Proportionally selects uncertain samples from each cluster |
| **Typical Diversity** | Selects samples closest to cluster centroids |

#### Running the Pipeline

```python
# Configure
class Config:
    MODEL_PATH = "path/to/model/weights/best.pt"
    IMAGE_DIR = "path/to/unlabeled/pool/images"
    N_IMAGES_TO_SELECT = 200
    KMEANS_N_CLUSTERS = 30

# Extract features and cluster
features_scaled, labels, centroids = perform_clustering_analysis(features, n_clusters=30)

# Apply selection strategy
selection_results = apply_selection_strategies(
    features_scaled, labels, centroids, confidences, n_select=200
)

# Export selected samples
export_selected_dataset(image_paths, selection_results['Typical Diversity'], label_dir, output_dir)
```

## 📊 Results

### OBB vs HBB Comparison

| Model | Type | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------|------|---------|--------------|-----------|--------|
| YOLOv12x | HBB | - | - | - | - |
| YOLOv12x | OBB | - | - | - | - |

*Results to be filled after experiments*

### Active Learning Performance

| Iteration | Training Samples | Strategy | mAP@0.5 | mAP@0.5:0.95 |
|-----------|------------------|----------|---------|--------------|
| 0 | 200 | Initial | - | - |
| 1 | 400 | Typical Diversity | - | - |
| 2 | 600 | Typical Diversity | - | - |

*Results to be filled after experiments*

## 🚀 Usage

### Training a Model

```python
from ultralytics import YOLO

# OBB Detection
model = YOLO('yolo12x-obb.yaml')
model.load('yolo12x.pt')
model.train(data='data.yaml', imgsz=1024, batch=8, epochs=300)

# HBB Detection
model = YOLO('yolo12x.pt')
model.train(data='data.yaml', imgsz=1024, batch=8, epochs=300)
```

### Evaluating a Model

```python
model = YOLO('runs/obb/train/weights/best.pt')
metrics = model.val(data='data.yaml', split='test', imgsz=1024)

print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
```

### Running Inference

```python
model = YOLO('runs/obb/train/weights/best.pt')
results = model.predict(source='path/to/images', conf=0.25, save=True)
```

## 📝 Dataset Structure

```
datasets/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Label Format

**OBB Format** (8 coordinates):
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

**HBB Format** (4 coordinates):
```
class_id x_center y_center width height
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{author2025underwater,
  title={Object Detection for Underwater Fish Detection: OBB vs HBB Comparison and Active Learning},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [scikit-learn](https://scikit-learn.org/) for clustering algorithms
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) for density-based clustering

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: Your Name  
**Contact**: your.email@university.edu  
**Supervisor**: Supervisor Name
