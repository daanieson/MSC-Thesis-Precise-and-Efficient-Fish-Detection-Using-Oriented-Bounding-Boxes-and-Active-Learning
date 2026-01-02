# Precise and Efficient Fish Detection Using Oriented Bounding Boxes and Active Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the implementation and experiments for my MSc thesis on automated underwater fish detection. The research addresses two key challenges in aquatic ecosystem monitoring: **geometric imprecision** of standard detection methods for elongated, variably-oriented fish, and the **prohibitive cost** of manual annotation required to train deep learning models.

<p align="center">
  <img src="assets/obb_vs_hbb.png" alt="OBB vs HBB Comparison" width="600"/>
</p>

## 📋 Table of Contents

- [Abstract](#abstract)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Results](#results)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 📝 Abstract

Automated monitoring of aquatic ecosystems using computer vision is hindered by two interrelated challenges: the geometric imprecision of standard object detection methods when applied to elongated and variably oriented fish, and the prohibitive cost of manual annotation required to train deep learning models. This research addresses both limitations through a synergistic methodology combining **Oriented Bounding Boxes (OBB)** for enhanced localisation with **Active Learning** for annotation efficiency.

**Key Results:**
- OBB consistently outperforms HBB, achieving **10-28% improvements** in mAP@0.5:0.95
- **Representative Diversity** active learning achieves **93.16%** of full dataset performance using only **19.35%** of samples
- Translates to an **80% reduction** in annotation effort (~9.7 months saved labour)

## 🎯 Key Findings

### 1. Oriented Bounding Box Superiority
- OBB provides **10-28%** improvement in mAP@0.5:0.95 across all datasets
- Performance gains most pronounced in dense aggregations and camouflage scenarios
- YOLOv12x-OBB emerges as optimal architecture
- Negligible computational overhead (<1ms additional latency)

### 2. Active Learning Validation
- **Representative Diversity** (HDBSCAN clustering) identified as most effective strategy
- Achieves 93.16% of full dataset performance with only 19.35% of samples
- **Pure Uncertainty sampling unsuitable** for marine environments due to noise sensitivity
- Dynamic feature layer selection improves clustering quality across iterations

### 3. Architecture Insights
- YOLOv12x Area Attention mechanism particularly suited for partial occlusions
- Larger models benefit more from diversity-based sampling
- Extended training beneficial for OBB (sustained learning to Epoch 688 vs 369 for HBB)

## 📁 Repository Structure

```
├── notebooks/
│   ├── OBB_vs_HBB_Comparison.ipynb    # Experiment 1: Bounding box comparison
│   └── Active_Learning_Experiment.ipynb # Experiment 2: Active learning pipeline
├── data.yaml                           # Dataset configuration template
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA RTX 6000 Ada, 48GB VRAM)
- CUDA 12.x

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fish-detection-obb-al.git
cd fish-detection-obb-al

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
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
scipy>=1.11.0
```

## 📊 Datasets

Four novel datasets curated from African aquatic ecosystems in collaboration with the **South African Institute for Aquatic Biodiversity (SAIAB)**:

| Dataset | Location | Images | Annotations | Acquisition | Challenges |
|---------|----------|--------|-------------|-------------|------------|
| **Aldabra Atoll** | Seychelles | 1,910 | 11,800 | SCUBA Diver | Complex coral backgrounds, camouflage, motion blur |
| **Lake Tanganyika** | Tanzania | 2,510 | 40,944 | BRUVs | High-density schooling, 250+ endemic cichlid species |
| **Pondoland MPA** | South Africa | 4,002 | 43,264 | BRUVs | Turbidity, dense aggregations, temperate conditions |
| **Protea Banks MPA** | South Africa | 3,980 | 33,386 | BRUVs | Pelagic environment, variable lighting |

**Total: 12,402 images with 129,394 annotations**

### Dataset Structure

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

### Label Formats

**OBB Format** (8 normalised coordinates - polygon corners):
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

**HBB Format** (YOLO standard):
```
class_id x_center y_center width height
```

## 🧪 Experiments

### Experiment 1: OBB vs HBB Comparison

**Notebook**: `OBB_vs_HBB_Comparison.ipynb`

Comprehensive evaluation of Oriented vs Horizontal Bounding Boxes across YOLO architectures (v9, v10, v11, v12).

**Configuration:**
- Image size: 1024×1024
- Epochs: 1500 (early stopping patience: 100)
- Optimizer: AdamW (auto)
- Batch size: 8 (large models) / 16 (small models)

### Experiment 2: Active Learning

**Notebook**: `Active_Learning_Experiment.ipynb`

Evaluation of four query strategies for efficient sample selection:

| Strategy | Type | Description |
|----------|------|-------------|
| **Random** | Baseline | Uniform random sampling |
| **Pure Uncertainty** | Exploitation | Selects lowest confidence samples |
| **Clustered Uncertainty** | Hybrid | Uncertain samples within HDBSCAN clusters |
| **Representative Diversity** | Exploration | Samples nearest to cluster centroids |

**Active Learning Configuration:**
- Initial pool: 200 images
- Selection per iteration: 200 images
- Epochs per iteration: 300
- HDBSCAN: min_cluster_size=15, min_samples=10

## 📈 Results

### OBB vs HBB Performance (YOLOv12x)

| Dataset | Method | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | Precision | Recall |
|---------|--------|--------------|---------|----------|-----------|--------|
| Aldabra | HBB | 51.5% | 79.8% | 58.6% | 79.5% | 71.7% |
| Aldabra | **OBB** | **66.4%** | **86.4%** | **77.4%** | **84.1%** | **77.3%** |
| Tanganyika | HBB | 71.0% | 96.2% | 78.5% | 93.7% | 91.9% |
| Tanganyika | **OBB** | **81.2%** | **98.3%** | **92.2%** | **94.4%** | **95.9%** |
| Pondoland | HBB | 67.4% | 94.2% | 74.5% | 92.1% | 89.3% |
| Pondoland | **OBB** | **79.0%** | **97.4%** | **90.7%** | **95.7%** | **92.5%** |
| Protea Banks | HBB | 68.2% | 94.8% | 75.8% | 91.8% | 90.1% |
| Protea Banks | **OBB** | **80.7%** | **97.4%** | **91.9%** | **95.1%** | **92.9%** |

### Active Learning Performance

| Strategy | Final mAP@0.5:0.95 | % of Full Dataset | Samples Used |
|----------|-------------------|-------------------|--------------|
| Full Dataset | 79.0% | 100% | 12,402 |
| **Representative Diversity** | **73.6%** | **93.16%** | **2,400 (19.35%)** |
| Clustered Uncertainty | 72.8% | 92.15% | 2,400 |
| Random | 70.2% | 88.86% | 2,400 |
| Pure Uncertainty | 68.4% | 86.58% | 2,400 |

## 🚀 Usage

### Training an OBB Model

```python
from ultralytics import YOLO

# Load architecture and pretrained weights
model = YOLO('yolo12x-obb.yaml')
model.load('yolo12x.pt')

# Train
results = model.train(
    data='data.yaml',
    imgsz=1024,
    batch=8,
    epochs=1500,
    patience=100,
    optimizer='auto'
)
```

### Evaluation

```python
model = YOLO('runs/obb/train/weights/best.pt')
metrics = model.val(
    data='data.yaml',
    split='test',
    imgsz=1024,
    save_json=True
)

print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"Precision: {metrics.box.p[0]:.3f}")
print(f"Recall: {metrics.box.r[0]:.3f}")
```

### Active Learning Selection

```python
from sklearn.cluster import HDBSCAN
import numpy as np

# Extract features from model neck layer
features = extract_features(model, unlabeled_pool, layer_idx=3)

# Cluster with HDBSCAN
clusterer = HDBSCAN(min_cluster_size=15, min_samples=10)
labels = clusterer.fit_predict(features)

# Select representative samples (nearest to centroids)
selected_indices = []
for cluster_id in np.unique(labels[labels != -1]):
    cluster_mask = labels == cluster_id
    centroid = features[cluster_mask].mean(axis=0)
    distances = np.linalg.norm(features[cluster_mask] - centroid, axis=1)
    nearest_idx = np.where(cluster_mask)[0][np.argmin(distances)]
    selected_indices.append(nearest_idx)
```

## 📖 Citation

```bibtex
@mastersthesis{salie2025fish,
  title={Precise and Efficient Fish Detection Using Oriented Bounding Boxes and Active Learning},
  author={Salie, Daanyaal},
  year={2025},
  school={Rhodes University},
  address={Grahamstown, South Africa}
}
```

## 🙏 Acknowledgements

- **Supervisor**: Prof. Dane Brown, Rhodes University
- **Data Provider**: South African Institute for Aquatic Biodiversity (SAIAB)
- **Funding**: Telkom SA and the Distributed Multimedia CoE at Rhodes University
- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) for density-based clustering

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: Daanyaal Salie  
**Institution**: Rhodes University  
**Supervisor**: Prof. Dane Brown  
**Year**: 2025
