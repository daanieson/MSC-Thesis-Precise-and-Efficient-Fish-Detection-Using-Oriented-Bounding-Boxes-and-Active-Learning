# Precise and Efficient Fish Detection Using Oriented Bounding Boxes and Active Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the implementation and experiments for my MSc thesis on automated underwater fish detection. The research addresses two key challenges in aquatic ecosystem monitoring: **geometric imprecision** of standard detection methods for elongated, variably-oriented fish, and the **prohibitive cost** of manual annotation required to train deep learning models.

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

Automated monitoring of aquatic ecosystems using underwater imagery is hindered by two fundamental limitations: the geometric imprecision of conventional horizontal bounding box detection (HBB) methods and the prohibitively high cost of manual annotation. This research addresses both limitations through the integration of **Oriented Bounding Boxes (OBB)** for enhanced geometric precision and **Active Learning** for efficient annotation.

**Key Results:**
- OBB consistently outperforms HBB, achieving **8.5 to 14.9 percentage point improvements** in mAP@0.5:0.95
- **Representative Diversity** active learning achieves **93.2%** of full dataset performance using only **27.6%** of training samples
- Translates to a **72.4% reduction** in annotation effort (~6.1 months saved labour)

## 🎯 Key Findings

### 1. Oriented Bounding Box Superiority
- OBB provides **8.5 to 14.9 percentage point** improvement in mAP@0.5:0.95 across all datasets
- Performance differential widens at stricter thresholds, reaching **18.8 percentage points at mAP@0.75**
- Recall improvements of **7 to 10 percentage points** indicate HBB systematically underestimates fish abundance
- YOLOv12x-OBB emerges as optimal architecture, achieving **0.825 mAP@0.5:0.95** on Protea Banks
- Negligible computational overhead (<0.5ms additional latency)

### 2. Active Learning Validation
- **Representative Diversity** and **Clustered Uncertainty** identified as equally effective strategies
- Both achieve **0.736 mAP@0.5:0.95** with only 27.6% of training data (2,400 of 8,681 images)
- **Pure Uncertainty sampling unsuitable** for marine environments due to aleatoric noise sensitivity
- Dynamic feature layer selection improves clustering quality across iterations
- High-capacity models (YOLOv12x) achieve **0.754 mAP@0.5:0.95** under same protocol

### 3. Architecture Insights
- YOLOv12x Area Attention mechanism particularly suited for partial occlusions
- Larger models access deep semantic features (block 20) for superior clustering
- Extended training beneficial for OBB (sustained learning to Epoch 688 vs 73 for HBB on Aldabra)
- Two-phase workflow recommended: YOLOv12x for annotation prioritisation, YOLOv12s for deployment

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

Four datasets curated from African aquatic ecosystems in collaboration with the **South African Institute for Aquatic Biodiversity (SAIAB)**:

| Dataset | Location | Images | Instances | Avg. Density | Acquisition | Challenges |
|---------|----------|--------|-----------|--------------|-------------|------------|
| **Aldabra Atoll** | Seychelles | 1,910 | 11,800 | 6.2 | SCUBA Diver | Complex coral backgrounds, high species diversity, motion blur |
| **Lake Tanganyika** | Tanzania | 2,510 | 40,944 | 16.3 | BRUVs | High-density schooling, murky water, rocky substrates |
| **Pondoland MPA** | South Africa | 4,000 | 21,344 | 5.3 | BRUVs | Turbidity, morphologically similar species, dense aggregations |
| **Protea Banks** | South Africa | 3,982 | 55,306 | 13.9 | BRUVs | Pelagic environment, frontal approach geometry, variable lighting |

**Total: 12,402 images with 129,394 annotated instances**

### Dataset Split

| Dataset | Train (70%) | Validation (20%) | Test (10%) |
|---------|-------------|------------------|------------|
| Aldabra Atoll | 1,337 | 382 | 191 |
| Lake Tanganyika | 1,757 | 502 | 251 |
| Pondoland MPA | 2,800 | 800 | 400 |
| Protea Banks | 2,787 | 796 | 399 |
| **Combined** | **8,681** | **2,480** | **1,241** |

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

Comprehensive evaluation of Oriented vs Horizontal Bounding Boxes across YOLO architectures (v9e, v10x, v11x, v12x).

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
- Initial pool: 400 images
- Selection per iteration: 200 images
- Total iterations: 10
- Total samples: 2,400 (27.6% of training pool)
- Epochs per iteration: 300 (early stopping patience: 50)
- HDBSCAN: min_cluster_size=15, min_samples=10

## 📈 Results

### OBB vs HBB Performance (YOLOv12x)

| Dataset | Method | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | Precision | Recall |
|---------|--------|--------------|---------|----------|-----------|--------|
| Aldabra | HBB | 0.515 | 0.798 | 0.586 | 0.795 | 0.717 |
| Aldabra | **OBB** | **0.664** | **0.864** | **0.774** | **0.841** | **0.773** |
| Tanganyika | HBB | 0.710 | 0.962 | 0.785 | 0.937 | 0.919 |
| Tanganyika | **OBB** | **0.812** | **0.983** | **0.922** | **0.944** | **0.959** |
| Pondoland | HBB | 0.688 | 0.955 | 0.765 | 0.939 | 0.907 |
| Pondoland | **OBB** | **0.807** | **0.974** | **0.896** | **0.957** | **0.925** |
| Protea Banks | HBB | 0.740 | 0.940 | 0.802 | 0.947 | 0.886 |
| Protea Banks | **OBB** | **0.825** | **0.945** | **0.888** | **0.966** | **0.895** |

### Active Learning Performance (Representative Diversity)

| Architecture | Full Dataset | Active Learning (27.6%) | Performance Retained |
|--------------|--------------|-------------------------|---------------------|
| YOLOv12s | 0.790 | 0.736 | 93.2% |
| YOLOv12x | 0.810 | 0.754 | 93.1% |

### Query Strategy Comparison (YOLOv12s)

| Strategy | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | Precision | Recall |
|----------|--------------|---------|----------|-----------|--------|
| **Representative Diversity** | **0.736** | 0.930 | **0.837** | 0.934 | 0.866 |
| **Clustered Uncertainty** | **0.736** | 0.925 | 0.834 | **0.940** | 0.859 |
| Random Sampling | 0.725 | **0.932** | 0.826 | **0.940** | **0.867** |
| Pure Uncertainty | 0.718 | 0.927 | 0.823 | 0.935 | 0.856 |

### Cost-Benefit Analysis

| Metric | Full Training Set | Active Learning | Savings |
|--------|-------------------|-----------------|---------|
| Samples Annotated | 8,681 | 2,400 | 72.4% reduction |
| Estimated Time | 8.4 months | 2.3 months | 6.1 months saved |
| mAP@0.5:0.95 (YOLOv12s) | 0.790 | 0.736 | 6.8% reduction |
| mAP@0.5:0.95 (YOLOv12x) | 0.810 | 0.754 | 6.9% reduction |

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
from hdbscan import HDBSCAN
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
