# 🌪️ Task II: Cylinder Parameter Generalization

This directory contains the code, models, and evaluation scripts for **Task II** of TransportBench. This task evaluates the robustness of neural architectures against wide unstructured parameter variations (Knudsen number $Kn$ and Mach number $Ma$) and strong non-equilibrium wake expansion effects around a circular cylinder.

## 📂 Directory Structure
- `train.py`: Unified training script with auto-saving logic and dynamic input parsing.
- `eval.py`: Standardized evaluation script to compute MAE / Relative $L_2$ errors and generate pure contour visualizations with solid obstacle masking.
- `data_loader.py`: Handles geometric masking and data parsing for diverse topological learners.
- `model_*.py`: Model definitions for all 6 baseline architectures configured to $\sim$1M parameters.
- `checkpoints/`: Contains the pre-trained best weights for all evaluated models.

## 🚀 Quick Start

### 1. Evaluation (Inference)
To evaluate a pre-trained model (e.g., Point Transformer) and reproduce the pure flow-field visualizations, run:
```bash
python eval.py --model pt --data_path ./data/cylinder_full_2400.pt

### 2. Training from Scratch
To train a model from scratch using the unified pipeline:
```bash
python train.py --model pt --epochs 2500 --batch_size 16 --lr 5e-4 --data_path ./data/cylinder_full_2400.pt