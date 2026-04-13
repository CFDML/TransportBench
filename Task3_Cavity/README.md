# 📦 Task III: Cavity Flow (Micro-Macro Scale Bridge)

This directory contains the codebase for **Task III** of TransportBench. Unlike standard surrogate modeling tasks that predict low-order scalar fields, this task evaluates the capability of neural operators to act as a **Micro-Macro Scale Bridge** in rarefied gas dynamics. 

Models are challenged to predict 10-channel high-order non-equilibrium statistics (including the full stress tensor $\mathbf{P}$ and heat flux $\mathbf{q}$). Accurate predictions in this regime prove that the neural network can implicitly reconstruct the high-dimensional microscopic velocity distribution function (VDF) distortions dictated by the Boltzmann equation.

## 📂 Directory Structure
- `train.py`: Unified training script for all 6 architectures.
- `eval.py`: Standardized evaluation script. Computes MAE / Relative $L_2$ errors and generates streamline visualizations alongside corner-artifact diagnostics.
- `data_loader.py`: Data parser supporting the 10-channel high-order physical fields.
- `model_*.py`: Model definitions constrained to the Standard Regime budget ($\sim$1M parameters).
- `checkpoints/`: Contains the pre-trained best weights for all evaluated models.

## 🚀 Quick Start

### 1. Evaluation (Inference & Visualization)
To evaluate a pre-trained model (e.g., DeepONet) and reproduce the full-field streamlines and corner zooms, run:
```bash
python eval.py --model deeponet --data_dir ./data/cavity
```

### 2. Training from Scratch
To train a model from scratch using the unified pipeline:
```bash
python train.py --model deeponet --epochs 2500 --batch_size 256 --lr 1e-3 --data_dir ./data/cavity
```

## 🏆 Performance Overview
In this confined, smooth flow regime, **Operator Learners** demonstrate absolute dominance, effortlessly resolving the physical fields without inducing the blocky numerical artifacts typically seen in visual models.

| Model | MAE | Highlights |
| :--- | :--- | :--- |
| **DeepONet** | **0.00007** | **SOTA.** Perfect mesh-free smoothness; successfully reconstructs microscopic VDFs. |
| Point Transformer | 0.00179 | Strong performance via unstructured point querying. |
| FNO | 0.00245 | Excellent global harmonic mapping. |
| U-Net | 0.00425 | Prone to blocky numerical artifacts at domain corners. |
| AutoEncoder | 0.00985 | Smooth but exhibits minor latent compression loss. |
| ViT | 0.02061 | Severe grid artifacts due to patch-based attention. |
```
