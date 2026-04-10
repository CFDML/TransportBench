#  Task I: Airfoil Geometry Generalization

This directory contains the code, models, and evaluation scripts for **Task I** of TransportBench. This task evaluates the zero-shot adaptability of neural architectures to complex geometric transformations of an aerodynamic airfoil under rarefied gas dynamics.

##  Directory Structure
- `train.py`: Unified training script with auto-saving logic and `argparse` configuration.
- `eval.py`: Standardized evaluation script to compute MAE / Relative $L_2$ errors and generate flow-field visualizations.
- `data_loader.py`: Handles geometry masking and data parsing for both grid-based and coordinate-based architectures.
- `model_*.py`: Model definitions for all 6 baseline architectures (DeepONet, FNO, U-Net, ViT, AutoEncoder, Point Transformer).
- `checkpoints/`: Contains the pre-trained weights (`best_model_*.pth`) for all evaluated models (approx. 1M parameters each).

##  Data Preparation
The dataset for Task I (`airfoil_unified_128x128.pt`) is hosted externally due to file size limits. 
1. Download the dataset from [Zenodo / Hugging Face Link] (Insert your link here).
2. Place the downloaded `.pt` file in this directory.

##  Quick Start

### 1. Evaluation (Inference)
To evaluate a pre-trained model (e.g., FNO) and reproduce the visualization figures presented in the paper, simply run:
```bash
python eval.py --model fno --data_path ./airfoil_unified_128x128.pt
Supported model tags: fno, unet, vit, ae, deeponet, pt.
The script will output the MAE and Relative L2 Error, and generate a high-quality visualization evaluation_fno.png showing the Ground Truth, Prediction, and Absolute Error.

### 2. Training from Scratch
To train a model from scratch, use the unified train.py script. The script will automatically save the best weights to the checkpoints/ directory based on test set performance.
```bash
python train.py --model fno --epochs 2500 --batch_size 16 --lr 1e-3 --data_path ./airfoil_unified_128x128.pt