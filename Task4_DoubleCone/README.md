# Task 4: Double Cone Flow Prediction

This task focuses on predicting hypersonic flow fields around a double cone geometry using various neural operator architectures.

## Pre-trained Models Available ✓

The repository includes pre-trained models ready for evaluation:

| Model | Test Loss | Epoch | Parameters |
|-------|-----------|-------|------------|
| FNO | 0.290208 | 1085 | 16.81M |
| U-Net | 0.328101 | 1952 | 29.67M |
| ViT | 0.254257 | 1812 | 32.14M |
| AutoEncoder | 0.273707 | 1030 | 50.05M |
| DeepONet | 0.277433 | 1246 | 27.93M |

All models are located in `checkpoints/` directory.

## Quick Start with Pre-trained Models

### 1. Test Checkpoints

Verify all pre-trained models can be loaded:

```bash
python test_checkpoints.py
```

### 2. Evaluate a Single Model

```bash
python eval_pretrained.py --model fno --num_samples 3
```

Available models: `fno`, `unet`, `vit`, `ae`, `deeponet`

### 3. Evaluate All Models

```bash
python eval_all.py
```

### 4. Generate Comparison Plots

```bash
python plot_results.py
```

This creates:
- Model performance comparison bar chart
- Performance summary table with all metrics

## Dataset

- Input: 5 channels [x_coord, y_coord, Ma, Tv, Re] - Shape: [N, 5, 17, 384]
- Output: 4 channels [rho, u, v, p] - Shape: [N, 4, 17, 384]
- Total samples: 51 cases (45 train / 6 test)
- Data path: `DoubleCone_Benchmark/Data/processed/double_cone_dataset_with_physics.pt`

## Models

All models are located in the root directory:

- `model_fno.py` - Fourier Neural Operator
- `model_unet.py` - U-Net with skip connections
- `model_vit.py` - Vision Transformer
- `model_ae.py` - Convolutional AutoEncoder
- `model_deeponet.py` - DeepONet with Fourier encoding
- `model_pt.py` - Point Latent Transformer

## Training New Models (Optional)

If you want to train models from scratch:

### Train a single model:

```bash
python train.py --model fno --epochs 3000 --batch_size 8 --lr 5e-4
```

Available models: `fno`, `unet`, `vit`, `ae`, `deeponet`

Note: Point Transformer (pt) checkpoint is not available in the pre-trained set.

### Train all models:

```bash
python train_all.py
```

This will train all 6 models sequentially with default hyperparameters.

## Evaluation

### Using Pre-trained Models (Recommended)

Evaluate pre-trained models directly:

```bash
python eval_pretrained.py --model fno --num_samples 3
```

### Evaluate a single model:

```bash
python eval.py --model fno --num_samples 3
```

### Evaluate all models:

```bash
python eval_all.py
```

## Visualization

Generate comparison plots for all pre-trained models:

```bash
python plot_results.py
```

This creates:
- Model performance comparison bar chart
- Performance summary table

For training history curves (requires training from scratch):

```bash
python plot_comparison.py
```

This creates:
- Training/test loss curves comparison
- Final metrics bar chart
- Performance summary table

## Key Features

- Fourier feature encoding for high-frequency shock wave capture
- Curriculum learning with delayed weight scheduling (starts at epoch 800)
- OneCycleLR scheduler with 40% warmup period
- Weighted loss focusing on wall boundaries and pressure
- Log10 transform for pressure channel
- Gradient clipping for stability

## Directory Structure

```
Task4_DoubleCone/
├── model_*.py           # Model architectures
├── data_loader.py       # Dataset and normalization
├── train.py            # Single model training
├── train_all.py        # Train all models
├── eval.py             # Single model evaluation
├── eval_all.py         # Evaluate all models
├── plot_comparison.py  # Generate comparison plots
├── checkpoints/        # Saved model weights
├── output/            # Evaluation visualizations
└── DoubleCone_Benchmark/  # Original data and experiments
```

## Notes

- Pressure channel undergoes log10 transformation before training
- Models save only after epoch 1000 to avoid early overfitting
- Curriculum learning gradually increases weights for wall boundaries and pressure
- All models use Fourier encoding to capture shock discontinuities
