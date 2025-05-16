# Trans_BVM Training Pipeline Documentation

This document provides a high-level overview of the training pipeline for the trans_bvm saliency model.

## 1. Project Structure
```
trans_bvm/
├── data/                # Data loader and preprocessing modules
├── model/               # Model definitions (Generator, Descriptor, ResNet encoders)
│   └── ResNet_models.py
├── train.py             # Training script
├── utils/               # Utility functions (AvgMeter, l2_regularisation, etc.)
└── DOCUMENTATION.md     # This documentation file
```

## 2. Data Loading & Preprocessing
- **Loader**: `get_loader` in `data/__init__.py` loads RGB images, ground truth saliency maps, and transmission maps.
- **Normalization**: Images normalized to ImageNet statistics.
- **Multi-scale**: Supports `size_rates` list for multi-scale training; default is [1]. No scaling applied by default.

## 3. Model Architecture
- **Generator (`Generator`)**: VAE-based saliency generator.
  - Encoders:
    - `xy_encoder`: Encodes concatenated image and ground truth to posterior latent distribution (`muxy`, `logvarxy`).
    - `x_encoder`: Encodes image alone to prior latent distribution (`mux`, `logvarx`).
  - `reparametrize`: Samples latent vectors via the reparameterization trick.
  - `sal_encoder`: Decodes latent samples into initial (`sal_init`) and refined (`sal_ref`) saliency maps.
  - **Forward**:
    - Training: returns posterior init/ref, prior init/ref predictions, plus KL divergence loss.
    - Inference: returns final saliency probability map from prior.
- **Descriptor**: Energy-based model for refining saliency maps (not detailed in this pipeline).

## 4. Loss Functions
- **Reconstruction Loss**: `MSELoss(reduction='mean')` between refined saliency (`sal_ref_post`) and ground truth.
- **BCE Loss**: `BCEWithLogitsLoss(reduction='mean')` between refined saliency logits and ground truth.
- **Structure Loss**: Combines weighted BCE and IoU:
  - Weight map highlights boundary regions: `1 + 5 * |avg_pool(mask) - mask|`.
  - IoU: `(pred * mask)/(pred + mask - pred*mask)`.
- **Smoothness Loss**: `smoothness_loss` penalizes large gradients in predictions.
- **Local Saliency Coherence**: `LocalSaliencyCoherence` enhances local consistency.
- **KL Divergence**: VAE latent loss between posterior and prior distributions.

### Total Loss Composition
```
Total = MSE(sal_ref_post, gt) + 0.1 * BCE(sal_ref_post, gt) + KL(post, prior)
```
Additional losses (smoothness, coherence, regularization) can be added via `opt.sm_weight`, `opt.lsc_weight`, `opt.reg_weight`.

## 5. Training Loop (`train.py`)
1. **Initialization**:
   - Parse hyperparameters (epoch, lr, batchsize, weights, paths).
   - Set device, initialize model and optimizer.
   - Load pretrained weights if provided.
2. **Per-epoch**:
   - Adjust learning rate via `scheduler`.
   - Iterate over batches:
     - Move data to device.
     - Forward pass through `Generator`.
     - Compute losses and total loss.
     - Backpropagate and update parameters.
     - Every 10 steps: compute pixel-level TP/FP/TN/FN and log metrics.
3. **Checkpointing**:
   - Save model state at specified intervals to `opt.save_model_path`.

## 6. Logging & Visualization
- **Metrics**: Loss components and confusion matrix logged every 10 steps.
- **Visualization scripts**:
  - `visualize_prediction_init` and `visualize_prediction_ref`: save predicted saliency maps.
  - `visualize_gt`: save ground truth maps.
  - `visualize_original_img`: save original input images.

## 7. Hyperparameter Tuning
- **Learning rates**: `--lr_gen`, `--lr_des`.
- **Loss weights**: `--sm_weight`, `--reg_weight`, `--lat_weight`, `--vae_loss_weight`.
- **Batch size**: `--batchsize`.
- **Multi-scale**: modify `size_rates` in `train.py`.

## 8. Best Practices
- Use latest PyTorch APIs: `F.interpolate`, `reduction` parameters, no `Variable`.
- Leverage GPU (`.to(device)`) and mixed precision for faster training.
- Monitor TP/FP/TN/FN to understand per-pixel prediction performance.

---
_End of documentation._
