# BVM Smoke Segmentation Project

This repository contains the code and experiments for smoke segmentation using Bayesian Generative Models (BVM). The primary goal is to accurately segment smoke in camera footage by exploring and comparing fully-supervised, weakly-supervised, and domain adaptation (CEDANet) approaches.

## 1. Environment Setup

First, set up the Conda environment using the provided file.

```bash
conda env create -f environment.yml
conda activate dl2024
```

For use on the Snellius cluster, you can load the necessary modules before activating the environment:
```bash
module purge
module load 2024
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dl2024
```

## 2. Directory Structure

The project follows a structured layout for data, models, and results:

```
└── Thesis/
    ├── data/
    │   ├── ijmond_camera/      # Raw video data and generated pseudo-labels
    │   ├── ijmond_data/        # Processed IJmond dataset (train/test splits)
    │   └── SMOKE5K_Dataset/    # SMOKE5K dataset
    ├── src/
    │   └── bvm_training/       # Main source code directory
    │       ├── CEDANet/
    │       ├── trans_bvm/
    │       ├── trans_bvm_self_supervised/
    │       └── trans_bvm_self_supervised_thesis/
    ├── logs_err/               # SLURM error logs
    ├── logs_out/               # SLURM output logs
    ├── models/                 # Trained model weights
    │   ├── full-supervision/
    │   ├── semi-supervision/
    │   └── thesis/
    ├── results/                # Prediction results for evaluation
    ├── *.job                   # SLURM job scripts for training/inference
    └── environment.yml         # Conda environment file
```

## 3. Data Preprocessing: Transmission Map Generation

Before starting any training, you must generate transmission maps for your image datasets. This is a crucial preprocessing step.

**Run the `transmission_map.py` script for your dataset:**
```bash
# For a custom dataset like IJmond
python src/bvm_training/transmission_map.py --dataset_name "ijmond" --output "data/ijmond_data" --mode "train"

# For SMOKE5K dataset (if starting from the ZIP file)
python src/bvm_training/transmission_map.py --dataset_name "SMOKE5K" --dataset_zip "data/SMOKE5K.zip" --output "data/SMOKE5K" --mode "train"
```

## 4. General Workflow

The core workflow for training models, especially for weakly-supervised and domain adaptation tasks, involves three main stages:

1.  **Pseudo-Label Generation (Inference)**: Use a pre-trained model to run inference on unlabeled data (e.g., videos) to generate pseudo-labels. This creates a new labeled dataset.
2.  **Training**: Train a new model on the dataset containing the generated pseudo-labels.
3.  **Testing**: Evaluate the performance of the newly trained model on a test set.

---

## 5. Fully-Supervised BVM

This model is trained on a fully labeled dataset.

### Step 1: Pseudo-Label Generation (Optional)

You can use pre-trained fully-supervised models to generate pseudo-labels from raw videos. This is useful for data expansion or creating new datasets.

**Python Execution:**
```bash
# Example: Generate labels using a model pre-trained on SMOKE5K
python src/bvm_training/trans_bvm_self_supervised_thesis/inference.py \
    --videos_path "data/ijmond_camera/videos" \
    --output_path "data/ijmond_camera/SMOKE5K-full" \
    --pretrained_weights "models/full-supervision/SMOKE5K-supervised/SMOKE5K_Dataset_SMOKE5K_train/SMOKE5K_Dataset_SMOKE5K_train_best_model.pth" \
    --context_frames 2 \
    --threshold 0.6 \
    --constraint_type none
```
The generated pseudo-labels and corresponding images will be saved in subdirectories under the `--output_path`.

**Job File Execution (Snellius):**
The `inference_full.job` script contains commands to run inference with multiple models and constraint types (`none`, `citizen`, `expert`).
```bash
sbatch inference_full.job
```

### Step 2: Training

Train the BVM model on a labeled dataset (e.g., `ijmond_data` or `SMOKE5K`).

**Key Training Parameters:**
*   `--epoch`: Number of training epochs.
*   `--batchsize`: Number of samples per batch.
*   `--lr_gen`: Learning rate for the generator.
*   `--dataset_path`: Path to the training dataset.
*   `--save_model_path`: Directory to save the trained models.
*   `--patience`: Patience for early stopping.

**Python Execution:**
```bash
# Example: Train on the IJmond dataset from scratch
python src/bvm_training/trans_bvm/train.py \
    --epoch 100 \
    --dataset_path "data/ijmond_data/train" \
    --save_model_path "models/full-supervision/ijmond-custom-train" \
    --random_seed 15 \
    --aug \
    --patience 40
```

**Job File Execution (Snellius):**
Modify `train_bvm.job` to set the correct paths and parameters, then run:
```bash
sbatch train_bvm.job
```

### Step 3: Testing

Evaluate the trained model on a test set.

**Configuration:**
**Note:** Before running, you must manually edit the `src/bvm_training/trans_bvm/test.py` script to set the following variables:
*   `dataset_path`: Path to the test dataset images.
*   `model_path`: Path to the trained `.pth` model file.

**Python Execution:**
```bash
# First, edit the paths inside the script, then run:
python src/bvm_training/trans_bvm/test.py
```
The prediction results (mask images) will be saved to a subdirectory within the `results/` folder, named after the model.

---

## 6. Weakly-Supervised BVM

This approach uses a combination of labeled and unlabeled (or pseudo-labeled) data.

### Step 1: Pseudo-Label Generation

Generate pseudo-labels using a self-supervised model. These labels will be used for training the weakly-supervised model.

**Python Execution:**
```bash
# Example: Generate labels with a self-supervised model
python src/bvm_training/trans_bvm_self_supervised/inference.py \
    --videos_path "data/ijmond_camera/videos" \
    --output_path "data/ijmond_camera/SMOKE5K-self" \
    --pretrained_weights "models/weak-supervision/SMOKE5K_Dataset_SMOKE5K_train_ssl_SMOKE5K_Dataset_SMOKE5K_weak_supervision/SMOKE5K_Dataset_SMOKE5K_train_ssl_SMOKE5K_Dataset_SMOKE5K_weak_supervision_best_model.pth" \
    --threshold 0.7 \
    --constraint_type none
```
The generated pseudo-labels (`pl`), transmission maps (`trans`), and original images (`img`) will be saved in subdirectories under the `--output_path`.

**Job File Execution (Snellius):**
The `train_bvm_weakly.job` script (named for its training purpose, but runs inference) can be used to generate pseudo-labels.
```bash
sbatch train_bvm_weakly.job
```

### Step 2: Training

Train the model using both a labeled dataset and the generated pseudo-labels.

**Key Training Parameters:**
*   `--labeled_dataset_path`: Path to the dataset with ground truth labels.
*   `--unlabeled_dataset_path`: Path to the dataset with generated pseudo-labels.
*   `--save_model_path`: Directory to save the new model.
*   `--contrastive_loss_weight`: Weight for the contrastive loss component.

**Python Execution:**
```bash
python src/bvm_training/trans_bvm_self_supervised/train.py \
    --labeled_dataset_path "data/SMOKE5K_Dataset/SMOKE5K_train" \
    --unlabeled_dataset_path "data/ijmond_camera/SMOKE5K-self/non_constraint" \
    --save_model_path "models/semi-supervision/my_weakly_model" \
    --contrastive_loss_weight 0.1 \
    --epoch 100
```

### Step 3: Testing

Evaluate the trained weakly-supervised model.

**Configuration:**
**Note:** Before running, you must manually edit the `src/bvm_training/trans_bvm_self_supervised/test.py` script to set the following variables:
*   `dataset_path`: Path to the test dataset images.
*   `model_path`: Path to the trained weakly-supervised `.pth` model file.

**Python Execution:**
```bash
# First, edit the paths inside the script, then run:
python src/bvm_training/trans_bvm_self_supervised/test.py
```
The prediction results will be saved to a subdirectory within the `results/` folder.

---

## 7. CEDANet (Domain Adaptation)

This is the thesis model, which adapts a model from a source domain (e.g., SMOKE5K) to a target domain (e.g., IJmond).

### Step 1: Pseudo-Label Generation

This step is implicitly handled by the training script, which adapts the model to the target domain. No separate pseudo-label generation script is needed beforehand.

### Step 2: Training

Train the CEDANet model for domain adaptation.

**Key Training Parameters:**
*   `--source_dataset_path`: Path to the source domain training data (e.g., SMOKE5K).
*   `--target_dataset_path`: Path to the target domain training data (e.g., IJmond).
*   `--save_model_path`: Directory to save the adapted model.
*   `--use_ldconv`: Flag to enable LD-Conv module.
*   `--use_attention_pool`: Flag to enable attention pooling.

**Python Execution:**
```bash
python src/bvm_training/CEDANet/train.py \
    --source_dataset_path "data/SMOKE5K_Dataset/SMOKE5K_train" \
    --target_dataset_path "data/ijmond_data/train" \
    --save_model_path "models/thesis/my_domain_adaptation_model" \
    --use_ldconv \
    --use_attention_pool \
    --epoch 100
```

### Step 3: Testing

Test the final domain-adapted model on the target domain's test set.

**Python Execution:**
```bash
python src/bvm_training/CEDANet/test.py \
    --model_path "models/thesis/my_domain_adaptation_model/my_domain_adaptation_model_best.pth" \
    --test_dataset ijmond \
    --use_ldconv \
    --use_attention_pool
```
The prediction results will be saved under `results/thesis/ijmond/`, in a folder named after the tested model.

**Job File Execution (Snellius):**
Modify `test_thesis.job` to point to your trained model, then run:
```bash
sbatch test_thesis.job
```

---

## 8. Advanced Evaluation (Smoke Opacity)

To evaluate how well the model's predictions correlate with different smoke opacity levels (thick vs. thin smoke), use the `eval_opacity.py` script. This requires a ground truth dataset with ternary labels (e.g., 0 for background, 128 for thin smoke, 255 for thick smoke).

**Configuration:**
The script requires you to configure the prediction and ground truth directories inside the file. Specifically, modify the `pred_dir` and `gt_dir` variables.

**Python Execution:**
```bash
# First, edit paths in the script, then run:
python src/bvm_training/CEDANet/eval_opacity.py
```
The script will compute and save detailed statistics, including per-class IoU, Recall, and F1-scores, to a text file in the `logs_tri/` directory.