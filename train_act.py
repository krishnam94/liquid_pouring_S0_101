#!/usr/bin/env python3
"""
ACT Training Script for Google Colab
Dataset: krishnam94/pour_50ml_SO_101

Run this in Google Colab with A100 GPU enabled.
Training time: ~2-3 hours for 40K steps.
"""

# =============================================================================
# CELL 1: Install LeRobot
# =============================================================================
# !pip install -q lerobot

# =============================================================================
# CELL 2: Mount Google Drive (for saving checkpoints)
# =============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')
"""

# =============================================================================
# CELL 3: Download Dataset
# =============================================================================
import os
import json

# Download dataset from HuggingFace
os.system("huggingface-cli download krishnam94/pour_50ml_SO_101 --repo-type dataset --local-dir ./dataset --force-download")

# Fix stats locally (required for image normalization)
stats = {
    "action": {
        "mean": [4.468, -49.7, 52.16, 6.83, -39.0, 19.83],
        "std": [4.83, 44.6, 42.19, 12.4, 20.45, 0.196],
        "min": [-11.35, -100.0, -28.15, -30.33, -91.45, 19.45],
        "max": [19.85, 28.91, 100.0, 47.41, 34.16, 20.37]
    },
    "observation.state": {
        "mean": [4.43, -48.82, 54.13, 7.19, -39.05, 21.76],
        "std": [4.83, 45.18, 41.03, 12.31, 20.47, 0.06],
        "min": [-11.54, -99.48, -25.26, -29.1, -91.5, 21.6],
        "max": [19.46, 30.32, 100.0, 47.24, 33.72, 22.82]
    },
    "observation.images.wrist": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 1.0]
    },
    "observation.images.front": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 1.0]
    }
}

with open("./dataset/meta/stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("✅ Dataset downloaded and stats fixed!")

# =============================================================================
# CELL 4: Train ACT - Baseline
# =============================================================================
"""
!lerobot-train \
    --dataset.repo_id=krishnam94/pour_50ml_SO_101 \
    --dataset.root=./dataset \
    --policy.type=act \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --optimizer.lr=1e-5 \
    --steps=40000 \
    --save_freq=5000 \
    --output_dir=/content/drive/MyDrive/act_pour_50ml \
    --wandb.enable=false
"""

# =============================================================================
# CELL 5: Train ACT - Robust (RECOMMENDED)
# =============================================================================
# This variant includes:
# - Image augmentation (robust to lighting changes)
# - Lower kl_weight=5 (captures demonstration style better)
# - Produces faster, more confident robot motion
"""
!lerobot-train \
    --dataset.repo_id=krishnam94/pour_50ml_SO_101 \
    --dataset.root=./dataset \
    --policy.type=act \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.kl_weight=5 \
    --dataset.image_transforms.enable=true \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --optimizer.lr=1e-5 \
    --steps=40000 \
    --save_freq=5000 \
    --output_dir=/content/drive/MyDrive/act_pour_50ml_robust \
    --wandb.enable=false
"""

# =============================================================================
# CELL 6: Resume Training (if Colab disconnects)
# =============================================================================
"""
!lerobot-train \
    --config_path=/content/drive/MyDrive/act_pour_50ml/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
"""

# =============================================================================
# CELL 7: Upload Best Checkpoint to HuggingFace
# =============================================================================
"""
!huggingface-cli upload krishnam94/act_pour_50ml \
    /content/drive/MyDrive/act_pour_50ml/checkpoints/last/pretrained_model
"""
