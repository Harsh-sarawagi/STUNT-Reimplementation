import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import numpy as np

from src.data import load_income
from src.encoder import TabularEncoder
from src.protonet import ProtoNet
from src.train import (
    TrainingConfig,
    train,
)


# =========================================================
# Load Income
# =========================================================

(
    X_train,
    X_val,
    y_train,
    y_val,
    X_test,
    y_test,
) = load_income()


print("=" * 60)
print("INCOME DATASET")
print("=" * 60)

print(
    "Meta-training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)

print(
    "Features:",
    X_train.shape[1]
)

print(
    "Meta-training labels:",
    np.bincount(y_train).tolist()
)

print(
    "Validation labels:",
    np.bincount(y_val).tolist()
)

print(
    "Test labels:",
    np.bincount(y_test).tolist()
)


# =========================================================
# Create encoder
# =========================================================

encoder = TabularEncoder(
    input_dim=X_train.shape[1],
    hidden_dim=1024,
    embedding_dim=1024,
)


# =========================================================
# Create ProtoNet
# =========================================================

model = ProtoNet(
    encoder
)


# =========================================================
# Training configuration
# =========================================================

config = TrainingConfig(

    n_way=2,

    k_shot=2,

    q_query=5,

    mask_ratio_min=0.2,

    mask_ratio_max=0.5,

    episodes=10000,

    task_batch_size=4,

    learning_rate=1e-3,

    weight_decay=1e-5,

    seed=42,

)


# =========================================================
# STUNT meta-training
# =========================================================

print()
print("=" * 60)
print("STUNT META-TRAINING")
print("=" * 60)

history = train(
    X=X_train,
    model=model,
    config=config,
    X_val=X_val,
    y_val=y_val,
)


# =========================================================
# Training summary
# =========================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Initial loss:",
    round(
        history["loss"][0],
        4
    )
)

print(
    "Final loss:",
    round(
        history["loss"][-1],
        4
    )
)


# =========================================================
# Save trained encoder/model
# =========================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "income_stunt_best.pt"
)

import torch

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print(
    "Model saved to:",
    MODEL_PATH
)