import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import numpy as np

from src.evaluate import evaluate_embeddings

from src.data import load_wdbc
from src.encoder import TabularEncoder
from src.protonet import ProtoNet
from src.train import (
    TrainingConfig,
    train,
)

from sklearn.model_selection import train_test_split



# =========================================================
# 1. Load WDBC
# =========================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
) = load_wdbc()

# =========================================================
# 1b. Split unlabeled training data
# =========================================================

X_meta_train, X_val, y_meta_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train,
)

print("=" * 60)
print("WDBC DATASET")
print("=" * 60)

print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Test samples:",
    X_test.shape[0]
)

print(
    "Features:",
    X_train.shape[1]
)

print(
    "Training labels:",
    np.bincount(y_train).tolist()
)

print(
    "Test labels:",
    np.bincount(y_test).tolist()
)


# =========================================================
# 2. Create encoder
# =========================================================

encoder = TabularEncoder(
    input_dim=30,
    hidden_dim=1024,
    embedding_dim=1024,
)


# =========================================================
# 3. Create ProtoNet
# =========================================================

model = ProtoNet(
    encoder
)


# =========================================================
# 4. Training configuration
# =========================================================

config = TrainingConfig(

    n_way=2,

    k_shot=2,

    q_query=5,

    episodes=500,

    task_batch_size=4,

    learning_rate=1e-3,

    seed=42,
)


# =========================================================
# 5. Meta-train
# =========================================================

print()
print("=" * 60)
print("STUNT META-TRAINING")
print("=" * 60)
history = train(
    X=X_meta_train,
    model=model,
    config=config,
    X_val=X_val,
    y_val=y_val,
)

# =========================================================
# 6. Training summary
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
# 7. Downstream evaluation
# =========================================================

print()
print("=" * 60)
print("DOWNSTREAM EVALUATION")
print("=" * 60)

results = evaluate_embeddings(
    encoder=model.encoder,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

print(
    "Accuracy:",
    f"{results['accuracy']:.4f}"
)

print(
    "Precision:",
    f"{results['precision']:.4f}"
)

print(
    "Recall:",
    f"{results['recall']:.4f}"
)

print(
    "F1:",
    f"{results['f1']:.4f}"
)

print()
print("Confusion matrix:")
print(
    results["confusion_matrix"]
)