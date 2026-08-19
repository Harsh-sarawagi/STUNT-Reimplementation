import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import torch

from src.data import load_income
from src.encoder import TabularEncoder
from src.protonet import ProtoNet
from src.evaluate import evaluate_embeddings


# =========================================================
# Load data
# =========================================================

(
    X_train,
    X_val,
    y_train,
    y_val,
    X_test,
    y_test,
) = load_income()


# =========================================================
# Create model
# =========================================================

encoder = TabularEncoder(
    input_dim=X_train.shape[1],
    hidden_dim=1024,
    embedding_dim=1024,
)

model = ProtoNet(
    encoder
)


# =========================================================
# Load trained STUNT model
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "income_stunt_best.pt"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

model.load_state_dict(
    checkpoint
)

print("=" * 60)
print("LOADED STUNT MODEL")
print("=" * 60)

print(
    MODEL_PATH
)


# =========================================================
# Downstream evaluation
# =========================================================

print()
print("=" * 60)
print("INCOME DOWNSTREAM EVALUATION")
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