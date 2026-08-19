import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.data import load_income
from src.evaluate import evaluate_raw


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
print("INCOME RAW FEATURE BASELINE")
print("=" * 60)

print(
    "Train:",
    X_train.shape
)

print(
    "Test:",
    X_test.shape
)


# =========================================================
# Raw evaluation
# =========================================================

results = evaluate_raw(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)


# =========================================================
# Results
# =========================================================

print()
print("=" * 60)
print("RAW DOWNSTREAM EVALUATION")
print("=" * 60)

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