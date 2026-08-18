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

from sklearn.linear_model import LogisticRegression

from src.data import load_wdbc
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =========================================================
# Load WDBC
# =========================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
) = load_wdbc()


# =========================================================
# Raw-feature baseline
# =========================================================

classifier = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

classifier.fit(
    X_train,
    y_train,
)


# =========================================================
# Prediction
# =========================================================

predictions = classifier.predict(
    X_test
)


# =========================================================
# Metrics
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
)

recall = recall_score(
    y_test,
    predictions,
)

f1 = f1_score(
    y_test,
    predictions,
)

cm = confusion_matrix(
    y_test,
    predictions,
)


# =========================================================
# Results
# =========================================================

print("=" * 60)
print("RAW FEATURE BASELINE")
print("=" * 60)

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1:        {f1:.4f}"
)

print()
print("Confusion matrix:")
print(cm)