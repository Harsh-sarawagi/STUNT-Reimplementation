"""
Data loading utilities for the STUNT project.

The WDBC dataset contains:
- 569 samples
- 30 numerical features
- binary diagnosis labels

The STUNT training pipeline receives only X.
The original diagnosis y is retained separately for evaluation.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_wdbc(
    path="data/raw/wdbc.data",
    test_size=0.2,
    random_state=42,
):
    """
    Load and preprocess the Wisconsin Diagnostic Breast Cancer dataset.

    Returns
    -------
    X_train:
        Training features.

    X_test:
        Test features.

    y_train:
        Original labels for evaluation.

    y_test:
        Original labels for evaluation.

    scaler:
        Fitted StandardScaler.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    # ---------------------------------------------------------
    # WDBC format
    #
    # column 0  = ID
    # column 1  = diagnosis (M/B)
    # columns 2-31 = 30 features
    # ---------------------------------------------------------

    data = np.loadtxt(
        path,
        delimiter=",",
        dtype=str,
    )

    patient_ids = data[:, 0]

    diagnosis = data[:, 1]

    X = data[:, 2:].astype(
        np.float32
    )

    # Convert labels:
    #
    # M = 1
    # B = 0

    y = (
        diagnosis == "M"
    ).astype(
        np.int64
    )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ---------------------------------------------------------
    # Standardization
    #
    # IMPORTANT:
    # Fit scaler ONLY on training data.
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    ).astype(
        np.float32
    )

    X_test = scaler.transform(
        X_test
    ).astype(
        np.float32
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    )