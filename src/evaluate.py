"""
Downstream evaluation for a trained STUNT encoder.
"""

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def get_embeddings(
    encoder,
    X,
    device="cpu",
):
    """
    Convert tabular data into embeddings using
    the trained encoder.

    The input tensor is explicitly moved to the
    same device as the encoder.
    """

    encoder.eval()

    # ---------------------------------------------------------
    # Make sure we use the same device as the encoder.
    # ---------------------------------------------------------

    model_device = next(
        encoder.parameters()
    ).device

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=model_device,
    )

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------

    with torch.no_grad():

        embeddings = encoder(
            X_tensor
        )

    # LogisticRegression expects NumPy arrays,
    # so move embeddings back to CPU.
    return embeddings.detach().cpu().numpy()


def evaluate_embeddings(
    encoder,
    X_train,
    y_train,
    X_test,
    y_test,
    device="cpu",
):
    """
    Train a simple classifier on frozen STUNT
    embeddings and evaluate it.
    """

    # ---------------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------------

    train_embeddings = get_embeddings(
        encoder,
        X_train,
        device,
    )

    test_embeddings = get_embeddings(
        encoder,
        X_test,
        device,
    )

    # ---------------------------------------------------------
    # Downstream classifier
    # ---------------------------------------------------------

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    classifier.fit(
        train_embeddings,
        y_train,
    )

    predictions = classifier.predict(
        test_embeddings
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    results = {

        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions,
            ),
    }

    return results

def evaluate_raw(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Evaluate raw tabular features using the same
    downstream classifier as the embedding evaluation.

    No STUNT encoder is used.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = classifier.predict(
        X_test
    )

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ),
    }