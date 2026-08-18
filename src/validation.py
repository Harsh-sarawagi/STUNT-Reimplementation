"""
Pseudo-validation for STUNT.

Validation uses all features of the validation subset
and K-means-generated pseudo-labels.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans


def generate_validation_labels(
    X: np.ndarray,
    n_way: int,
    random_state: int = 42,
):
    """
    Generate pseudo-labels using K-means on all features.

    Unlike STUNT training tasks, validation does not
    randomly mask or perturb features.
    """

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D array."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "X contains NaN or infinite values."
        )

    kmeans = KMeans(
        n_clusters=n_way,
        random_state=random_state,
        n_init=10,
    )

    labels = kmeans.fit_predict(
        X
    )

    return labels

def make_validation_episode(
    X: np.ndarray,
    n_way: int,
    k_shot: int,
    q_query: int,
    random_state: int = 42,
):
    """
    Create a pseudo-validation episode.

    Pseudo-labels are generated using K-means over
    all features. No feature masking or perturbation
    is performed.
    """

    labels = generate_validation_labels(
        X=X,
        n_way=n_way,
        random_state=random_state,
    )

    rng = np.random.default_rng(
        random_state
    )

    support_indices = []
    query_indices = []

    for cls in range(n_way):

        class_indices = np.flatnonzero(
            labels == cls
        )

        required = (
            k_shot + q_query
        )

        if len(class_indices) < required:
            raise ValueError(
                f"Class {cls} has only "
                f"{len(class_indices)} samples, "
                f"but {required} are required."
            )

        selected = rng.choice(
            class_indices,
            size=required,
            replace=False,
        )

        support_indices.extend(
            selected[:k_shot]
        )

        query_indices.extend(
            selected[k_shot:]
        )

    support_indices = np.asarray(
        support_indices,
        dtype=np.int64,
    )

    query_indices = np.asarray(
        query_indices,
        dtype=np.int64,
    )

    return (
        X[support_indices],
        labels[support_indices],
        X[query_indices],
        labels[query_indices],
    )

def pseudo_validate(
    model,
    X_val: np.ndarray,
    n_way: int,
    k_shot: int,
    q_query: int,
    device: str = "cpu",
    random_state: int = 42,
    n_episodes: int = 20,
):
    """
    Evaluate a STUNT model using multiple
    pseudo-validation episodes.

    Each episode uses a different random seed.
    The final validation accuracy is the mean
    accuracy across all episodes.
    """

    device = torch.device(device)

    was_training = model.training

    model.eval()

    accuracies = []

    for episode in range(n_episodes):

        episode_seed = (
            random_state + episode
        )

        (
            support_x,
            support_y,
            query_x,
            query_y,
        ) = make_validation_episode(
            X=X_val,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            random_state=episode_seed,
        )

        support_x = torch.tensor(
            support_x,
            dtype=torch.float32,
            device=device,
        )

        support_y = torch.tensor(
            support_y,
            dtype=torch.long,
            device=device,
        )

        query_x = torch.tensor(
            query_x,
            dtype=torch.float32,
            device=device,
        )

        query_y = torch.tensor(
            query_y,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():

            predictions = model.predict(
                support_x,
                support_y,
                query_x,
            )

        accuracy = (
            (predictions == query_y)
            .float()
            .mean()
            .item()
        )

        accuracies.append(accuracy)

    if was_training:
        model.train()

    return float(
        np.mean(accuracies)
    )

def real_validate(
    model,
    X_val,
    y_val,
    n_way,
    k_shot,
    q_query,
    device="cpu",
    n_episodes=20,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    model.eval()
    accuracies = []

    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    classes = np.unique(y_val)

    if len(classes) < n_way:
        raise ValueError("Validation set has fewer classes than n_way.")

    with torch.no_grad():

        for episode in range(n_episodes):

            selected_classes = rng.choice(
                classes,
                size=n_way,
                replace=False,
            )

            support_indices = []
            query_indices = []

            for cls in selected_classes:

                class_indices = np.flatnonzero(
                    y_val == cls
                )

                required = k_shot + q_query

                if len(class_indices) < required:
                    raise ValueError(
                        f"Class {cls} has only "
                        f"{len(class_indices)} samples."
                    )

                selected = rng.choice(
                    class_indices,
                    size=required,
                    replace=False,
                )

                support_indices.extend(
                    selected[:k_shot]
                )

                query_indices.extend(
                    selected[k_shot:]
                )

            support_indices = np.asarray(
                support_indices,
                dtype=np.int64,
            )

            query_indices = np.asarray(
                query_indices,
                dtype=np.int64,
            )

            support_x = torch.tensor(
                X_val[support_indices],
                dtype=torch.float32,
                device=device,
            )

            query_x = torch.tensor(
                X_val[query_indices],
                dtype=torch.float32,
                device=device,
            )

            # Convert the selected real classes to 0...n_way-1
            support_y = np.concatenate([
                np.full(k_shot, i)
                for i in range(n_way)
            ])

            query_y = np.concatenate([
                np.full(q_query, i)
                for i in range(n_way)
            ])

            support_y = torch.tensor(
                support_y,
                dtype=torch.long,
                device=device,
            )

            query_y = torch.tensor(
                query_y,
                dtype=torch.long,
                device=device,
            )

            predictions = model.predict(
                support_x,
                support_y,
                query_x,
            )

            accuracy = (
                predictions == query_y
            ).float().mean().item()

            accuracies.append(accuracy)

    model.train()

    return float(np.mean(accuracies))