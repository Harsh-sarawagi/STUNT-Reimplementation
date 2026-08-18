import numpy as np
import torch

from src.encoder import TabularEncoder
from src.protonet import ProtoNet
from src.validation import (
    generate_validation_labels,
    make_validation_episode,
    pseudo_validate,
)


def make_toy_data(
    n_rows=120,
    n_features=10,
    seed=0,
):

    rng = np.random.default_rng(
        seed
    )

    rows_per_class = (
        n_rows // 3
    )

    parts = []

    centers = [
        -4.0,
        0.0,
        4.0,
    ]

    for center in centers:

        part = rng.normal(
            loc=center,
            scale=0.5,
            size=(
                rows_per_class,
                n_features,
            ),
        )

        parts.append(part)

    return np.vstack(
        parts
    ).astype(
        np.float32
    )


def test_validation_labels():

    X = make_toy_data()

    labels = generate_validation_labels(
        X,
        n_way=3,
        random_state=42,
    )

    assert labels.shape == (
        len(X),
    )

    assert set(
        np.unique(labels)
    ) == {0, 1, 2}


def test_validation_episode():

    X = make_toy_data()

    (
        support_x,
        support_y,
        query_x,
        query_y,
    ) = make_validation_episode(
        X,
        n_way=3,
        k_shot=2,
        q_query=5,
        random_state=42,
    )

    assert support_x.shape == (
        6,
        10,
    )

    assert support_y.shape == (
        6,
    )

    assert query_x.shape == (
        15,
        10,
    )

    assert query_y.shape == (
        15,
    )


def test_pseudo_validate():

    X = make_toy_data()

    encoder = TabularEncoder(
        input_dim=10
    )

    model = ProtoNet(
        encoder
    )

    accuracy = pseudo_validate(
        model=model,
        X_val=X,
        n_way=3,
        k_shot=2,
        q_query=5,
        device="cpu",
        random_state=42,
    )

    assert 0.0 <= accuracy <= 1.0