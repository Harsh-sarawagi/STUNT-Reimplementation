import numpy as np
import pytest

from src.task_generator import (
    STUNTTaskGenerator
)


def make_toy_data(
    n_rows=120,
    n_features=10,
    seed=0
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
        4.0
    ]

    for center in centers:

        part = rng.normal(
            loc=center,
            scale=0.5,
            size=(
                rows_per_class,
                n_features
            )
        )

        parts.append(part)

    return np.vstack(
        parts
    ).astype(
        np.float32
    )


def test_generate_task_shapes():

    X = make_toy_data()

    generator = STUNTTaskGenerator(
        n_way=3,
        k_shot=2,
        q_query=5,
        r1=0.2,
        r2=0.5,
        random_state=42,
    )

    task = generator.generate(
        X
    )

    assert task.x.shape == X.shape

    assert task.y.shape == (
        len(X),
    )

    assert task.support_x.shape == (
        3 * 2,
        X.shape[1]
    )

    assert task.support_y.shape == (
        3 * 2,
    )

    assert task.query_x.shape == (
        3 * 5,
        X.shape[1]
    )

    assert task.query_y.shape == (
        3 * 5,
    )


def test_support_query_disjoint():

    X = make_toy_data()

    generator = STUNTTaskGenerator(
        n_way=3,
        k_shot=2,
        q_query=5,
        random_state=7,
    )

    task = generator.generate(
        X
    )

    assert set(
        task.support_indices
    ).isdisjoint(
        set(
            task.query_indices
        )
    )


def test_class_counts():

    X = make_toy_data()

    generator = STUNTTaskGenerator(
        n_way=3,
        k_shot=2,
        q_query=5,
        random_state=11,
    )

    task = generator.generate(
        X
    )

    for cls in range(3):

        assert np.sum(
            task.support_y == cls
        ) == 2

        assert np.sum(
            task.query_y == cls
        ) == 5


def test_selected_columns_are_perturbed():

    X = make_toy_data()

    generator = STUNTTaskGenerator(
        n_way=3,
        k_shot=2,
        q_query=5,
        random_state=42,
    )

    task = generator.generate(
        X
    )

    selected = (
        task.selected_columns
    )

    unselected = np.flatnonzero(
        ~task.mask
    )

    # Unselected columns should
    # remain unchanged.
    np.testing.assert_array_equal(
        task.x[:, unselected],
        X[:, unselected]
    )

    # Every generated value in a
    # selected column must come from
    # that original column's marginal.
    for col in selected:

        original_values = set(
            X[:, col].tolist()
        )

        generated_values = set(
            task.x[:, col].tolist()
        )

        assert generated_values.issubset(
            original_values
        )


def test_invalid_input():

    generator = STUNTTaskGenerator(
        n_way=3,
        k_shot=2,
        q_query=5,
    )

    with pytest.raises(
        ValueError
    ):

        generator.generate(
            np.array(
                [1, 2, 3]
            )
        )

    with pytest.raises(
        ValueError
    ):

        generator.generate(
            np.array([
                [1.0, np.nan],
                [2.0, 3.0]
            ])
        )