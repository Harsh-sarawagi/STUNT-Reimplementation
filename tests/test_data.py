import numpy as np

from src.data import load_wdbc


def test_load_wdbc():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    ) = load_wdbc()

    # WDBC has 30 numerical features.
    assert X_train.shape[1] == 30
    assert X_test.shape[1] == 30

    # Features and labels should have
    # matching numbers of samples.
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # Labels should be binary.
    assert set(
        np.unique(y_train)
    ).issubset({0, 1})

    assert set(
        np.unique(y_test)
    ).issubset({0, 1})

    # Standardized training data should
    # approximately have mean 0.
    assert np.allclose(
        X_train.mean(axis=0),
        0,
        atol=1e-5,
    )