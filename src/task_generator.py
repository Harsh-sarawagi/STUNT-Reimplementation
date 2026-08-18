"""
STUNT task generation.

Implements the STUNT task-generation procedure:

1. Sample a masking ratio p ~ Uniform(r1, r2).
2. Randomly select floor(p * d) feature columns.
3. Run K-means on the ORIGINAL values of the selected columns
   to create pseudo-labels.
4. Replace the selected columns with samples from their
   empirical marginal distributions.
5. Create a synthetic classification task.
6. Split the task into disjoint support/query sets.

Input X must be a 2-D numeric matrix.
Categorical variables should be preprocessed before reaching this module.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class STUNTTask:
    """Container for one generated STUNT few-shot task."""

    x: np.ndarray
    y: np.ndarray

    support_x: np.ndarray
    support_y: np.ndarray

    query_x: np.ndarray
    query_y: np.ndarray

    support_indices: np.ndarray
    query_indices: np.ndarray

    selected_columns: np.ndarray
    mask: np.ndarray
    mask_ratio: float


class STUNTTaskGenerator:
    """
    Generate synthetic few-shot classification tasks.

    Parameters
    ----------
    n_way:
        Number of pseudo-classes generated using K-means.

    k_shot:
        Number of support examples per class.

    q_query:
        Number of query examples per class.

    r1:
        Minimum feature masking ratio.

    r2:
        Maximum feature masking ratio.

    random_state:
        Random seed.

    max_attempts:
        Maximum attempts to generate a valid task.
    """

    def __init__(
        self,
        n_way: int,
        k_shot: int,
        q_query: int,
        r1: float = 0.2,
        r2: float = 0.5,
        random_state: Optional[int] = 42,
        max_attempts: int = 50,
    ):

        if n_way < 2:
            raise ValueError("n_way must be >= 2.")

        if k_shot < 1:
            raise ValueError("k_shot must be >= 1.")

        if q_query < 1:
            raise ValueError("q_query must be >= 1.")

        if not (0.0 < r1 <= r2 <= 1.0):
            raise ValueError(
                "Require 0 < r1 <= r2 <= 1."
            )

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be >= 1."
            )

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.r1 = r1
        self.r2 = r2

        self.max_attempts = max_attempts

        self.rng = np.random.default_rng(
            random_state
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @staticmethod
    def _validate_x(X: np.ndarray) -> np.ndarray:

        X = np.asarray(
            X,
            dtype=np.float32
        )

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D, got shape {X.shape}."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "X must contain at least one row."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

        return X

    # ---------------------------------------------------------
    # Step 1: Random feature mask
    # ---------------------------------------------------------

    def sample_mask(self, n_features: int):

        # Sample masking ratio
        p = float(
            self.rng.uniform(
                self.r1,
                self.r2
            )
        )

        # Number of columns to mask
        n_selected = int(
            np.floor(
                p * n_features
            )
        )

        # At least one feature
        n_selected = max(
            1,
            min(
                n_selected,
                n_features
            )
        )

        # Randomly select columns
        selected_columns = np.sort(
            self.rng.choice(
                n_features,
                size=n_selected,
                replace=False
            )
        )

        # Binary mask
        mask = np.zeros(
            n_features,
            dtype=bool
        )

        mask[selected_columns] = True

        return (
            p,
            selected_columns,
            mask
        )

    # ---------------------------------------------------------
    # Step 2: Generate pseudo-labels using K-means
    # ---------------------------------------------------------

    def generate_pseudo_labels(
        self,
        X: np.ndarray,
        selected_columns: np.ndarray,
    ):

        # IMPORTANT:
        # K-means operates on ORIGINAL selected columns.

        X_selected = X[
            :,
            selected_columns
        ]

        kmeans = KMeans(
            n_clusters=self.n_way,
            n_init=10,
            random_state=int(
                self.rng.integers(
                    0,
                    2**31 - 1
                )
            ),
        )

        labels = kmeans.fit_predict(
            X_selected
        )

        return labels.astype(
            np.int64
        )

    # ---------------------------------------------------------
    # Step 3: Perturb selected columns
    # ---------------------------------------------------------

    def perturb_selected_columns(
        self,
        X: np.ndarray,
        selected_columns: np.ndarray,
    ):
        """
        Permute the selected columns independently across rows.

        This matches the official STUNT implementation:
        each selected column is randomly permuted while
        preserving its empirical marginal distribution.
        """

        X_perturbed = X.copy()

        n_rows = X.shape[0]

        for column in selected_columns:

            permutation = self.rng.permutation(
                n_rows
            )

            X_perturbed[:, column] = (
                X_perturbed[
                    permutation,
                    column
                ]
            )

        return X_perturbed

    # ---------------------------------------------------------
    # Step 4: Support / Query split
    # ---------------------------------------------------------

    def sample_support_query(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):

        required_per_class = (
            self.k_shot +
            self.q_query
        )

        support_indices = []
        query_indices = []

        for cls in range(
            self.n_way
        ):

            class_indices = np.flatnonzero(
                y == cls
            )

            if len(class_indices) < required_per_class:

                raise ValueError(
                    f"Pseudo-class {cls} has "
                    f"{len(class_indices)} examples, "
                    f"but {required_per_class} "
                    f"are required."
                )

            chosen = self.rng.choice(
                class_indices,
                size=required_per_class,
                replace=False
            )

            support_indices.extend(
                chosen[:self.k_shot]
            )

            query_indices.extend(
                chosen[self.k_shot:]
            )

        support_indices = np.asarray(
            support_indices,
            dtype=np.int64
        )

        query_indices = np.asarray(
            query_indices,
            dtype=np.int64
        )

        # Shuffle examples
        self.rng.shuffle(
            support_indices
        )

        self.rng.shuffle(
            query_indices
        )

        return (
            X[support_indices],
            y[support_indices],

            X[query_indices],
            y[query_indices],

            support_indices,
            query_indices,
        )

    # ---------------------------------------------------------
    # Main function
    # ---------------------------------------------------------

    def generate(
        self,
        X: np.ndarray
    ) -> STUNTTask:

        X = self._validate_x(X)

        n_rows, n_features = X.shape

        minimum_rows = (
            self.n_way *
            (
                self.k_shot +
                self.q_query
            )
        )

        if n_rows < minimum_rows:

            raise ValueError(
                "Not enough rows for the requested "
                f"episode. Need at least "
                f"{minimum_rows}, got {n_rows}."
            )

        last_error = None

        for _ in range(
            self.max_attempts
        ):

            try:

                # -----------------------------------------
                # 1. Random feature mask
                # -----------------------------------------

                (
                    p,
                    selected_columns,
                    mask
                ) = self.sample_mask(
                    n_features
                )

                # -----------------------------------------
                # 2. Generate pseudo-labels
                # -----------------------------------------

                y = self.generate_pseudo_labels(
                    X,
                    selected_columns
                )

                # Check cluster sizes
                counts = np.bincount(
                    y,
                    minlength=self.n_way
                )

                required = (
                    self.k_shot +
                    self.q_query
                )

                if np.any(
                    counts < required
                ):

                    raise ValueError(
                        "K-means produced an "
                        "undersized pseudo-class."
                    )

                # -----------------------------------------
                # 3. Perturb selected columns
                # -----------------------------------------

                X_perturbed = (
                    self.perturb_selected_columns(
                        X,
                        selected_columns
                    )
                )

                # -----------------------------------------
                # 4. Support / Query
                # -----------------------------------------

                (
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    support_indices,
                    query_indices,
                ) = self.sample_support_query(
                    X_perturbed,
                    y
                )

                # -----------------------------------------
                # Return task
                # -----------------------------------------

                return STUNTTask(

                    x=X_perturbed,

                    y=y,

                    support_x=support_x,

                    support_y=support_y,

                    query_x=query_x,

                    query_y=query_y,

                    support_indices=support_indices,

                    query_indices=query_indices,

                    selected_columns=selected_columns,

                    mask=mask,

                    mask_ratio=p,
                )

            except ValueError as exc:

                last_error = exc

        raise RuntimeError(
            "Could not generate a valid STUNT "
            f"task after {self.max_attempts} "
            f"attempts. Last error: {last_error}"
        )