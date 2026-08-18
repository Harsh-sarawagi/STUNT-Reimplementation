import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import numpy as np

from src.task_generator import STUNTTaskGenerator


# ---------------------------------------------------------
# Create toy unlabeled data
# ---------------------------------------------------------

rng = np.random.default_rng(
    42
)

X = np.vstack([

    rng.normal(
        -4,
        0.5,
        size=(40, 10)
    ),

    rng.normal(
        0,
        0.5,
        size=(40, 10)
    ),

    rng.normal(
        4,
        0.5,
        size=(40, 10)
    ),

]).astype(
    np.float32
)


# ---------------------------------------------------------
# Create STUNT generator
# ---------------------------------------------------------

generator = STUNTTaskGenerator(

    n_way=3,

    k_shot=2,

    q_query=5,

    r1=0.2,

    r2=0.5,

    random_state=42,
)


# ---------------------------------------------------------
# Generate task
# ---------------------------------------------------------

task = generator.generate(
    X
)


# ---------------------------------------------------------
# Print information
# ---------------------------------------------------------

print(
    "Original X:",
    X.shape
)

print(
    "Perturbed X:",
    task.x.shape
)

print(
    "Selected columns:",
    task.selected_columns.tolist()
)

print(
    "Mask ratio:",
    round(
        task.mask_ratio,
        4
    )
)

print(
    "Pseudo-label counts:",
    np.bincount(
        task.y
    ).tolist()
)

print(
    "Support:",
    task.support_x.shape,
    task.support_y.shape
)

print(
    "Query:",
    task.query_x.shape,
    task.query_y.shape
)

print(
    "Support indices:",
    task.support_indices.tolist()
)

print(
    "Query indices:",
    task.query_indices.tolist()
)