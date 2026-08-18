import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

import numpy as np

from src.encoder import TabularEncoder
from src.protonet import ProtoNet
from src.train import (
    TrainingConfig,
    train,
)


# ---------------------------------------------------------
# Create toy unlabeled dataset
# ---------------------------------------------------------

rng = np.random.default_rng(
    42
)

X = np.vstack([

    rng.normal(
        -4,
        0.5,
        size=(100, 10)
    ),

    rng.normal(
        0,
        0.5,
        size=(100, 10)
    ),

    rng.normal(
        4,
        0.5,
        size=(100, 10)
    ),

]).astype(
    np.float32
)


# ---------------------------------------------------------
# Encoder
# ---------------------------------------------------------

encoder = TabularEncoder(

    input_dim=10,

    hidden_dim=64,

    embedding_dim=32,
)


# ---------------------------------------------------------
# ProtoNet
# ---------------------------------------------------------

model = ProtoNet(
    encoder
)


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

config = TrainingConfig(

    n_way=3,

    k_shot=2,

    q_query=5,

    episodes=100,

    learning_rate=1e-3,

    seed=42,
)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

history = train(

    X=X,

    model=model,

    config=config,
)


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print()

print(
    "Training complete."
)

print(
    "Initial loss:",
    round(
        history["loss"][0],
        4
    )
)

print(
    "Final loss:",
    round(
        history["loss"][-1],
        4
    )
)