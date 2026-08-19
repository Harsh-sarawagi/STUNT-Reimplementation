"""
Meta-training loop for STUNT + ProtoNet.

Training is performed on batches of STUNT-generated
few-shot tasks, following the batched structure of
the official STUNT implementation.
"""

from dataclasses import dataclass

import numpy as np
import torch

from src.protonet import ProtoNet
from src.task_generator import STUNTTaskGenerator
from src.validation import real_validate
from src.validation import pseudo_validate




# =========================================================
# Training configuration
# =========================================================

@dataclass
class TrainingConfig:
    """
    Configuration for STUNT meta-training.
    """

    # WDBC is a binary classification problem.
    n_way: int = 2

    k_shot: int = 2

    q_query: int = 5

    # STUNT feature-selection ratio.
    mask_ratio_min: float = 0.2
    mask_ratio_max: float = 0.5

    # Number of outer/meta-training updates.
    episodes: int = 5

    # Number of tasks processed together.
    task_batch_size: int = 4

    learning_rate: float = 1e-3

    weight_decay: float = 1e-5

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    seed: int = 42

    validation_interval: int = 50


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int):
    """
    Make training reproducible.
    """

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Meta-training
# =========================================================

def train(
    X: np.ndarray,
    model: ProtoNet,
    config: TrainingConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
):
    """
    Meta-train ProtoNet using STUNT-generated tasks.

    The task generator creates individual few-shot tasks.
    A batch of tasks is stacked into:

        support_x: [B, N_support, features]
        support_y: [B, N_support]

        query_x:   [B, N_query, features]
        query_y:   [B, N_query]

    ProtoNet then processes the entire task batch in
    one forward/backward pass.
    """

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    set_seed(
        config.seed
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        config.device
    )

    print(
        f"Training device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -----------------------------------------------------
    # Move model to device
    # -----------------------------------------------------

    model = model.to(
        device
    )

    model.train()

    # -----------------------------------------------------
    # STUNT task generator
    # -----------------------------------------------------

    # Validate the full dataset once.
    X = np.asarray(
        X,
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Training data contains NaN or infinite values."
        )

    task_generator = STUNTTaskGenerator(

        n_way=config.n_way,

        k_shot=config.k_shot,

        q_query=config.q_query,

        r1=config.mask_ratio_min,

        r2=config.mask_ratio_max,

        random_state=config.seed,
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=config.learning_rate,

        weight_decay=config.weight_decay,
    )

    # -----------------------------------------------------
    # Training history
    # -----------------------------------------------------

    history = {
        "loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -float(
        "inf"
    )

    # =====================================================
    # Meta-training loop
    # =====================================================

    for episode in range(
        config.episodes
    ):

        model.train()

        optimizer.zero_grad()

        # -------------------------------------------------
        # Generate task batch
        # -------------------------------------------------

        support_x_list = []
        support_y_list = []

        query_x_list = []
        query_y_list = []

        for task_number in range(
            config.task_batch_size
        ):

            task = task_generator.generate(
                X
            )

            support_x_list.append(
                task.support_x
            )

            support_y_list.append(
                task.support_y
            )

            query_x_list.append(
                task.query_x
            )

            query_y_list.append(
                task.query_y
            )

        # -------------------------------------------------
        # Stack tasks
        #
        # Result:
        #
        # support_x:
        # [batch, n_way*k_shot, features]
        #
        # support_y:
        # [batch, n_way*k_shot]
        #
        # query_x:
        # [batch, n_way*q_query, features]
        #
        # query_y:
        # [batch, n_way*q_query]
        # -------------------------------------------------

        support_x = torch.tensor(
            np.stack(
                support_x_list,
                axis=0,
            ),
            dtype=torch.float32,
            device=device,
        )

        support_y = torch.tensor(
            np.stack(
                support_y_list,
                axis=0,
            ),
            dtype=torch.long,
            device=device,
        )

        query_x = torch.tensor(
            np.stack(
                query_x_list,
                axis=0,
            ),
            dtype=torch.float32,
            device=device,
        )

        query_y = torch.tensor(
            np.stack(
                query_y_list,
                axis=0,
            ),
            dtype=torch.long,
            device=device,
        )

        # -------------------------------------------------
        # Forward + loss
        #
        # ProtoNet now processes all tasks together.
        # -------------------------------------------------

        loss = model.loss(

            support_x,

            support_y,

            query_x,

            query_y,
        )

        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        loss.backward()

        optimizer.step()

        # -------------------------------------------------
        # Store loss
        # -------------------------------------------------

        loss_value = float(
            loss.detach().cpu()
        )

        history["loss"].append(
            loss_value
        )

        # -------------------------------------------------
        # Pseudo-validation
        # -------------------------------------------------

        if (
            X_val is not None
            and
            (episode + 1)
            % config.validation_interval
            == 0
        ):

            validation = real_validate(
                model=model,
                X_val=X_val,
                y_val=y_val,
                n_way=config.n_way,
                k_shot=config.k_shot,
                q_query=30,
                n_episodes=20,
                device=config.device,
                random_state=config.seed + episode,
            )

            val_accuracy = pseudo_validate(
                model=model,
                X_val=X_val,
                n_way=config.n_way,
                k_shot=config.k_shot,
                q_query=config.q_query,
                device=config.device,
                n_episodes=20,
                random_state=config.seed + episode,
            )

            history[
                "val_accuracy"
            ].append(
                val_accuracy
            )

            print(
                f"Pseudo-validation "
                f"| accuracy = "
                f"{val_accuracy:.4f}"
            )

            # -------------------------------------------------
            # Save best model
            # -------------------------------------------------

            if (
                val_accuracy
                >
                best_val_accuracy
            ):

                best_val_accuracy = (
                    val_accuracy
                )

                torch.save(
                    model.state_dict(),
                    "best_model.pt",
                )

                print(
                    "New best model saved."
                )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if (
            episode == 0
            or
            (episode + 1) % 10 == 0
        ):

            print(
                f"Episode "
                f"{episode + 1:04d}/"
                f"{config.episodes:04d} "
                f"| loss = "
                f"{loss_value:.4f}"
            )

    # =====================================================
    # Restore best model
    # =====================================================

    if (
        X_val is not None
        and
        history["val_accuracy"]
    ):

        checkpoint = torch.load(
            "best_model.pt",
            map_location=device,
        )

        model.load_state_dict(
            checkpoint
        )

        print()
        print(
            "Best model restored."
        )

        print(
            "Best pseudo-validation "
            f"accuracy: "
            f"{best_val_accuracy:.4f}"
        )

    return history