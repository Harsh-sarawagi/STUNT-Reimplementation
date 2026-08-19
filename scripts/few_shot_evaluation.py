"""
Repeated few-shot evaluation of STUNT representations
vs raw features.

Both methods receive exactly the same labeled examples
for every trial.

STUNT:
    few-shot labeled data
        -> frozen STUNT encoder
        -> embeddings
        -> Logistic Regression

Baseline:
    few-shot labeled data
        -> raw features
        -> Logistic Regression

For every shot level, the experiment is repeated over
multiple randomly selected support sets.

The final result reports mean +/- standard deviation.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import numpy as np
import torch

import json
import shutil
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

import json
import csv
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data import load_wdbc
from src.encoder import TabularEncoder


# =========================================================
# Configuration
# =========================================================

RANDOM_STATE = 42

SHOT_LEVELS = [
    1,
    2,
    5,
    10,
]

# Number of independent support sets per shot level.
#
# 30 is enough for a first experiment and is still
# reasonably fast on your laptop.
#
# Later, if needed, we can increase this to 100
# to match the paper's repeated-seed style more closely.
N_TRIALS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BEST_MODEL_PATH = (
    PROJECT_ROOT / "best_model.pt"
)


# =========================================================
# Utility: create balanced few-shot subset
# =========================================================

def make_few_shot_split(
    X,
    y,
    shots_per_class,
    random_state=42,
):
    """
    Select exactly `shots_per_class` examples
    from every class.

    The selected examples are returned together
    with their original indices.

    The SAME selected examples are used for
    both STUNT and the raw-feature baseline.
    """

    rng = np.random.default_rng(
        random_state
    )

    selected_indices = []

    classes = np.unique(y)

    for cls in classes:

        class_indices = np.flatnonzero(
            y == cls
        )

        if len(class_indices) < shots_per_class:

            raise ValueError(
                f"Not enough samples for class "
                f"{cls} to create "
                f"{shots_per_class}-shot split."
            )

        chosen = rng.choice(
            class_indices,
            size=shots_per_class,
            replace=False,
        )

        selected_indices.extend(
            chosen.tolist()
        )

    selected_indices = np.array(
        selected_indices,
        dtype=int,
    )

    # Shuffle support examples.
    rng.shuffle(
        selected_indices
    )

    return (
        X[selected_indices],
        y[selected_indices],
        selected_indices,
    )


# =========================================================
# STUNT embeddings
# =========================================================

def get_embeddings(
    encoder,
    X,
    device="cpu",
):
    """
    Convert tabular rows into frozen
    STUNT embeddings.
    """

    encoder.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():

        embeddings = encoder(
            X_tensor
        )

    return embeddings.cpu().numpy()


# =========================================================
# Evaluation
# =========================================================

def evaluate_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train Logistic Regression and return
    classification metrics.
    """

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
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
    }


# =========================================================
# Statistics
# =========================================================

def summarize(values):
    """
    Return mean and standard deviation.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(np.mean(values)),
        float(np.std(values)),
    )

def save_experiment_results(results):
    """
    Save few-shot results, plots, and the model checkpoint.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        PROJECT_ROOT
        / "experiments"
        / f"wdbc_fewshot_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    with open(
        output_dir / "results.json",
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    df = pd.DataFrame(results)

    df.to_csv(
        output_dir / "results.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Accuracy plot
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        df["shots"],
        df["stunt_accuracy_mean"],
        yerr=df["stunt_accuracy_std"],
        marker="o",
        capsize=4,
        label="STUNT",
    )

    plt.errorbar(
        df["shots"],
        df["raw_accuracy_mean"],
        yerr=df["raw_accuracy_std"],
        marker="o",
        capsize=4,
        label="Raw",
    )

    plt.xlabel("Shots per class")
    plt.ylabel("Accuracy")
    plt.title("STUNT vs Raw - Few-Shot Accuracy")
    plt.xticks(df["shots"])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "accuracy.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------------------------------
    # F1 plot
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        df["shots"],
        df["stunt_f1_mean"],
        yerr=df["stunt_f1_std"],
        marker="o",
        capsize=4,
        label="STUNT",
    )

    plt.errorbar(
        df["shots"],
        df["raw_f1_mean"],
        yerr=df["raw_f1_std"],
        marker="o",
        capsize=4,
        label="Raw",
    )

    plt.xlabel("Shots per class")
    plt.ylabel("F1 score")
    plt.title("STUNT vs Raw - Few-Shot F1")
    plt.xticks(df["shots"])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "f1.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------------------------------
    # Copy model checkpoint
    # -----------------------------------------------------

    if BEST_MODEL_PATH.exists():

        shutil.copy2(
            BEST_MODEL_PATH,
            output_dir / "model.pt",
        )

    # -----------------------------------------------------
    # Save configuration
    # -----------------------------------------------------

    config = {
        "random_state": RANDOM_STATE,
        "shot_levels": SHOT_LEVELS,
        "n_trials": N_TRIALS,
        "device": DEVICE,
        "model_checkpoint": str(
            BEST_MODEL_PATH
        ),
    }

    with open(
        output_dir / "config.json",
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(
        f"Directory: {output_dir}"
    )
    print(
        f"CSV:       {output_dir / 'results.csv'}"
    )
    print(
        f"JSON:      {output_dir / 'results.json'}"
    )
    print(
        f"Accuracy:  {output_dir / 'accuracy.png'}"
    )
    print(
        f"F1:        {output_dir / 'f1.png'}"
    )
    print(
        f"Model:     {output_dir / 'model.pt'}"
    )
# =========================================================
# Main experiment
# =========================================================

def main():

    # -----------------------------------------------------
    # Load WDBC
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    ) = load_wdbc()

    print("=" * 70)
    print("REPEATED FEW-SHOT EVALUATION")
    print("=" * 70)

    print(
        "Training samples:",
        X_train.shape[0],
    )

    print(
        "Test samples:",
        X_test.shape[0],
    )

    print(
        "Features:",
        X_train.shape[1],
    )

    print(
        "Trials per shot:",
        N_TRIALS,
    )

    print(
        "Device:",
        DEVICE,
    )

    print()

    # -----------------------------------------------------
    # Load trained STUNT encoder
    # -----------------------------------------------------

    if not BEST_MODEL_PATH.exists():

        raise FileNotFoundError(
            "best_model.pt was not found.\n"
            "Run scripts/train_wdbc.py first."
        )

    # IMPORTANT:
    # This must exactly match the architecture
    # used during STUNT training.

    encoder = TabularEncoder(
        input_dim=X_train.shape[1],
        hidden_dim=1024,
        embedding_dim=1024,
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
    )

    # best_model.pt contains the complete
    # ProtoNet state dict.
    #
    # Encoder parameters therefore have
    # the prefix "encoder.".

    encoder_state = {}

    for key, value in checkpoint.items():

        if key.startswith(
            "encoder."
        ):

            new_key = key[
                len("encoder.") :
            ]

            encoder_state[
                new_key
            ] = value

    encoder.load_state_dict(
        encoder_state
    )

    encoder = encoder.to(
        DEVICE
    )

    encoder.eval()

    print(
        "Loaded STUNT encoder:",
        BEST_MODEL_PATH,
    )

    print()

    # -----------------------------------------------------
    # Pre-compute STUNT embeddings
    # -----------------------------------------------------

    #
    # The encoder is frozen.
    #
    # We only need to calculate embeddings
    # once because the encoder does not change
    # during few-shot evaluation.
    #

    train_embeddings = get_embeddings(
        encoder,
        X_train,
        DEVICE,
    )

    test_embeddings = get_embeddings(
        encoder,
        X_test,
        DEVICE,
    )

    print(
        "STUNT embedding dimension:",
        train_embeddings.shape[1],
    )

    print()

    # -----------------------------------------------------
    # Store final results
    # -----------------------------------------------------

    results = []

    # -----------------------------------------------------
    # Few-shot experiments
    # -----------------------------------------------------

    for shots in SHOT_LEVELS:

        print("=" * 70)

        print(
            f"{shots}-SHOT PER CLASS"
        )

        print(
            f"Running {N_TRIALS} independent trials..."
        )

        print("-" * 70)

        # -------------------------------------------------
        # Per-trial results
        # -------------------------------------------------

        stunt_accuracy_values = []
        stunt_precision_values = []
        stunt_recall_values = []
        stunt_f1_values = []

        raw_accuracy_values = []
        raw_precision_values = []
        raw_recall_values = []
        raw_f1_values = []

        # -------------------------------------------------
        # Repeat experiment
        # -------------------------------------------------

        for trial in range(
            N_TRIALS
        ):

            trial_seed = (
                RANDOM_STATE
                + trial
            )

            # -------------------------------------------------
            # Create support set
            # -------------------------------------------------

            (
                X_few,
                y_few,
                selected_indices,
            ) = make_few_shot_split(
                X_train,
                y_train,
                shots_per_class=shots,
                random_state=trial_seed,
            )

            # -------------------------------------------------
            # STUNT
            # -------------------------------------------------

            stunt_X_few = (
                train_embeddings[
                    selected_indices
                ]
            )

            stunt_metrics = (
                evaluate_classifier(
                    stunt_X_few,
                    y_few,
                    test_embeddings,
                    y_test,
                )
            )

            # -------------------------------------------------
            # RAW FEATURES
            # -------------------------------------------------

            raw_metrics = (
                evaluate_classifier(
                    X_few,
                    y_few,
                    X_test,
                    y_test,
                )
            )

            # -------------------------------------------------
            # Store STUNT metrics
            # -------------------------------------------------

            stunt_accuracy_values.append(
                stunt_metrics[
                    "accuracy"
                ]
            )

            stunt_precision_values.append(
                stunt_metrics[
                    "precision"
                ]
            )

            stunt_recall_values.append(
                stunt_metrics[
                    "recall"
                ]
            )

            stunt_f1_values.append(
                stunt_metrics[
                    "f1"
                ]
            )

            # -------------------------------------------------
            # Store RAW metrics
            # -------------------------------------------------

            raw_accuracy_values.append(
                raw_metrics[
                    "accuracy"
                ]
            )

            raw_precision_values.append(
                raw_metrics[
                    "precision"
                ]
            )

            raw_recall_values.append(
                raw_metrics[
                    "recall"
                ]
            )

            raw_f1_values.append(
                raw_metrics[
                    "f1"
                ]
            )

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                trial == 0
                or
                (trial + 1) % 10 == 0
            ):

                print(
                    f"Trial "
                    f"{trial + 1:02d}/"
                    f"{N_TRIALS:02d} complete"
                )

        # -------------------------------------------------
        # Calculate statistics
        # -------------------------------------------------

        stunt_acc_mean, stunt_acc_std = (
            summarize(
                stunt_accuracy_values
            )
        )

        stunt_precision_mean, stunt_precision_std = (
            summarize(
                stunt_precision_values
            )
        )

        stunt_recall_mean, stunt_recall_std = (
            summarize(
                stunt_recall_values
            )
        )

        stunt_f1_mean, stunt_f1_std = (
            summarize(
                stunt_f1_values
            )
        )

        raw_acc_mean, raw_acc_std = (
            summarize(
                raw_accuracy_values
            )
        )

        raw_precision_mean, raw_precision_std = (
            summarize(
                raw_precision_values
            )
        )

        raw_recall_mean, raw_recall_std = (
            summarize(
                raw_recall_values
            )
        )

        raw_f1_mean, raw_f1_std = (
            summarize(
                raw_f1_values
            )
        )

        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

        print()

        print(
            "STUNT:"
        )

        print(
            f"  Accuracy:  "
            f"{stunt_acc_mean:.4f} "
            f"+/- {stunt_acc_std:.4f}"
        )

        print(
            f"  Precision: "
            f"{stunt_precision_mean:.4f} "
            f"+/- {stunt_precision_std:.4f}"
        )

        print(
            f"  Recall:    "
            f"{stunt_recall_mean:.4f} "
            f"+/- {stunt_recall_std:.4f}"
        )

        print(
            f"  F1:        "
            f"{stunt_f1_mean:.4f} "
            f"+/- {stunt_f1_std:.4f}"
        )

        print()

        print(
            "RAW FEATURES:"
        )

        print(
            f"  Accuracy:  "
            f"{raw_acc_mean:.4f} "
            f"+/- {raw_acc_std:.4f}"
        )

        print(
            f"  Precision: "
            f"{raw_precision_mean:.4f} "
            f"+/- {raw_precision_std:.4f}"
        )

        print(
            f"  Recall:    "
            f"{raw_recall_mean:.4f} "
            f"+/- {raw_recall_std:.4f}"
        )

        print(
            f"  F1:        "
            f"{raw_f1_mean:.4f} "
            f"+/- {raw_f1_std:.4f}"
        )

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        results.append(
            {
                "shots": shots,

                "stunt_accuracy_mean":
                    stunt_acc_mean,

                "stunt_accuracy_std":
                    stunt_acc_std,

                "stunt_precision_mean":
                    stunt_precision_mean,

                "stunt_precision_std":
                    stunt_precision_std,

                "stunt_recall_mean":
                    stunt_recall_mean,

                "stunt_recall_std":
                    stunt_recall_std,

                "stunt_f1_mean":
                    stunt_f1_mean,

                "stunt_f1_std":
                    stunt_f1_std,

                "raw_accuracy_mean":
                    raw_acc_mean,

                "raw_accuracy_std":
                    raw_acc_std,

                "raw_precision_mean":
                    raw_precision_mean,

                "raw_precision_std":
                    raw_precision_std,

                "raw_recall_mean":
                    raw_recall_mean,

                "raw_recall_std":
                    raw_recall_std,

                "raw_f1_mean":
                    raw_f1_mean,

                "raw_f1_std":
                    raw_f1_std,
            }
        )

        print()

    # =====================================================
    # Final comparison
    # =====================================================

    print()
    print("=" * 90)
    print("FINAL REPEATED FEW-SHOT COMPARISON")
    print("=" * 90)

    print(
        f"{'Shot':<8}"
        f"{'STUNT Acc':<20}"
        f"{'Raw Acc':<20}"
        f"{'STUNT F1':<20}"
        f"{'Raw F1':<20}"
    )

    print("-" * 90)

    for result in results:

        stunt_acc = (
            f"{result['stunt_accuracy_mean']:.4f}"
            f" +/- "
            f"{result['stunt_accuracy_std']:.4f}"
        )

        raw_acc = (
            f"{result['raw_accuracy_mean']:.4f}"
            f" +/- "
            f"{result['raw_accuracy_std']:.4f}"
        )

        stunt_f1 = (
            f"{result['stunt_f1_mean']:.4f}"
            f" +/- "
            f"{result['stunt_f1_std']:.4f}"
        )

        raw_f1 = (
            f"{result['raw_f1_mean']:.4f}"
            f" +/- "
            f"{result['raw_f1_std']:.4f}"
        )

        print(
            f"{result['shots']:<8}"
            f"{stunt_acc:<20}"
            f"{raw_acc:<20}"
            f"{stunt_f1:<20}"
            f"{raw_f1:<20}"
        )

    print()
    print(
        "Evaluation complete."
    )
    save_experiment_results(
        results
    )

    


if __name__ == "__main__":
    main()