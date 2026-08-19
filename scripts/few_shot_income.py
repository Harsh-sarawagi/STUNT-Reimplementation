"""
Income few-shot evaluation.

STUNT:
    few-shot support
        -> frozen STUNT encoder
        -> Logistic Regression
        -> fixed test set

RAW:
    few-shot support
        -> raw features
        -> Logistic Regression
        -> fixed test set

The exact same support indices are used for STUNT and RAW.

Outputs:
    experiments/
        income_fewshot_YYYYMMDD_HHMMSS/
            results.json
            results.csv
            summary.txt
            accuracy.png
            f1.png
            config.json
            model.pt
"""

import sys
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# =========================================================
# PROJECT IMPORTS
# =========================================================

from src.data import load_income
from src.encoder import TabularEncoder
from src.protonet import ProtoNet


# =========================================================
# CONFIGURATION
# =========================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

RANDOM_STATE = 42

SHOTS = [
    1,
    2,
    5,
    10,
]

# ---------------------------------------------------------
# Use 100 first.
#
# For final paper-quality evaluation:
# change to 500.
# ---------------------------------------------------------

N_TRIALS = 100

INPUT_DIM = 105
HIDDEN_DIM = 1024
EMBEDDING_DIM = 1024


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / f"income_fewshot_{timestamp}"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# FEW-SHOT SAMPLING
# =========================================================

def make_few_shot_split(
    X,
    y,
    shots_per_class,
    random_state,
):
    """
    Select exactly `shots_per_class`
    examples from every class.

    Returns:
        X_support
        y_support
        support_indices
    """

    rng = np.random.RandomState(
        random_state
    )

    classes = np.unique(y)

    selected_indices = []

    for cls in classes:

        class_indices = np.where(
            y == cls
        )[0]

        if (
            len(class_indices)
            < shots_per_class
        ):
            raise ValueError(
                f"Class {cls} has only "
                f"{len(class_indices)} samples."
            )

        selected = rng.choice(
            class_indices,
            size=shots_per_class,
            replace=False,
        )

        selected_indices.extend(
            selected.tolist()
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    return (
        X[selected_indices],
        y[selected_indices],
        selected_indices,
    )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

def evaluate_logistic_regression(
    X_support,
    y_support,
    X_test,
    y_test,
):
    """
    Train Logistic Regression on the
    few-shot support set and evaluate
    on the fixed test set.
    """

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    classifier.fit(
        X_support,
        y_support,
    )

    predictions = classifier.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    return (
        float(accuracy),
        float(f1),
    )


# =========================================================
# SAVE CSV
# =========================================================

def save_csv(results):
    path = (
        RESULTS_DIR
        / "results.csv"
    )

    fieldnames = [
        "shot",

        "stunt_accuracy_mean",
        "stunt_accuracy_std",

        "raw_accuracy_mean",
        "raw_accuracy_std",

        "stunt_f1_mean",
        "stunt_f1_std",

        "raw_f1_mean",
        "raw_f1_std",

        "accuracy_gain",
        "f1_gain",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:

            writer.writerow(row)

    return path


# =========================================================
# SAVE JSON
# =========================================================

def save_json(results):
    path = (
        RESULTS_DIR
        / "results.json"
    )

    with open(
        path,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    return path


# =========================================================
# SAVE CONFIG
# =========================================================

def save_config():
    config = {
        "dataset": "income",

        "device": DEVICE,

        "random_state":
            RANDOM_STATE,

        "shots": SHOTS,

        "n_trials":
            N_TRIALS,

        "input_dim":
            INPUT_DIM,

        "hidden_dim":
            HIDDEN_DIM,

        "embedding_dim":
            EMBEDDING_DIM,

        "classifier":
            "LogisticRegression",

        "comparison": (
            "STUNT embeddings vs raw features"
        ),

        "same_support_indices":
            True,
    }

    path = (
        RESULTS_DIR
        / "config.json"
    )

    with open(
        path,
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )

    return path


# =========================================================
# SAVE SUMMARY
# =========================================================

def save_summary(results):
    path = (
        RESULTS_DIR
        / "summary.txt"
    )

    with open(
        path,
        "w",
    ) as f:

        f.write(
            "INCOME FEW-SHOT RESULTS\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            f"N trials: {N_TRIALS}\n"
        )

        f.write(
            f"Random state: "
            f"{RANDOM_STATE}\n\n"
        )

        f.write(
            f"{'Shot':<6}"
            f"{'STUNT Acc':<22}"
            f"{'Raw Acc':<22}"
            f"{'STUNT F1':<22}"
            f"{'Raw F1':<22}\n"
        )

        for r in results:

            f.write(
                f"{r['shot']:<6}"

                f"{r['stunt_accuracy_mean']:.4f} +/- "
                f"{r['stunt_accuracy_std']:.4f}    "

                f"{r['raw_accuracy_mean']:.4f} +/- "
                f"{r['raw_accuracy_std']:.4f}    "

                f"{r['stunt_f1_mean']:.4f} +/- "
                f"{r['stunt_f1_std']:.4f}    "

                f"{r['raw_f1_mean']:.4f} +/- "
                f"{r['raw_f1_std']:.4f}\n"
            )

        f.write(
            "\n"
            "STUNT IMPROVEMENT OVER RAW\n"
        )

        f.write(
            "=" * 80
            + "\n"
        )

        for r in results:

            f.write(
                f"{r['shot']}-shot: "
                f"Acc {r['accuracy_gain']:+.4f} | "
                f"F1 {r['f1_gain']:+.4f}\n"
            )

    return path


# =========================================================
# ACCURACY PLOT
# =========================================================

def save_accuracy_plot(results):

    shots = [
        r["shot"]
        for r in results
    ]

    stunt_mean = [
        r["stunt_accuracy_mean"]
        for r in results
    ]

    stunt_std = [
        r["stunt_accuracy_std"]
        for r in results
    ]

    raw_mean = [
        r["raw_accuracy_mean"]
        for r in results
    ]

    raw_std = [
        r["raw_accuracy_std"]
        for r in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        shots,
        stunt_mean,
        yerr=stunt_std,
        marker="o",
        capsize=4,
        label="STUNT",
    )

    plt.errorbar(
        shots,
        raw_mean,
        yerr=raw_std,
        marker="o",
        capsize=4,
        label="Raw",
    )

    plt.xlabel(
        "Shots per class"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Income Few-Shot Accuracy"
    )

    plt.xticks(
        shots
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    path = (
        RESULTS_DIR
        / "accuracy.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


# =========================================================
# F1 PLOT
# =========================================================

def save_f1_plot(results):

    shots = [
        r["shot"]
        for r in results
    ]

    stunt_mean = [
        r["stunt_f1_mean"]
        for r in results
    ]

    stunt_std = [
        r["stunt_f1_std"]
        for r in results
    ]

    raw_mean = [
        r["raw_f1_mean"]
        for r in results
    ]

    raw_std = [
        r["raw_f1_std"]
        for r in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        shots,
        stunt_mean,
        yerr=stunt_std,
        marker="o",
        capsize=4,
        label="STUNT",
    )

    plt.errorbar(
        shots,
        raw_mean,
        yerr=raw_std,
        marker="o",
        capsize=4,
        label="Raw",
    )

    plt.xlabel(
        "Shots per class"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "Income Few-Shot F1"
    )

    plt.xticks(
        shots
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    path = (
        RESULTS_DIR
        / "f1.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("INCOME FEW-SHOT EVALUATION")
    print("=" * 80)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Results directory:",
        RESULTS_DIR,
    )

    # =====================================================
    # 1. LOAD INCOME
    # =====================================================

    (
        X_train,
        X_val,
        y_train,
        y_val,
        X_test,
        y_test,
    ) = load_income()

    print()
    print(
        "Train:",
        X_train.shape,
    )

    print(
        "Validation:",
        X_val.shape,
    )

    print(
        "Test:",
        X_test.shape,
    )

    print(
        "Classes:",
        np.unique(y_train),
    )

    # =====================================================
    # 2. LOAD TRAINED MODEL
    # =====================================================

    model_path = (
        PROJECT_ROOT
        / "best_model.pt"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Could not find best_model.pt at:\n"
            f"{model_path}\n\n"
            "Put the trained model in the "
            "project root or change model_path."
        )

    encoder = TabularEncoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
    )

    model = ProtoNet(
        encoder
    ).to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    # -----------------------------------------------------
    # Support either:
    #
    # 1. state_dict directly
    # 2. {"model_state_dict": ...}
    # 3. {"state_dict": ...}
    # -----------------------------------------------------

    if isinstance(
        checkpoint,
        dict,
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print()
    print(
        "Loaded model:",
        model_path,
    )

    # =====================================================
    # 3. COMPUTE STUNT EMBEDDINGS
    # =====================================================

    print()
    print(
        "Computing STUNT embeddings..."
    )

    train_embeddings = []

    test_embeddings = []

    with torch.no_grad():

        # -------------------------------------------------
        # Process in batches to avoid excessive memory
        # -------------------------------------------------

        batch_size = 2048

        for start in range(
            0,
            len(X_train),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(X_train),
            )

            batch = torch.tensor(
                X_train[start:end],
                dtype=torch.float32,
                device=DEVICE,
            )

            embedding = (
                model.encoder(
                    batch
                )
                .cpu()
                .numpy()
            )

            train_embeddings.append(
                embedding
            )

        for start in range(
            0,
            len(X_test),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(X_test),
            )

            batch = torch.tensor(
                X_test[start:end],
                dtype=torch.float32,
                device=DEVICE,
            )

            embedding = (
                model.encoder(
                    batch
                )
                .cpu()
                .numpy()
            )

            test_embeddings.append(
                embedding
            )

    train_embeddings = np.concatenate(
        train_embeddings,
        axis=0,
    )

    test_embeddings = np.concatenate(
        test_embeddings,
        axis=0,
    )

    print(
        "STUNT embeddings:",
        train_embeddings.shape,
    )

    # =====================================================
    # 4. FEW-SHOT EVALUATION
    # =====================================================

    results = []

    for shots in SHOTS:

        print()
        print(
            "=" * 80
        )

        print(
            f"{shots}-SHOT EVALUATION"
        )

        print(
            "=" * 80
        )

        stunt_accuracy_values = []
        stunt_f1_values = []

        raw_accuracy_values = []
        raw_f1_values = []

        for trial in range(
            N_TRIALS
        ):

            trial_seed = (
                RANDOM_STATE
                + trial
            )

            # -------------------------------------------------
            # SAME SUPPORT SET FOR BOTH METHODS
            # -------------------------------------------------

            (
                X_support,
                y_support,
                support_indices,
            ) = make_few_shot_split(
                X_train,
                y_train,
                shots_per_class=shots,
                random_state=trial_seed,
            )

            # =================================================
            # STUNT
            # =================================================

            stunt_support = (
                train_embeddings[
                    support_indices
                ]
            )

            stunt_accuracy, stunt_f1 = (
                evaluate_logistic_regression(
                    X_support=stunt_support,
                    y_support=y_support,
                    X_test=test_embeddings,
                    y_test=y_test,
                )
            )

            # =================================================
            # RAW
            # =================================================

            raw_accuracy, raw_f1 = (
                evaluate_logistic_regression(
                    X_support=X_support,
                    y_support=y_support,
                    X_test=X_test,
                    y_test=y_test,
                )
            )

            # =================================================
            # Store trial
            # =================================================

            stunt_accuracy_values.append(
                stunt_accuracy
            )

            stunt_f1_values.append(
                stunt_f1
            )

            raw_accuracy_values.append(
                raw_accuracy
            )

            raw_f1_values.append(
                raw_f1
            )

            if (
                trial == 0
                or
                (trial + 1) % 10 == 0
            ):

                print(
                    f"Trial "
                    f"{trial + 1:03d}/"
                    f"{N_TRIALS:03d}"
                )

        # =====================================================
        # Statistics
        # =====================================================

        stunt_accuracy_mean = float(
            np.mean(
                stunt_accuracy_values
            )
        )

        stunt_accuracy_std = float(
            np.std(
                stunt_accuracy_values
            )
        )

        raw_accuracy_mean = float(
            np.mean(
                raw_accuracy_values
            )
        )

        raw_accuracy_std = float(
            np.std(
                raw_accuracy_values
            )
        )

        stunt_f1_mean = float(
            np.mean(
                stunt_f1_values
            )
        )

        stunt_f1_std = float(
            np.std(
                stunt_f1_values
            )
        )

        raw_f1_mean = float(
            np.mean(
                raw_f1_values
            )
        )

        raw_f1_std = float(
            np.std(
                raw_f1_values
            )
        )

        accuracy_gain = (
            stunt_accuracy_mean
            - raw_accuracy_mean
        )

        f1_gain = (
            stunt_f1_mean
            - raw_f1_mean
        )

        results.append(
            {
                "shot": shots,

                "stunt_accuracy_mean":
                    stunt_accuracy_mean,

                "stunt_accuracy_std":
                    stunt_accuracy_std,

                "raw_accuracy_mean":
                    raw_accuracy_mean,

                "raw_accuracy_std":
                    raw_accuracy_std,

                "stunt_f1_mean":
                    stunt_f1_mean,

                "stunt_f1_std":
                    stunt_f1_std,

                "raw_f1_mean":
                    raw_f1_mean,

                "raw_f1_std":
                    raw_f1_std,

                "accuracy_gain":
                    float(accuracy_gain),

                "f1_gain":
                    float(f1_gain),
            }
        )

    # =====================================================
    # 5. PRINT RESULTS
    # =====================================================

    print()
    print("=" * 80)
    print("INCOME FEW-SHOT RESULTS")
    print("=" * 80)

    print(
        f"{'Shot':<6}"
        f"{'STUNT Acc':<22}"
        f"{'Raw Acc':<22}"
        f"{'STUNT F1':<22}"
        f"{'Raw F1':<22}"
    )

    for r in results:

        print(
            f"{r['shot']:<6}"

            f"{r['stunt_accuracy_mean']:.4f} +/- "
            f"{r['stunt_accuracy_std']:.4f}    "

            f"{r['raw_accuracy_mean']:.4f} +/- "
            f"{r['raw_accuracy_std']:.4f}    "

            f"{r['stunt_f1_mean']:.4f} +/- "
            f"{r['stunt_f1_std']:.4f}    "

            f"{r['raw_f1_mean']:.4f} +/- "
            f"{r['raw_f1_std']:.4f}"
        )

    print()
    print("=" * 80)
    print("STUNT IMPROVEMENT OVER RAW")
    print("=" * 80)

    for r in results:

        print(
            f"{r['shot']}-shot:"
            f"  Acc {r['accuracy_gain']:+.4f}"
            f"  | F1 {r['f1_gain']:+.4f}"
        )

    # =====================================================
    # 6. SAVE EVERYTHING
    # =====================================================

    print()
    print(
        "Saving results..."
    )

    save_csv(
        results
    )

    save_json(
        results
    )

    save_config()

    save_summary(
        results
    )

    save_accuracy_plot(
        results
    )

    save_f1_plot(
        results
    )

    # -----------------------------------------------------
    # Save a copy of the model used for evaluation
    # -----------------------------------------------------

    shutil.copy2(
        model_path,
        RESULTS_DIR
        / "model.pt",
    )

    # =====================================================
    # DONE
    # =====================================================

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()