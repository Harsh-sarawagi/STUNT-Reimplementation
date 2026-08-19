from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "census+income"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "income"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Adult dataset columns
# =========================================================

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


NUMERICAL_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# =========================================================
# Find raw files
# =========================================================

def find_file(filename):

    matches = list(
        RAW_DIR.rglob(filename)
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} "
            f"inside {RAW_DIR}"
        )

    return matches[0]


TRAIN_FILE = find_file(
    "adult.data"
)

TEST_FILE = find_file(
    "adult.test"
)


# =========================================================
# Load Adult dataset
# =========================================================

def load_adult_train():

    df = pd.read_csv(
        TRAIN_FILE,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )

    return df


def load_adult_test():

    df = pd.read_csv(
        TEST_FILE,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
        skiprows=1,
    )

    return df


# =========================================================
# Clean target
# =========================================================

def clean_target(df):

    df = df.copy()

    # Adult test labels contain a trailing period:
    #
    # <=50K.
    # >50K.
    #
    # Remove it.

    df["income"] = (
        df["income"]
        .astype(str)
        .str.replace(
            ".",
            "",
            regex=False,
        )
        .str.strip()
    )

    valid_labels = {
        "<=50K",
        ">50K",
    }

    invalid = ~df["income"].isin(
        valid_labels
    )

    if invalid.any():

        raise ValueError(
            "Unexpected income labels: "
            f"{df.loc[invalid, 'income'].unique()}"
        )

    df["income"] = (
        df["income"]
        .map(
            {
                "<=50K": 0,
                ">50K": 1,
            }
        )
        .astype(np.int64)
    )

    return df


# =========================================================
# Main preprocessing
# =========================================================

def main():

    print("=" * 60)
    print("ADULT / CENSUS INCOME PREPROCESSING")
    print("=" * 60)

    print()
    print("Raw directory:")
    print(RAW_DIR)

    # -----------------------------------------------------
    # Load raw data
    # -----------------------------------------------------

    train_df = load_adult_train()
    test_df = load_adult_test()

    train_df = clean_target(
        train_df
    )

    test_df = clean_target(
        test_df
    )

    print()
    print(
        "Original training samples:",
        len(train_df),
    )

    print(
        "Official test samples:",
        len(test_df),
    )

    # -----------------------------------------------------
    # Separate X / y
    # -----------------------------------------------------

    X = train_df.drop(
        columns=["income"]
    )

    y = train_df["income"]

    X_test = test_df.drop(
        columns=["income"]
    )

    y_test = test_df["income"]

    # -----------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------

    X_meta_train, X_val, y_meta_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print()
    print(
        "Meta-training samples:",
        len(X_meta_train),
    )

    print(
        "Validation samples:",
        len(X_val),
    )

    # -----------------------------------------------------
    # Preprocessing
    #
    # IMPORTANT:
    # Fit ONLY on meta-training data.
    # -----------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    print()
    print("Fitting preprocessor...")

    preprocessor.fit(
        X_meta_train
    )

    # -----------------------------------------------------
    # Transform all splits
    # -----------------------------------------------------

    X_meta_train_processed = (
        preprocessor.transform(
            X_meta_train
        )
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # -----------------------------------------------------
    # Convert to float32
    # -----------------------------------------------------

    X_meta_train_processed = (
        np.asarray(
            X_meta_train_processed,
            dtype=np.float32,
        )
    )

    X_val_processed = (
        np.asarray(
            X_val_processed,
            dtype=np.float32,
        )
    )

    X_test_processed = (
        np.asarray(
            X_test_processed,
            dtype=np.float32,
        )
    )

    y_meta_train = np.asarray(
        y_meta_train,
        dtype=np.int64,
    )

    y_val = np.asarray(
        y_val,
        dtype=np.int64,
    )

    y_test = np.asarray(
        y_test,
        dtype=np.int64,
    )

    # -----------------------------------------------------
    # Print final dimensions
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("PROCESSED DATA")
    print("=" * 60)

    print(
        "Meta-training:",
        X_meta_train_processed.shape,
    )

    print(
        "Validation:",
        X_val_processed.shape,
    )

    print(
        "Test:",
        X_test_processed.shape,
    )

    print()
    print(
        "Meta-training labels:",
        np.bincount(y_meta_train).tolist(),
    )

    print(
        "Validation labels:",
        np.bincount(y_val).tolist(),
    )

    print(
        "Test labels:",
        np.bincount(y_test).tolist(),
    )

    # -----------------------------------------------------
    # Save NumPy arrays
    # -----------------------------------------------------

    np.save(
        OUTPUT_DIR / "train_x.npy",
        X_meta_train_processed,
    )

    np.save(
        OUTPUT_DIR / "train_y.npy",
        y_meta_train,
    )

    np.save(
        OUTPUT_DIR / "val_x.npy",
        X_val_processed,
    )

    np.save(
        OUTPUT_DIR / "val_y.npy",
        y_val,
    )

    np.save(
        OUTPUT_DIR / "test_x.npy",
        X_test_processed,
    )

    np.save(
        OUTPUT_DIR / "test_y.npy",
        y_test,
    )

    # -----------------------------------------------------
    # Save preprocessing pipeline
    # -----------------------------------------------------

    with open(
        OUTPUT_DIR / "preprocessor.pkl",
        "wb",
    ) as f:

        pickle.dump(
            preprocessor,
            f,
        )

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        "Saved to:",
        OUTPUT_DIR,
    )

    print()
    print("Files:")

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        print(
            " ",
            path.name,
        )


if __name__ == "__main__":
    main()