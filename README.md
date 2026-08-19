# STUNT Reimplementation

A reproducible, from-scratch implementation of **STUNT** (*Self-supervised Tabular UNsupervised preTraining*) for few-shot learning on tabular data, with Prototypical Network meta-training, STUNT task generation, validation, and comparative evaluation against raw-feature baselines.

> **Paper:** Nam, J., et al. *"STUNT: Few-shot Tabular Learning with Self-generated Tasks from Unlabeled Tables."* ICLR 2023.

---

## Table of Contents

- [Overview](#overview)
- [Method Summary](#method-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
  - [WDBC (Breast Cancer)](#wdbc-breast-cancer)
  - [Adult / Census Income](#adult--census-income)
- [Evaluation Protocol](#evaluation-protocol)
- [Architecture Details](#architecture-details)
- [Tests](#tests)
- [Acknowledgements](#acknowledgements)

---

## Overview

STUNT addresses a core challenge in tabular machine learning: **how to learn useful representations when labeled data is extremely scarce** (1–10 labeled examples per class). It does this by generating synthetic few-shot classification tasks from *unlabeled* tabular data using feature masking and K-means clustering, then meta-training an encoder via Prototypical Networks on these self-generated tasks.

This repository is an **independent educational/research reproduction** — not the official implementation. It was built to:

1. Understand and verify the STUNT task-generation algorithm in detail.
2. Reproduce the meta-training pipeline end-to-end.
3. Evaluate STUNT embeddings against raw-feature baselines under controlled few-shot settings.

---

## Method Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     STUNT Task Generation                       │
│                                                                 │
│  Unlabeled table X  (N rows × D features)                       │
│         │                                                       │
│         ▼                                                       │
│  1. Sample mask ratio  p ~ Uniform(r₁, r₂)                     │
│  2. Select ⌊p·D⌋ feature columns at random                     │
│  3. K-means on selected columns → pseudo-labels (n_way classes) │
│  4. Permute selected columns (destroy label signal in input)    │
│  5. Split into support / query sets                             │
│         │                                                       │
│         ▼                                                       │
│  Synthetic few-shot task  (support_x, support_y, query_x, ...)  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Prototypical Network                          │
│                                                                 │
│  MLP Encoder → class prototypes (mean embeddings)               │
│  Classify queries by squared Euclidean distance to prototypes   │
│  Cross-entropy loss → backprop → update encoder                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Few-Shot Evaluation                            │
│                                                                 │
│  Freeze encoder → embed train/test data                         │
│  Sample k-shot support set → Logistic Regression → test set     │
│  Repeat 100 trials → report mean ± std                          │
│  Compare: STUNT embeddings  vs  raw features                    │
└─────────────────────────────────────────────────────────────────┘
```

**Key idea:** Pseudo-labels are derived from *original* feature values via K-means, but those same features are then *permuted* in the input, forcing the encoder to reconstruct the clustering structure from the remaining (unmasked) features. This creates a meaningful self-supervised pretext task without any human labels.

---

## Project Structure

```
stunt-reproduction/
│
├── src/                          # Core library
│   ├── data.py                   # Data loaders (WDBC, Income)
│   ├── encoder.py                # 2-layer MLP tabular encoder
│   ├── protonet.py               # Prototypical Network (single + batched)
│   ├── task_generator.py         # STUNT task generation pipeline
│   ├── train.py                  # Meta-training loop
│   ├── validation.py             # Pseudo-validation + real-label validation
│   ├── evaluate.py               # Downstream evaluation (embeddings + raw)
│   └── utils.py                  # Utility functions
│
├── scripts/                      # Runnable experiments
│   ├── train_wdbc.py             # Train STUNT on WDBC
│   ├── train_income.py           # Train STUNT on Adult/Income
│   ├── few_shot_evaluation.py    # WDBC few-shot eval (STUNT vs raw)
│   ├── few_shot_income.py        # Income few-shot eval (STUNT vs raw)
│   ├── preprocess_income.py      # Preprocess Adult/Census Income dataset
│   ├── baseline_wdbc.py          # WDBC supervised baselines
│   ├── evaluate_income.py        # Income downstream eval (STUNT)
│   ├── evaluate_income_raw.py    # Income downstream eval (raw features)
│   ├── train_demo.py             # Quick demo of the training loop
│   └── demo_task_generator.py    # Demo of STUNT task generation
│
├── tests/                        # Unit tests
│   ├── test_data.py
│   ├── test_protonet.py
│   ├── test_task_generator.py
│   └── test_validation.py
│
├── configs/
│   └── base.yaml                 # Default hyperparameters
│
├── data/                         # Dataset files (not tracked in git)
│   ├── raw/                      # Raw dataset files
│   └── income/                   # Preprocessed Income arrays
│
├── experiments/                  # Experiment outputs (not tracked)
├── results/                      # Training logs (not tracked)
├── models/                       # Saved model checkpoints (not tracked)
├── docs/
│   └── method.md                 # Method notes
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Harsh-sarawagi/STUNT-Reimplementation.git
cd STUNT-Reimplementation

# Create a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

| Package       | Purpose                              |
|---------------|--------------------------------------|
| `torch`       | Neural network training (MLP, ProtoNet) |
| `numpy`       | Array operations                     |
| `pandas`      | Data loading (Income dataset)        |
| `scikit-learn` | K-means, Logistic Regression, metrics |
| `scipy`       | Scientific computing utilities       |
| `matplotlib`  | Plotting evaluation results          |
| `seaborn`     | Visualization                        |
| `pyyaml`      | Configuration files                  |
| `pytest`      | Unit testing                         |

> **GPU support:** Install `torch` with CUDA for significantly faster training. The code automatically detects and uses CUDA when available.

---

## Datasets

### WDBC (Wisconsin Diagnostic Breast Cancer)

- **Samples:** 569 (357 benign, 212 malignant)
- **Features:** 30 real-valued
- **Source:** Place `wdbc.data` in `data/raw/`
- **Download:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Adult / Census Income

- **Samples:** 32,561 train + 16,281 test
- **Features:** 6 numerical + 8 categorical → 105 after one-hot encoding
- **Classes:** ≤50K (0) / >50K (1)
- **Source:** Place `adult.data` and `adult.test` in `data/raw/census+income/`
- **Download:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult)

**Preprocessing** (Income only):

```bash
python scripts/preprocess_income.py
```

This fits a `StandardScaler` + `OneHotEncoder` on the training split and saves `.npy` arrays to `data/income/`.

---

## Usage

### WDBC (Breast Cancer)

#### 1. Train STUNT encoder

```bash
python scripts/train_wdbc.py
```

Runs 500 meta-training episodes with batched STUNT task generation, periodic pseudo-validation, and saves the best model to `best_model.pt`.

#### 2. Few-shot evaluation (STUNT vs Raw)

```bash
python scripts/few_shot_evaluation.py
```

Evaluates at 1, 2, 5, and 10 shots per class across 100 trials. For each trial, the same support indices are used for both STUNT embeddings and raw features to ensure a fair comparison. Outputs results as JSON, CSV, plots, and a summary text file to `experiments/`.

---

### Adult / Census Income

#### 1. Preprocess

```bash
python scripts/preprocess_income.py
```

#### 2. Train STUNT encoder

```bash
python scripts/train_income.py
```

Runs 10,000 meta-training episodes. Saves the best model to `models/income_stunt_best.pt`.

#### 3. Few-shot evaluation (STUNT vs Raw)

```bash
python scripts/few_shot_income.py
```

Same protocol as WDBC: evaluates at 1, 2, 5, 10 shots × 100 trials with matched support sets.

#### 4. Full-data downstream evaluation

```bash
# STUNT embeddings
python scripts/evaluate_income.py

# Raw feature baseline
python scripts/evaluate_income_raw.py
```

---

## Evaluation Protocol

The few-shot evaluation follows this protocol:

1. **Freeze** the trained STUNT encoder.
2. **Embed** all train/test data through the encoder.
3. For each shot count `k ∈ {1, 2, 5, 10}`:
   - Repeat for 100 trials:
     - Sample `k` examples per class from the training set (the support set).
     - Train a **Logistic Regression** classifier on the support set.
     - Evaluate on the **full, fixed test set**.
     - Use the **exact same support indices** for both STUNT and raw-feature baselines.
   - Report `mean ± std` for accuracy and F1 score.
4. Compute the **gain** of STUNT over raw features for each shot count.

### Validation During Training

Two validation strategies are used during meta-training:

- **Pseudo-validation:** Generates synthetic episodes using K-means pseudo-labels on the validation set (no real labels needed). Used for model selection (best checkpoint).
- **Real validation:** Uses held-out real labels to form episodic evaluation tasks. Used for monitoring true generalization.

---

## Architecture Details

### Tabular Encoder

A 2-layer MLP adapted from the official STUNT `MLPProto` model:

```
Input (D features)
  → Linear(D, 1024) + ReLU
  → Linear(1024, 1024)
Output (1024-dim embedding)
```

### Prototypical Network

- Computes class **prototypes** as the mean embedding of support examples per class.
- Classifies query examples by **squared Euclidean distance** to prototypes.
- Supports both **single-task** and **batched-task** forward passes for efficient meta-training.

### STUNT Task Generator

| Parameter      | Default | Description                                    |
|----------------|---------|------------------------------------------------|
| `n_way`        | 2       | Number of pseudo-classes (K-means clusters)    |
| `k_shot`       | 2       | Support examples per class                     |
| `q_query`      | 5       | Query examples per class                       |
| `r1`           | 0.2     | Minimum feature masking ratio                  |
| `r2`           | 0.5     | Maximum feature masking ratio                  |
| `max_attempts` | 50      | Max retries for valid task generation           |

### Training Configuration

| Parameter          | WDBC   | Income  |
|--------------------|--------|---------|
| `n_way`            | 2      | 2       |
| `k_shot`           | 2      | 2       |
| `q_query`          | 5      | 5       |
| `episodes`         | 500    | 10,000  |
| `task_batch_size`   | 4      | 4       |
| `learning_rate`    | 1e-3   | 1e-3    |
| `weight_decay`     | 1e-5   | 1e-5    |
| `validation_interval` | 50  | 50      |

---

## Tests

Run the unit test suite:

```bash
pytest tests/ -v
```

Tests cover:
- **Data loading:** WDBC parsing and preprocessing
- **Task generator:** Mask sampling, pseudo-label generation, support/query splitting, edge cases
- **ProtoNet:** Single-task and batched forward/loss/predict
- **Validation:** Pseudo-validation and real-label validation episodes

---

## Acknowledgements

- **Original paper:** Nam, J., et al. *"STUNT: Few-shot Tabular Learning with Self-generated Tasks from Unlabeled Tables."* ICLR 2023. ([arXiv](https://arxiv.org/abs/2303.00918))
- **Prototypical Networks:** Snell, J., Swersky, K., & Zemel, R. *"Prototypical Networks for Few-shot Learning."* NeurIPS 2017.
- **Datasets:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/)

This is an independent educational/research reproduction, not the official STUNT implementation.
