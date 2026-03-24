# CADSR — Complexity-Aware Deep Symbolic Regression

**Paper:** [Complexity-Aware Deep Symbolic Regression with Robust Risk-Seeking Policy Gradients](https://arxiv.org/abs/2406.06751)
**Authors:** Zachary Bastiani, Robert M. Kirby, Jacob Hochhalter, Shandian Zhe

---

## Overview

CADSR is a deep reinforcement learning approach to **symbolic regression** — the task of discovering a compact, interpretable mathematical expression that fits a dataset. Rather than returning a black-box model, CADSR searches over a space of expression trees and returns a human-readable equation.

CADSR builds on the Deep Symbolic Regression (DSR) framework and introduces three key contributions over prior work:

1. **Decoder-only Transformer with frequency-domain attention** — replaces the RNN backbone used in earlier DSR methods, with a dual-indexed positional encoding that enables layer-wise (rather than token-wise) expression tree generation.

2. **BIC-based reward function** — a Bayesian Information Criterion reward that automatically balances expression complexity against data fitness, removing the need to manually tune a complexity penalty.

3. **Ranking-based weighted policy update** — a risk-seeking policy gradient method that avoids the "tail barrier" problem (where near-zero probabilities can zero out gradients), leading to more stable and effective training.

---

## Repository Structure

```
CADSR/
├── main.py                        # Entry point — example training run
├── trainer.py                     # Core CADSR class: training loop, policy, saving
├── BTS_Transformer_model.py       # Autoregressive Transformer that samples expression trees
├── expression_tree.py             # ExpressionTree data structure and evaluation
├── expression_tree_helpers.py     # Utilities for expression tree manipulation
├── policies.py                    # Risk-seeking policy implementation
├── position_encodings.py          # Positional encoding schemes (TwoDPE, etc.)
├── enums.py                       # Enums for reward functions, policies, sampling methods
├── environment.yml                # Conda environment (Python 3.11, PyTorch 2.1, CUDA 12.1)
├── requirements.txt               # pip-frozen dependency list
├── Exploration.ipynb              # Notebook for analyzing CADSR output files
├── CADSR_settings/
│   └── CADSR_default_parameters.json   # Default hyperparameter configuration
└── feynman_I_47_23/               # Example dataset (from SRBench / Feynman benchmark)
```

---

## Requirements

- Python 3.11
- PyTorch 2.1 with CUDA 12.1 (**GPU strongly recommended** — roughly 10× faster than CPU)
- See `environment.yml` for the full Conda environment or `requirements.txt` for pip
- Environment was initially tested on Windows 11.

### Setup with Conda (recommended)

```bash
conda env create -f environment.yml
conda activate Symbolic_Regression
```

> **Note:** The `environment.yml` `prefix:` field points to an absolute path. You may need to remove or update that line before creating the environment on your machine.

### Setup with pip

```bash
pip install -r requirements.txt
```

---

## Quick Start

The included `main.py` runs CADSR on the provided Feynman benchmark example:

```bash
python main.py
```

This will:
1. Load `feynman_I_47_23/feynman_I_47_23.tsv.gz`
2. Split 80/20 into train and test sets
3. Train CADSR for 500 epochs with a batch size of 1000
4. Print progress and save results to a `.pkl` file in `../run_data/`

---

## Using CADSR in Your Own Code

```python
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from trainer import CADSR

# Load dataset (features as columns, target column named "target")
df = pd.read_csv("my_dataset.csv")
train_df, test_df = train_test_split(df, train_size=0.8)

# x: shape [n_features, n_samples], y: shape [n_samples]
x = train_df.drop(columns="target").to_numpy().T
y = train_df["target"].to_numpy()

# Load default hyperparameters
with open("CADSR_settings/CADSR_default_parameters.json") as f:
    kwargs = json.load(f)

kwargs["base_name"] = "my_experiment"

torch.manual_seed(42)
np.random.seed(42)

model = CADSR(n_features=len(x), **kwargs)
best_equation = model.train(x, y, epochs=500, batch=1000, verbose=True)

print("Best equation found:", best_equation)

# Optionally attach test data to the saved results
model.add_test_info(
    test_df.drop(columns="target").to_numpy().T,
    test_df["target"].to_numpy()
)
```

---

## Key Hyperparameters

All hyperparameters are configurable via `CADSR_settings/CADSR_default_parameters.json` or passed as keyword arguments to the `CADSR` constructor.

| Parameter | Default | Description |
|---|---|---|
| `reward_function` | `"NMSE"` | Scoring function. `"NMSE"` optimizes data fit; `"BIC"` automatically penalizes expression complexity |
| `risk_epsilon` | `0.05` | Fraction of sampled equations used for the policy update (top 5%). Lower values = more aggressive exploration |
| `save_ratio` | `0.0` | Fraction of historical best equations retained between epochs (Long-Term Short-Term Queue). Set to `1.0` to always carry forward the best-found trees |
| `max_depth` | `32` | Maximum number of tokens in an expression tree. Smaller values constrain complexity and speed up search |
| `entropy_coef` | `0.005` | Weight on entropy regularization. Controls exploration breadth; decrease if `max_depth` is increased |
| `equation_optimizer` | `"lm"` | Constant optimizer: `"lm"` (Levenberg–Marquardt, more accurate) or `"L-BFGS"` (faster) |
| `bic_scaler` | `1.0` | Scales the complexity penalty in the BIC reward. Increase to prefer shorter equations, decrease to allow more complex ones |
| `opt_const` | `true` | Whether to include an optimizable real-valued constant token in the operator library |
| `policy` | `"Basic"` | Policy gradient variant: `"Basic"` (NLL), `"PPO"`, or `"GRPO"` |
| `batch` | — | Equations sampled per epoch (set in `train()`). Larger batches improve exploration at the cost of runtime |

### Operators

By default CADSR uses:
- **Binary operators:** `+`, `-`, `*`, `/`, `**`
- **Unary operators:** `np.log`, `np.sin`, `np.cos`, `np.sqrt`, `np.exp`

Customize via `one_children_funcs` and `two_children_funcs` in the settings file or constructor kwargs.

---

## Output Files

Results are saved to `../run_data/<base_name>_<count>.pkl`. Load with:

```python
import pickle as pkl

with open("../run_data/my_experiment_0.pkl", "rb") as f:
    results = pkl.load(f)

print(results["Best Function"]["Equation"])
print(results["Best Function"]["Loss"])   # Reward score
print(results["Best Function"]["R2"])
```

The `Exploration.ipynb` notebook provides ready-to-use analysis code including reward-over-time plots and equation complexity distributions.

> **Note:** Enabling `save_eq_dict=True` stores every unique equation encountered during training. For 2000-epoch runs this can produce files exceeding 6 GB.

---

## Benchmarks

CADSR is evaluated on the [SRBench](https://github.com/cavalab/srbench) symbolic regression benchmark suite, which includes Feynman physics equations and other real-world regression problems. The included example dataset (`feynman_I_47_23`) corresponds to:

```
y = omega * sqrt(x1² + v²/omega²)
```

Ablation configurations for reproducing the paper's experiments are provided in `CADSR_settings/`.

---

## Citation

If you use CADSR in your research, please cite:

```bibtex
@article{bastiani2024cadsr,
  title   = {Complexity-Aware Deep Symbolic Regression with Robust Risk-Seeking Policy Gradients},
  author  = {Bastiani, Zachary and Kirby, Robert M. and Hochhalter, Jacob and Zhe, Shandian},
  journal = {arXiv preprint arXiv:2406.06751},
  year    = {2024},
  url     = {https://arxiv.org/abs/2406.06751}
}
```

---
Readme was made with support from Claude.ai.