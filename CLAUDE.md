# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and running

Conda env `iris_nn` (Python 3.11, PyTorch 2.9.1, numpy/pandas/matplotlib):

```bash
conda env create -f environment-mac.yml   # macOS, CPU only
conda env create -f environment.yml       # Linux, CUDA 12.8 (CPU alternative commented inside)
conda activate iris_nn
python main.py                            # run from repo root: DATA_CSV/RESULTS_DIR are relative paths
```

`config.TRAINER` picks the training method: `"de"` (default, writes `results_de/`) or `"backprop"` (writes `results/`). Derived constants (`RESULTS_DIR`, `ITERATION_NAME`, `ITERATION_START`) and the plot labels are computed from it at import time, so change it in `config.py`, not at runtime.

There is no test suite, linter, or build step. Correctness checks are `assert`s inside the pipeline (fold stratification, prediction counts).

## What this project is

A 4-7-3 multilayer perceptron for the Iris dataset (`iris.csv`, 150 rows, header row) trained either with **hand-written backpropagation** or with **Differential Evolution**, following Espinal et al., "Comparison of PSO and DE for training neural networks" (MICAI 2011). The point of the project is the manual math, so:

- Do not use `torch.autograd`, `torch.nn`, `torch.optim`, or `torch.sigmoid`. Tensors are just a matrix library here; `sigmoid` in `network.py` is a custom numerically stable version.
- Full-batch gradient descent, sigmoid on both layers, loss = `(1/(2N)) * sum((Y_hat - Y)^2)` against one-hot targets. Training stops when loss ≤ `EPSILON` or at `MAX_EPOCHS`.
- Bias is handled by appending a column of ones on the **right** (`data.add_bias`), so weight matrices include a bias row as their last row: `V` is input→hidden (5×7), `W` is hidden→output (8×3). `backward` drops the last (bias) column of `delta_k @ W.T` before propagating to the hidden layer.
- Weights are initialised with a CPU `torch.Generator` seeded per subexperiment, then moved to `DEVICE`, so results are reproducible across CPU/GPU.

All hyperparameters and layer sizes live in `config.py`; `INPUT_WITH_BIAS`/`HIDDEN_WITH_BIAS` must stay consistent with the base sizes.

## Differential Evolution training (`algorithm/`, `de_training.py`)

- `algorithm/` is a PyTorch port of `../de-algorithm/algorithm` (that project still uses the numpy original; don't edit it from here). Its scheme is kept as is: DE/rand/1 mutation `x_r1 + F(x_r2 − x_r3)`, binomial crossover with one forced mutant component, `repair_bounds` clipping, and greedy selection `f(x_t) ≤ f(x_i)`. Method and variable names match the original, but each generation is vectorized: all `P` trials come from generation `t`'s population and are evaluated in one call. The objective takes `(P×d)` and returns `(P,)`, and every tensor (population, bounds, `rng`) lives on `rng.device`. `best_fitness` stays a 0-d device tensor until `run()` ends, to avoid a GPU sync per generation.
- `config.DE_CONFIG` takes its hyperparameters from `../de-algorithm/config.py` (`F = 0.9`, `CR = 0.8`, `max_generations = 20000`, `tolerance = 1e-8`), except `population_size = 20` (as in the paper) and `dimension = 59`. Keep that mutation scheme and those values unless the user asks otherwise; the paper's DE/current-to-best/1 with `F = 0.15` was tried and rejected. The search bounds `DE_LOWER_BOUND`/`DE_UPPER_BOUND` (±10) come from neither source; trained backprop weights reach |w| ≈ 9.
- Genome = `[V.flatten() | W.flatten()]` (fig. 1 of the paper), 5·7 + 8·3 = 59 values (`decode_genome` / `encode_weights`).
- The fitness is the network's own error, `network.mse_loss`, computed for the whole population by `network_error_population` (it reuses `network.sigmoid`). The paper's eq. 9 is the same sum without `1/(2N)`, so DE selects the same individuals either way. `train_subexperiment_de` asserts that the best fitness equals `mse_loss(forward(...))` and returns the same `(V, W, losses, converged)` tuple as `train_subexperiment`, so everything downstream is shared. `losses` has `max_generations + 1` entries (generation 0 = best of the initial population), DE never stops early, and `converged` only means the best fitness ended ≤ `EPSILON`.
- On CPU, one DE subexperiment at 20,000 generations takes about 15 s, and the full run about 20 min. It runs on CUDA when available; `DEVICE` does not select MPS.

## Experimental protocol (spans experiment.py and main.py)

- 35 experiments, each a stratified 2-fold split (25 samples per class per fold, 75/75) built by `create_two_folds` and verified by `check_folds`.
- Each experiment has two subexperiments: train on A / test on B, then train on B / test on A. Each one retrains from fresh weights.
- Standardisation (mean / population std) is fit on the training fold only and applied to both folds, before adding the bias column.
- Prediction is `argmax` over the 3 outputs. Per-experiment confusion matrix = sum of both subexperiments (must total 150); global matrix sums all experiments (must total 5250). Rows = true class, columns = predicted.
- Experiment `e` uses seed `BASE_SEED + e` for its fold split and for the weight init or DE population. Its two subexperiments reuse that seed, so A→B and B→A start from the same initial weights or population.
- The "best model" is the subexperiment with the highest test accuracy, ties broken by lowest final training loss (`main.better`).

## Outputs (`utils/`)

`utils/` mirrors the `utils/` package of the sibling project `../rbf_kmeans/rbf_iris` (same plot palette/rcParams, same file layout), adapted to the MLP. The root modules hold the math and the protocol; `utils/` only reads their result dicts and writes files:

- `results.py`: per experiment `<RESULTS_DIR>/experiment_NN/` (split, aggregated confusion matrix, summaries) and per subexperiment `subexperiment_A_to_B|B_to_A/` (`epochs.csv` with an `epoch` or `generation` column, `summary.json`, `weights.pt`, `test_outputs.csv` with output and hidden activations, `plots/`). `sub1` is A→B, `sub2` is B→A.
- `global_results.py`: `<RESULTS_DIR>/global/` (`all_epochs.csv.gz`, `subexperiments.csv`, `experiments.csv`, `global_statistics.json`, `config.json`, plots) and `<RESULTS_DIR>/best_model/` (`model.pt` + `model.json` loadable with `model_io.load_model`). `save_global` rebuilds the per-subexperiment loss curves by cutting `all_epoch_losses` with each row's `epochs_executed`, so `main.py` must keep appending losses and rows in the same order.
- `network.MLPNetwork` (dataclass, like `RBFNetwork`) bundles `V`, `W` and the fold's `mu`/`std` so it takes raw `X`; the utils use it to recompute hidden activations for plots.
- `SAVE_SUBEXPERIMENT_PLOTS` in `config.py` turns off the 280 per-subexperiment PNGs.

A full backprop run takes about 2 minutes on CPU.

## Conventions

- Style follows `rbf_iris/rbf.py`: `from __future__ import annotations`, type hints on every signature, `@dataclass` for state, shape comments like `# (N×K) ... ej. (75×3)`.
- Don't change the logic or variable/function names in `network.py`, `experiment.py`, `data.py`, `main.py` or `algorithm/` when adding features: build on them from new modules such as `utils/` or `de_training.py`. Keep existing spellings such as `confussion_matrix_experimento` and `apply_standarizer`; the dict keys themselves are `"confusion_matrix"`.
- Comments, docstrings, log messages and assert messages are in Spanish; keep new ones in Spanish.
